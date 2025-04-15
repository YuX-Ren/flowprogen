from flowprogen.utils.logging import get_logger
logger = get_logger(__name__)
import torch, os, wandb, time
import pandas as pd
from torch import nn
from .esmfold import ESMFold
from .alphafold import AlphaFold
from flowprogen.llmflow import LLMFlow
from flowprogen.transflow import TransFlow
from flowprogen.utils.loss import AlphaFoldLoss
from flowprogen.utils.diffusion import HarmonicPrior, rmsdalign, GaussianPrior
from flowprogen.utils import protein
from flowprogen.utils.misc import categorical_lddt, batch_encode_sequences, collate_dense_tensors
from openfold.utils.loss import lddt_ca
from openfold.utils.superimposition import superimpose
from openfold.utils.feats import pseudo_beta_fn
from openfold.data import data_transforms
from openfold.utils.exponential_moving_average import ExponentialMovingAverage

import pytorch_lightning as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_info
import numpy as np
from openfold.np import residue_constants
from openfold.utils.validation_metrics import (
    drmsd,
    gdt_ts,
    gdt_ha,
)
from openfold.utils.tensor_utils import (
    tensor_tree_map,
)
from collections import defaultdict
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler

def gather_log(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([l[key] for l in log_list], []) for key in log}
    return log

def get_log_mean(log):
    out = {}
    for key in log:
        try:
            out[key] = np.mean(log[key])
        except:
            pass
    return out


class ModelWrapper(pl.LightningModule):
    def _add_noise(self, batch):
        
        device = batch['aatype'].device
        batch_dims = batch['seq_length'].shape
        
        # noisy = self.harmonic_prior.sample(batch_dims)
        noisy = self.gaussian_prior.sample(batch_dims)
        try:
            noisy = rmsdalign(batch['pseudo_beta'], noisy, weights=batch['pseudo_beta_mask']).detach() # ?!?!
        except:
            logger.warning('SVD failed to converge!')
            batch['t'] = torch.ones(batch_dims, device=device)
            return
        
        t = torch.rand(batch_dims, device=device)
        noisy_beta = (1 - t[:,None,None]) * batch['pseudo_beta'] + t[:,None,None] * noisy
        
        pseudo_beta_dists = torch.sum((noisy_beta.unsqueeze(-2) - noisy_beta.unsqueeze(-3)) ** 2, dim=-1)**0.5
        batch['noised_pseudo_beta_dists'] = pseudo_beta_dists
        batch['t'] = t

    def disillation_training_step(self, batch):
        device = batch['aatype'].device
        batch_dims = batch['seq_length'].shape

        
        # orig_noisy = noisy = self.harmonic_prior.sample(batch_dims)
        orig_noisy = noisy = self.gaussian_prior.sample(batch_dims)
        schedule = np.linspace(1, 0, 11)

        orig_batch = {**batch}
        
        ## Forward pass of teacher model

        prev_outputs = None
        self.teacher.eval()
        with torch.no_grad():
            for t, s in zip(schedule[:-1], schedule[1:]):
                output = self.teacher(batch, prev_outputs=prev_outputs)
                pseudo_beta = pseudo_beta_fn(batch['aatype'], output['final_atom_positions'], None)
                noisy = rmsdalign(pseudo_beta, noisy)
                noisy = (s / t) * noisy + (1 - s / t) * pseudo_beta
                batch['noised_pseudo_beta_dists'] = torch.sum((noisy.unsqueeze(-2) - noisy.unsqueeze(-3)) ** 2, dim=-1)**0.5
                batch['t'] = torch.ones(batch_dims, device=noisy.device) * s
            if self.args.distill_self_cond:
                prev_outputs = output
                
        orig_batch['all_atom_positions'] = output['final_atom_positions']
        for t in [
            data_transforms.make_atom14_positions,
            data_transforms.atom37_to_frames,
            data_transforms.atom37_to_torsion_angles(""),
            data_transforms.make_pseudo_beta(""),
            data_transforms.get_backbone_frames,
            data_transforms.get_chi_angles,
        ]:
            orig_batch = t(orig_batch)

        orig_batch['noised_pseudo_beta_dists'] = torch.sum((orig_noisy.unsqueeze(-2) - orig_noisy.unsqueeze(-3)) ** 2, dim=-1)**0.5
        orig_batch['t'] = torch.ones(batch_dims, device=noisy.device)         
        
        student_output = self.model(orig_batch)
        loss, loss_breakdown = self.loss(student_output, orig_batch, _return_breakdown=True)

        with torch.no_grad():
            metrics = self._compute_validation_metrics(orig_batch, student_output, superimposition_metrics=False)
    
        for k, v in loss_breakdown.items():
            self.log(k, v.item())
        for k, v in metrics.items():
            self.log(k, v.item())

        self.log('dur', time.time() - self.last_log_time)
        self.last_log_time = time.time()
        return loss
        
    def training_step(self, batch, batch_idx, stage='train'):
        self.iter_step += 1
        device = batch["aatype"].device
        batch_size = batch['aatype'].shape[0]
        # self.harmonic_prior.to(device)
        self.gaussian_prior.to(device)
        self.stage = stage
        
        if not self.args.no_ema:
            if(self.ema.device != device):
                self.ema.to(device)
        
        if self.args.distillation:
            return self.distillation_training_step(batch)
        
        if torch.rand(1, generator=self.generator).item() < self.args.noise_prob:
            self._add_noise(batch)
            self.log(f'{self.stage}/time', float(batch['t'].mean().item()))
        else:
            self.log(f'{self.stage}/time', float(1.0))
        
        if self.args.extra_input:
            if torch.rand(1, generator=self.generator).item() < self.args.extra_input_prob:
                pass
            else:
                del batch['extra_all_atom_positions']

        
        # Ensure all tensors in batch require gradients
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dtype in [torch.float32, torch.float16, torch.float64]:
                batch[key] = value.requires_grad_(True)
        
        result = self.model.forward_modality(
            modalities=batch,
            times=None,
            modality_type=0,
            return_loss=True,
            return_loss_breakdown=True,
        )
        
        if isinstance(result, tuple) and len(result) == 2:
            outputs, (flow_loss, velocity_loss) = result
            loss, loss_breakdown = self.loss(outputs, batch, _return_breakdown=True)
            
            # Ensure losses are properly detached and have gradients
            flow_loss = flow_loss.detach().requires_grad_(True)
            velocity_loss = velocity_loss.detach().requires_grad_(True)
            loss = loss.detach().requires_grad_(True)
            
            with torch.no_grad():
                metrics = self._compute_validation_metrics(batch, outputs, superimposition_metrics=False)

            # added by hwxiao, use default log()
            self.log(f'{self.stage}/flow_loss', flow_loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=True)
            for k, v in loss_breakdown.items():
                self.log(f'{self.stage}/'+k, v.item(), prog_bar=True, on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=True)
            for k, v in metrics.items():
                self.log(f'{self.stage}/'+k, v.item(), prog_bar=True, on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=True)
        
            self.log(f'{self.stage}/dur', time.time() - self.last_log_time, prog_bar=True, on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=True)
            self.last_log_time = time.time()
            return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch['aatype'].shape[0]
        if not self.args.no_ema:
            if(self.cached_weights is None):
                self.load_ema_weights()
            
        if self.args.normal_validate:
            self.training_step(batch, batch_idx, 'val')
            if self.args.validate:
                self.try_print_log()
            return 
            
        self.iter_step += 1
        self.stage = 'val'
        # At the start of validation, load the EMA weights
        ref_prot = batch['ref_prot'][0]
        
        pred_prots = []
        for _ in range(self.args.val_samples):
            if self.args.distillation:
                prots = self.inference(batch, no_diffusion=True, noisy_first=True, as_protein=True)
            else:
                prots = self.inference(batch, as_protein=True)
            pred_prots.append(prots[-1])

        first_metrics = protein.global_metrics(ref_prot, prots[0])
        for key in first_metrics:
            self.log(f'{self.stage}/first_ref_'+key, first_metrics[key], prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)

        ref_metrics = []
        for pred_prot in pred_prots:
            ref_metrics.append(protein.global_metrics(ref_prot, pred_prot, lddt=True))

        self_metrics = []
        for i, pred_prot1 in enumerate(pred_prots):
            pred_prot2 = pred_prots[(i+1) % len(pred_prots)]
            self_metrics.append(protein.global_metrics(pred_prot1, pred_prot2, lddt=True))
        
        name_hash = hash(', '.join(batch['name']))
        self.log(f'{self.stage}/name', float(name_hash), batch_size=batch_size, sync_dist=True)
        
        ref_metrics = pd.DataFrame(ref_metrics)
        for key in ref_metrics:
            self.log(f'{self.stage}/mean_ref_'+key, ref_metrics[key].mean(), prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log(f'{self.stage}/max_ref_'+key, ref_metrics[key].max(), prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True) 
            self.log(f'{self.stage}/min_ref_'+key, ref_metrics[key].min(), prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True) 
        
        self_metrics = pd.DataFrame(self_metrics)
        for key in self_metrics:
            self.log(f'{self.stage}/self_'+key, self_metrics[key].mean(), prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        
        if self.args.validate:
            self.try_print_log()

    def restore_cached_weights(self):
        logger.info('Restoring cached weights')
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def load_ema_weights(self):
        # model.state_dict() contains references to model weights rather
        # than copies. Therefore, we need to clone them before calling 
        # load_state_dict().
        logger.info('Loading EMA weights')
        clone_param = lambda t: t.detach().clone()
        self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
        self.model.load_state_dict(self.ema.state_dict()["params"])
        
    def on_before_zero_grad(self, *args, **kwargs):
        if not self.args.no_ema:
            self.ema.update(self.model)

    def on_load_checkpoint(self, checkpoint):
        if 'distillation' not in self.args.__dict__:
            self.args.distillation = False
        if self.args.distillation:
            logger.info('Loading teacher model')
            def upgrade_state_dict(state_dict):
                import re
                """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
                prefixes = ["esmfold."]
                pattern = re.compile("^" + "|".join(prefixes))
                state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
                return state_dict
            try:
                self.teacher.load_state_dict(upgrade_state_dict(checkpoint['state_dict']))
                self.teacher.requires_grad_(False)
            except:
                logger.info('Loading teacher model failed, this is expected at distilled inference-time')                
            
        logger.info('Loading EMA state dict')
        if not self.args.no_ema:
            ema = checkpoint["ema"]
            self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        if self.cached_weights is not None:
            self.restore_cached_weights()
        if not self.args.no_ema:
            checkpoint["ema"] = self.ema.state_dict()
        
    def try_print_log(self):
        print_log = {}
        step = self.iter_step if self.args.validate else self.trainer.global_step 
        if (step + 1) % self.args.print_freq == 0:
            for k, v in self.trainer.logged_metrics.items():
                if k.endswith('_step'):
                    if isinstance(v, torch.Tensor):
                        print_log[k] = v.detach().cpu().numpy().tolist()
                    else:
                        print_log[k] = v

            print_log = gather_log(print_log, self.trainer.world_size)
            print_log = get_log_mean(print_log)
            print_log.update({'epoch': self.trainer.current_epoch, 'step': self.trainer.global_step})
            if self.trainer.is_global_zero:
                logger.info(f"print_log: {str(print_log)}")
                logger.info(f"print_log: {print_log}")
                if self.args.wandb:
                    wandb.log(print_log)
    
    def on_train_epoch_end(self):
        train_log = {}
        for k, v in self.trainer.logged_metrics.items():
            if k.startswith('train/'):
                if isinstance(v, torch.Tensor):
                    train_log[k] = v.detach().cpu().numpy().tolist()
                else:
                    train_log[k] = v
        train_log = gather_log(train_log, self.trainer.world_size)
        train_log = get_log_mean(train_log)
        train_log.update({
            'epoch': self.trainer.current_epoch,
            'step': self.trainer.global_step
        })
        
        if self.trainer.is_global_zero:
            logger.info(f"Train metrics: {train_log}")
            
            if self.args.wandb:
                wandb.log(train_log)
                
            path = os.path.join(
                os.environ["MODEL_DIR"], f"train_{self.trainer.current_epoch}.csv"
            )
            pd.DataFrame([train_log]).to_csv(path, index=False)
            
    def on_validation_epoch_end(self):
        if not self.args.no_ema:
            self.restore_cached_weights()
            
        # Get validation metrics from Lightning's callback metrics
        # which are collected during validation_step
        val_log = {}
        for k, v in self.trainer.callback_metrics.items():
            if k.startswith('val/'):
                if isinstance(v, torch.Tensor):
                    val_log[k] = v.detach().cpu().numpy().tolist()
                else:
                    val_log[k] = v
        
        # Gather logs from all processes if using distributed training
        val_log = gather_log(val_log, self.trainer.world_size)
        val_log = get_log_mean(val_log)

        if self.trainer.is_global_zero:
            logger.info(f"Validation metrics: {val_log}")
            
            if self.args.wandb:
                wandb.log(val_log)

            path = os.path.join(
                os.environ["MODEL_DIR"], f"val_{self.trainer.current_epoch}.csv"
            )
            pd.DataFrame([val_log]).to_csv(path, index=False)

    def on_before_optimizer_step(self, optimizer):
        self.try_print_log()
        if self.args.check_grad:
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    print(name)

    def inference(self, batch, as_protein=False, no_diffusion=False, self_cond=True, noisy_first=False, schedule=None):
        N = batch['aatype'].shape[1]
        device = batch['aatype'].device
        # prior = HarmonicPrior(N)
        prior = GaussianPrior(N)
        prior.to(device)
        noisy = prior.sample()

        sequences = batch["seqres"]
        if isinstance(sequences, str):
            sequences = [sequences]
        aatype, mask, _residx, linker_mask, chain_index = batch_encode_sequences(
            sequences, 512, "G" * 25
        )
        '''
        chain_index:
        tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        '''
        if not isinstance(_residx, torch.Tensor):
            _residx = collate_dense_tensors(_residx)

        aatype, mask, residx, linker_mask = map(
            lambda x: x.to(self.device), (aatype, mask, _residx, linker_mask)
        )

        if noisy_first:
            batch['noised_pseudo_beta_dists'] = torch.sum((noisy.unsqueeze(-2) - noisy.unsqueeze(-3)) ** 2, dim=-1)**0.5
            batch['t'] = torch.ones(1, device=noisy.device)
            
        if no_diffusion:
            print("no_diffusion!")
            # output = self.model(batch)
            s_s_0, s_z_0 = self.esm_model.get_encoder_outputs(batch)
            # s_s_0, s_z_0 = self.esm_model.get_encoder_outputs(
            #                 aatype=aatype,
            #                 mask=mask,
            #                 residx=residx,
            #                 masking_pattern=None,
            #                 num_recycles=4,
            # )
            output = self.esm_model.get_decoder_outputs(s_s_0, s_z_0, batch)
            output["atom37_atom_exists"] = output[
                    "atom37_atom_exists"
                ] * linker_mask.unsqueeze(2)
            # Expand plddt to match atom37_atom_exists dimensions and apply linker mask
            plddt_expanded = output["plddt"].unsqueeze(2).expand(-1, -1, 37) * linker_mask.unsqueeze(2)
            output["mean_plddt"] = (plddt_expanded * output["atom37_atom_exists"]).sum(dim=(1, 2)) / (output["atom37_atom_exists"].sum(dim=(1, 2)) + 1e-8)
            # output["chain_index"] = chain_index

            if as_protein:
                return protein.output_to_protein({**output, **batch})
            else:
                return [{**output, **batch}]

        if schedule is None:
            schedule = np.array([1.0, 0.75, 0.5, 0.25, 0.1, 0]) 

        outputs = []
        prev_outputs = None
        for t, s in zip(schedule[:-1], schedule[1:]):
            s_s_0, s_z_0 = self.esm_model.get_encoder_outputs(batch, prev_outputs=prev_outputs)
            output = self.esm_model.get_decoder_outputs(s_s_0, s_z_0, batch)
            pseudo_beta = pseudo_beta_fn(batch['aatype'], output['final_atom_positions'], None)
            outputs.append({**output, **batch})
            noisy = rmsdalign(pseudo_beta, noisy)
            noisy = (s / t) * noisy + (1 - s / t) * pseudo_beta
            batch['noised_pseudo_beta_dists'] = torch.sum((noisy.unsqueeze(-2) - noisy.unsqueeze(-3)) ** 2, dim=-1)**0.5
            batch['t'] = torch.ones(1, device=noisy.device) * s
            output["atom37_atom_exists"] = output[
                    "atom37_atom_exists"
                ] * linker_mask.unsqueeze(2)
            # Expand plddt to match atom37_atom_exists dimensions and apply linker mask
            plddt_expanded = output["plddt"].unsqueeze(2).expand(-1, -1, 37) * linker_mask.unsqueeze(2)
            output["mean_plddt"] = (plddt_expanded * output["atom37_atom_exists"]).sum(dim=(1, 2)) / (output["atom37_atom_exists"].sum(dim=(1, 2)) + 1e-8)
            # output["chain_index"] = chain_index
            if self_cond:
                prev_outputs = output

        del batch['noised_pseudo_beta_dists'], batch['t']
        if as_protein:
            prots = []
            for output in outputs:
                prots.extend(protein.output_to_protein(output))
            return prots
        else:
            return outputs
        
    def _compute_validation_metrics(self, 
        batch, 
        outputs, 
        superimposition_metrics=False
    ):
        metrics = {}
        
        gt_coords = batch["all_atom_positions"]
        pred_coords = outputs["final_atom_positions"]
        all_atom_mask = batch["all_atom_mask"]
    
        # This is super janky for superimposition. Fix later
        gt_coords_masked = gt_coords * all_atom_mask[..., None]
        pred_coords_masked = pred_coords * all_atom_mask[..., None]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
        all_atom_mask_ca = all_atom_mask[..., ca_pos]
    
        lddt_ca_score = lddt_ca(
            pred_coords,
            gt_coords,
            all_atom_mask,
            eps=1e-6,
            per_residue=False,
        )
   
        metrics["lddt_ca"] = lddt_ca_score
   
        drmsd_ca_score = drmsd(
            pred_coords_masked_ca,
            gt_coords_masked_ca,
            mask=all_atom_mask_ca, # still required here to compute n
        )
   
        metrics["drmsd_ca"] = drmsd_ca_score
    
        if(superimposition_metrics):
            superimposed_pred, alignment_rmsd = superimpose(
                gt_coords_masked_ca, pred_coords_masked_ca, all_atom_mask_ca,
            )
            gdt_ts_score = gdt_ts(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )
            gdt_ha_score = gdt_ha(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )

            metrics["alignment_rmsd"] = alignment_rmsd
            metrics["gdt_ts"] = gdt_ts_score
            metrics["gdt_ha"] = gdt_ha_score
    
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )
        
        lr_scheduler = AlphaFoldLRScheduler(optimizer, max_lr=self.args.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "AlphaFoldLRScheduler",
            }
        }

class TransFlowWrapper(ModelWrapper):
    def __init__(self, config, args, training=True):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = config
        self.args = args
        self.esm_model = ESMFold(config.model,
                extra_input=args and 'extra_input' in args.__dict__ and args.extra_input)
        if args.ckpt is None:
            rank_zero_info("Loading the model esmfold")
            path = "/share/project/xiaohongwang/Routine_ckpts/esm_pretrained_models/esmfold_3B_v1.pt"
            model_data = torch.load(path, weights_only=False)
            model_state = model_data["model"]
            self.esm_model.load_state_dict(model_state, strict=False)
            rank_zero_info("Model esmfold has been loaded")
        
        self.encoder = self.esm_model
        self.decoder = None
        self.model = TransFlow(
            num_text_tokens = 21,  # Number of amino acids
            dim_latent = 2176,
            channel_first_latent = False,  # Protein data is not channel-first
            modality_default_shape = (256, 256),  # Maximum sequence length
            modality_encoder = self.encoder,
            modality_decoder = self.encoder,
            add_pos_emb = True,
            modality_num_dim = 2,
            fallback_to_default_shape_if_invalid = True,
            reconstruction_loss_weight = 1.0,
            transformer = dict(
                dim = 128,
                depth = 4,
                dim_head = 32,
                heads = 4,
                attn_laser = True,
                use_flex_attn = True,
                use_gradient_checkpointing = True,
            )
        )
        
        if training:
            self.loss = AlphaFoldLoss(config.loss, esmfold=True)
            self.ema = ExponentialMovingAverage(
                model=self.model, decay=config.ema.decay
            )
            self.cached_weights = None
        
        self._log = defaultdict(list)

        # self.harmonic_prior = HarmonicPrior(config.data.train.crop_size)
        self.gaussian_prior = GaussianPrior(config.data.train.crop_size)
        self.generator = torch.Generator().manual_seed(137)
        self.last_log_time = time.time()
        self.iter_step = 0

    # def log(self, key, data):
    #     if isinstance(data, torch.Tensor):
    #         data = data.cpu().numpy()
    #     log = self._log
    #     # Convert scalar values to lists before extending
    #     if isinstance(data, (float, int, np.number)):
    #         # Convert numpy number types to Python float to avoid np.float64 in saved logs
    #         if isinstance(data, np.number):
    #             data = [float(data)]
    #         else:
    #             data = [data]
    #     elif isinstance(data, torch.Tensor):
    #         data = [float(data.cpu().numpy())]
    #     elif not isinstance(data, (list, np.ndarray)):
    #         data = [data]
        
    #     # Convert numpy values to Python float in lists/arrays
    #     if isinstance(data, (list, np.ndarray)):
    #         data = [float(x) if isinstance(x, np.number) else x for x in data]
            
    #     if self.stage == 'train' or self.args.validate:
    #         if "iter_" + key not in log:
    #             log["iter_" + key] = []
    #         log["iter_" + key].extend(data)
            
    #     if self.stage + "_" + key not in log:
    #         log[self.stage + "_" + key] = []
    #     log[self.stage + "_" + key].extend(data)

class LLMFlowWrapper(ModelWrapper):
    def __init__(self, cfg, args, training=True):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.args = args

        self.esm_model = ESMFold(cfg.model,
                extra_input=args and 'extra_input' in args.__dict__ and args.extra_input)
        if args.ckpt is None:
            rank_zero_info("Loading the model esmfold")
            path = "/share/project/xiaohongwang/Routine_ckpts/esm_pretrained_models/esm2_t33_650M_UR50D.pt"
            model_data = torch.load(path)
            model_state = model_data["model"]
            self.esm_model.load_state_dict(model_state, strict=False)
            rank_zero_info("Model esmfold has been loaded")
        
        self.evoformer_stack = self.esm_model.trunk
        self.structure_module = self.esm_model.trunk.structure_module
        self.model = LLMFlow(
            num_text_tokens = 21,  # Number of amino acids
            dim_latent = 21,  # Latent dimension for protein representation
            channel_first_latent = False,  # Protein data is not channel-first
            modality_default_shape = (512,),  # Maximum sequence length
            modality_encoder = self.evoformer_stack,
            modality_decoder = self.structure_module,
            pre_post_transformer_enc_dec = (
                nn.Linear(21, 2048),  # Adapt latent dimension to transformer dimension
                nn.Linear(2048, 21),
            ),
            add_pos_emb = True,  # Important for sequence data
            modality_num_dim = 1,  # 1D sequence data
            fallback_to_default_shape_if_invalid = True,
            reconstruction_loss_weight = 1.0,
            transformer={
                'use_llama': True,
                'dim': 2048,
                'model_name_or_path': '/share/project/xiaohongwang/LLM_checkpoints/Llama3.2/Llama-3.2-1B-Instruct',
                'use_gradient_checkpointing': True
            },
        )
        
        if not args.no_ema:
            self.ema = ExponentialMovingAverage(
                model=self.model, decay=cfg.ema.decay
            )
        
        self._log = defaultdict(list)

        # self.harmonic_prior = HarmonicPrior(cfg.data.train.crop_size)
        self.gaussian_prior = GaussianPrior(cfg.data.train.crop_size)
        self.generator = torch.Generator().manual_seed(137)
        self.last_log_time = time.time()
        self.iter_step = 0