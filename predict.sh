flow_steps=1

python predict.py --mode transflow --input_csv splits/cameo2022.csv --weights epoch=1-step=2.ckpt --samples 1 --flow_steps $flow_steps --outpdb ./predict_pdb_transflow_flow_steps_$flow_steps --self_cond --resample