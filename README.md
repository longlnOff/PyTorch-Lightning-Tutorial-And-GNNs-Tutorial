# PyTorch-Lightning-Tutorial-And-GNNs-Tutorial
- To train models, run the following command:
python3 Source/trainer.py fit --model model_name --config model_config.yaml --data.init_args.num_workers number_workers

- To test or validate models, run the following command:
python3 Source/trainer.py test --model model_name --config model_config.yaml --data.init_args.num_workers number_workers --ckpt_path path_checkpoint

# Link notes
https://www.notion.so/Hands-on-Graph-5ada478a3fbf496e9dd4a1621a22bbb0?pvs=4