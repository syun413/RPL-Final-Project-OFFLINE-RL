# tdmpc2-jepa
## Setup
```
conda env create -f environment-new.yaml
conda activate tdmpc2-ms
pip install -r requirement.txt
wget https://huggingface.co/jonathanzkoch/vjepa-self-driving/resolve/main/jepa-latest.pth.tar
wget https://huggingface.co/jonathanzkoch/vjepa-self-driving/resolve/main/params-encoder.yaml

# Edit vjepa-encoder module
vim ~/miniconda3/envs/tdmpc2-ms/lib/python3.9/site-packages/vjepa_encoder/vision_encoder.py
# Remember to import os
# find dump = os.path.join(args['logging']['folder'], 'params-encoder.yaml')
# Add one line
os.makedirs(os.path.dirname(dump), exist_ok=True)
```

## Start training
Below are example commands for training:
```
CUDA_VISIBLE_DEVICES=1 python train.py buffer_size=500_000 steps=400_000 seed=1 exp_name=default env_id=PushCube-v1 env_type=gpu num_envs=32 control_mode=pd_ee_delta_pose obs=rgb save_video_local=false wandb=false
CUDA_VISIBLE_DEVICES=0 python train.py buffer_size=500_000 steps=400_000 seed=1 exp_name=default env_id=PickSingleYCB-v1 env_type=gpu num_envs=32 control_mode=pd_ee_delta_pose obs=rgb save_video_local=false wandb=true
```
