#!/bin/bash
set -e

env_name=SoTeacher
user_name=chengyu
conda create -n "$env_name" python=3.9.12 -y
source /home/"$user_name"/anaconda3/bin/activate "$env_name"

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tensorflow==2.9.1  # for tensorboard support only
pip install tensorboard-logger==0.1.0

mkdir -p tmp
