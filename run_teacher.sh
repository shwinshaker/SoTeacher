#!/bin/bash
set -e

env_name=SoTeacher
user_name=chengyu
source /home/"$user_name"/anaconda3/bin/activate "$env_name"

# -- Config --
## Vanilla training
# python config_teacher.py --model wrn_40_2 \
#                          --gpu 2 --trial 0 \
#                          --epochs 5  # for testing

## SoTeacher
python config_teacher.py --model wrn_40_2 \
                         --lip --lip_alpha 0.00001 \
                         --consist --consist_alpha 1.0 --consist_scheduler linear \
                         --gpu 2 --trial 0 \
                         --epochs 5  # for testing
[[ $? -ne 0 ]] && echo 'exit' && exit 2


# -- Run --
path=$(cat tmp/config.tmp | grep 'save_folder' | awk '{print$NF}' | tr -d '"')
cp tmp/config.tmp "$path"
python train_teacher.py  --config_file "$path"/config.tmp
