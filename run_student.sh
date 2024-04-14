#!/bin/bash
set -e

env_name=SoTeacher
user_name=chengyu
source /home/"$user_name"/anaconda3/bin/activate "$env_name"

# -- Setup --
gpu=3
gamma=0.5
alpha=0.5
beta=0
distill_method=kd
teacher=wrn_40_2
student=wrn_40_1
teacher_name='cifar100_lr_0.05_decay_0.0005'  # vanilla teacher
# teacher_name=cifar100_lr_0.05_decay_0.0005_lip_alpha=1e-05_consist_alpha=1_linear  # SoTeacher
trial=0

# -- Config --
python config_student.py --path_t ./save/models/"$teacher"_"$teacher_name"_trial_"$trial"/"$teacher"_last.pth \
                         --distill $distill_method \
                         --model_s "$student" \
                         -r "$gamma" -a "$alpha" -b "$beta" \
                         --gpu "$gpu"
[[ $? -ne 0 ]] && echo 'exit' && exit 2

# -- Run --
path=$(cat tmp/config.tmp | grep 'save_folder' | awk '{print$NF}' | tr -d '"')
cp tmp/config.tmp $path
python train_student.py --config_file "$path/config.tmp"
