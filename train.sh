#!/bin/bash

#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 2-00:00
#SBATCH -p cox
#SBATCH --gres=gpu:4
#SBATCH --mem=80000
#SBATCH -o scripts/_train_%j.%N.out # STDOUT
#SBATCH -e scripts/_train_%j.%N.err # STDERR

module load cuda/10.2.89-fasrc01 cudnn/7.6.5.32_cuda10.2-fasrc01 gcc/7.1.0-fasrc01
source activate detr

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --backbone resnet50 --ytvos_path /n/pfister_lab2/Lab/vcg_natural/YouTube-VIS/ --masks --pretrained_weights 384_coco_r50.pth --batch_size 16 --num_workers 4 --output_dir r50_proptr_twoway_360_36e_16b --epochs 36 --lr_drop 24 #  --resume r50_proptr_twoway_360_36e/checkpoint.pth
