#!/bin/bash

#SBATCH -n 8
#SBATCH -N 1
#SBATCH -t 0-06:00
#SBATCH -p cox
#SBATCH --gres=gpu:4
#SBATCH --mem=80000
#SBATCH -o scripts/_test_%j.%N.out # STDOUT
#SBATCH -e scripts/_test_%j.%N.err # STDERR

module load cuda/10.2.89-fasrc01 cudnn/7.6.5.32_cuda10.2-fasrc01
module load gcc/7.1.0-fasrc01
source activate detr

# python inference.py --masks --model_path r50_proptr/checkpoint.pth --save_path output/results.json
python inference_prop_reduce.py --masks --model_path r50_proptr/checkpoint.pth --save_path output/results.json