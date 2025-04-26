#!/bin/bash

#SBATCH --job-name=pcam-train-sota
#SBATCH --account=csci_ga_3033_109-2025sp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/scratch/aa12938/pcam-interpretability    
#SBATCH --mail-type=all
#SBATCH --mail-user=aa12938@nyu.edu

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "Current working directory: $(pwd)"
echo "Activating conda shell: /home/$USER/miniconda3"
source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate pcamenv
echo "Activated conda env: $(which python)"


echo "========== Starting ResNet-50 Training (batch size 32, LR 7.5e-5, patience 5) =========="
python train.py \
  --model_name resnet50 \
  --epochs 20 \
  --batch_size 32 \
  --lr 7.5e-5 \
  --optimizer adamw \
  --scheduler cosine \
  --weight_decay 0.05 \
  --warmup_epochs 3 \
  --patience 5 \
  --amp


echo "========== Starting DINO ViT-S/16 Training (batch size 352, LR 6.875e-4, patience 7) =========="
python train.py \
  --model_name dino-vits16 \
  --epochs 30 \
  --batch_size 352 \
  --lr 6.875e-4 \
  --optimizer adamw \
  --scheduler cosine \
  --weight_decay 0.05 \
  --warmup_epochs 5 \
  --patience 7 \
  --amp

echo "############### Finished Run: $(date +%Y-%m-%d_%H:%M:%S) ###############"
