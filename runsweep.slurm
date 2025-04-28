#!/bin/bash

#SBATCH --job-name=pcam-train-sweep
#SBATCH --account=csci_ga_3033_109-2025sp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/scratch/aa12938/pcam-interpretability    
#SBATCH --mail-type=all
#SBATCH --mail-user=aa12938@nyu.edu

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "Current working directory: $(pwd)"
echo "Pulling latest changes from GitHub..."
git pull

echo "Activating conda shell: /home/$USER/miniconda3"
source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate pcamenv
echo "Activated conda env: $(which python)"

# DINO ViT-S/16 Sweep (batch_size=352, epochs=50, patience=10)

echo "========== ViT-V1: AdamW + Cosine, lr=7.5e-4, wd=0.04, warmup=10 =========="
python train.py \
  --model_name dino-vits16 \
  --epochs 50 \
  --batch_size 352 \
  --lr 7.5e-4 \
  --optimizer adamw \
  --scheduler cosine \
  --weight_decay 0.04 \
  --warmup_epochs 10 \
  --patience 10 \
  --amp \
  --run_name vit-v1

echo "========== ViT-V2: AdamW + Cosine, lr=1e-3, wd=0.02, warmup=15 =========="
python train.py \
  --model_name dino-vits16 \
  --epochs 50 \
  --batch_size 352 \
  --lr 1e-3 \
  --optimizer adamw \
  --scheduler cosine \
  --weight_decay 0.02 \
  --warmup_epochs 15 \
  --patience 10 \
  --amp \
  --run_name vit-v2

echo "========== ViT-V4: Adam + Cosine, lr=5e-4, wd=0.0 =========="
python train.py \
  --model_name dino-vits16 \
  --epochs 50 \
  --batch_size 352 \
  --lr 5e-4 \
  --optimizer adam \
  --scheduler cosine \
  --weight_decay 0.0 \
  --warmup_epochs 0 \
  --patience 10 \
  --amp \
  --run_name vit-v4

echo "========== ViT-V5: SGD + Cosine, lr=1e-2, wd=0.0, warmup=2 =========="
python train.py \
  --model_name dino-vits16 \
  --epochs 50 \
  --batch_size 352 \
  --lr 1e-2 \
  --optimizer sgd \
  --scheduler cosine \
  --weight_decay 0.0 \
  --warmup_epochs 2 \
  --patience 10 \
  --amp \
  --run_name vit-v5

echo "========== ViT-V8 (Baseline): Adam + None, lr=5e-4, no warmup, no wd =========="
python train.py \
  --model_name dino-vits16 \
  --epochs 50 \
  --batch_size 352 \
  --lr 5e-4 \
  --optimizer adam \
  --scheduler none \
  --weight_decay 0.0 \
  --warmup_epochs 0 \
  --patience 10 \
  --amp \
  --run_name vit-v8-baseline

echo "############### Sweep Complete: $(date +%Y-%m-%d_%H:%M:%S) ###############"

# ResNet-50 Sweep (batch_size=32, epochs=30, patience=7)

echo "========== ResNet-R1: AdamW + Cosine, lr=5e-4, wd=0.05, warmup=3 =========="
python train.py \
  --model_name resnet50 \
  --epochs 30 \
  --batch_size 32 \
  --lr 5e-4 \
  --optimizer adamw \
  --scheduler cosine \
  --weight_decay 0.05 \
  --warmup_epochs 3 \
  --patience 7 \
  --amp \
  --run_name resnet-r1

echo "========== ResNet-R2: AdamW + Cosine, lr=1e-3, wd=0.01, warmup=5 =========="
python train.py \
  --model_name resnet50 \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-3 \
  --optimizer adamw \
  --scheduler cosine \
  --weight_decay 0.01 \
  --warmup_epochs 5 \
  --patience 7 \
  --amp \
  --run_name resnet-r2

echo "========== ResNet-R3: Adam + Cosine, lr=3e-4, wd=0.0 =========="
python train.py \
  --model_name resnet50 \
  --epochs 30 \
  --batch_size 32 \
  --lr 3e-4 \
  --optimizer adam \
  --scheduler cosine \
  --weight_decay 0.0 \
  --warmup_epochs 0 \
  --patience 7 \
  --amp \
  --run_name resnet-r3

echo "========== ResNet-R4: SGD + Cosine, lr=1e-2, wd=5e-4 =========="
python train.py \
  --model_name resnet50 \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-2 \
  --optimizer sgd \
  --scheduler cosine \
  --weight_decay 5e-4 \
  --warmup_epochs 0 \
  --patience 7 \
  --amp \
  --run_name resnet-r4

echo "========== ResNet-R8 (Baseline): Adam + None, lr=5e-4, no warmup, no wd =========="
python train.py \
  --model_name resnet50 \
  --epochs 30 \
  --batch_size 32 \
  --lr 5e-4 \
  --optimizer adam \
  --scheduler none \
  --weight_decay 0.0 \
  --warmup_epochs 0 \
  --patience 7 \
  --amp \
  --run_name resnet-r8-baseline
