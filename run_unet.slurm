#!/bin/bash

#SBATCH --job-name=pcam-train
#SBATCH --account=csci_ga_3033_109-2025sp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --error=unet/err.err
#SBATCH --output=unet/out.out
#SBATCH --chdir=/home/spp9400/pcam-interpretability    
#SBATCH --mail-type=all
#SBATCH --mail-user=spp9400@nyu.edu

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "Current working directory: $(pwd)"
echo "Activating conda shell: /home/$USER/miniconda3"
#source /home/$USER/miniconda3/etc/profile.d/conda.sh
# conda activate unet
echo "Activated conda env: $(which python)"


echo "========== Starting ResNet-50 Training (batch size 32, LR 7.5e-5, patience 5) =========="
python train_attnmaps.py


echo "========== Starting DINO ViT-S/16 Training (batch size 352, LR 6.875e-4, patience 7) =========="
python unet/train_unet.py

echo "############### Finished Run: $(date +%Y-%m-%d_%H:%M:%S) ###############"

