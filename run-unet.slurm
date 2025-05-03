#!/bin/bash

#SBATCH --job-name=pcam-unet-train
#SBATCH --account=csci_ga_3033_109-2025sp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=c24m170-a100-2
#SBATCH --open-mode=append
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/scratch/spp9400/pcam-interpretability    
#SBATCH --mail-type=all
#SBATCH --mail-user=spp9400@nyu.edu

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
# echo "Current working directory: $(pwd)"
# echo "Activating conda shell: /home/$USER/miniconda3"
# source /home/$USER/miniconda3/etc/profile.d/conda.sh
# echo "Activated conda env: $(which python)"

singularity exec --bind /scratch --nv --overlay /scratch/spp9400/overlay-25GB-500K.ext3:rw /scratch/spp9400/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate unet
python train_unet_pcam.py \
  --epochs 5 \
  --batch_size 64 \
  --save_path unet_batch1.pth
echo "========== Starting U-net training Training (batch size 32, LR 7.5e-5, patience 5) =========="

echo "############### Finished Run: $(date +%Y-%m-%d_%H:%M:%S) ###############"

