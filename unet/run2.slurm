#!/bin/bash

#SBATCH --job-name=pcam-train
#SBATCH --account=csci_ga_3033_109-2025sp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=c24m170-a100-2
#SBATCH --open-mode=append
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2
#SBATCH --error=logs/err.err
#SBATCH --output=logs/out.out
#SBATCH --chdir=/home/spp9400/pcam-interpretability/unet    
#SBATCH --mail-type=all
#SBATCH --mail-user=spp9400@nyu.edu

# Create logs directory if it doesn't exist
mkdir -p logs

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Current working directory: $(pwd)"

# Activate conda environment
echo "Activating conda shell: /home/$USER/miniconda3"
source /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate unet
echo "Activated conda env: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"

# Run the first training script

# Run the second training script
echo "========== Starting DINO ViT-S/16 Training (batch size 352, LR 6.875e-4, patience 7) =========="
python train_unet1.py

# Check if the second script executed successfully
if [ $? -eq 0 ]; then
    echo "U-Net training completed successfully."
else
    echo "U-Net training ended with an error. Check logs for details."
    exit 1
fi

echo "############### Finished Run: $(date +%Y-%m-%d_%H:%M:%S) ###############"
