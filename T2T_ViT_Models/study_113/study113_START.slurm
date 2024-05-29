#!/bin/bash

#SBATCH --job-name=SJC_T2TViT_study113
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem-bind=local
#SBATCH --time=8:00:00
#SBATCH --output=study113_epoch001.out
#SBATCH --error=study113_epoch001.err
#SBATCH -vvv

# Check available GPUs
sinfo  -o "%P %.24G %N"
srun /usr/bin/echo $CUDA_AVAILABLE_DEVICES

# Load correct conda environment
source /data/anaconda3b/bin/activate skylar-pytorch

# Get start time
export date00=`date`

# Start the Code
python ../train_T2T_ViT.py @study113_START.input

# Get end time and print to stdout
export date01=`date`

echo "===================TIME STARTED==================="
echo $date00
echo "===================TIME FINISHED==================="
echo $date01