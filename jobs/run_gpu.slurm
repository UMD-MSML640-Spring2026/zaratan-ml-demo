#!/bin/bash
#SBATCH --job-name=digits_gpu_demo
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gpus=v100:1
#SBATCH --partition=gpu-v100
#SBATCH --output=logs/slurm-%j.out

# CRITICAL: Source profile first (enables module system)
source /etc/profile

# Load CUDA first for GPU support
module load cuda

# Load Python (don't specify hpcc/zaratan - it loads automatically)
module load python

# Show what's loaded (helpful for debugging)
module list

# Change to submission directory and activate venv
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

# Run your scripts
python src/train.py --epochs 10 --batch-size 128 --output-dir outputs
python src/predict.py --checkpoint outputs/model.pt --index 0 --output-file outputs/prediction.txt