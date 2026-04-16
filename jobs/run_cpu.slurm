#!/bin/bash
#SBATCH --job-name=digits_cpu_demo
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/slurm-%j.out

# Don't load python module - it pollutes PYTHONPATH
cd $SLURM_SUBMIT_DIR

# Use system python3 directly
/usr/bin/python3 src/train.py --epochs 10 --batch-size 64 --output-dir outputs
/usr/bin/python3 src/predict.py --checkpoint outputs/model.pt --index 0
