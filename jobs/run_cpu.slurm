#!/bin/bash
#SBATCH --job-name=digits_cpu_demo
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/slurm-%j.out

# DO NOT LOAD MODULES - they conflict with venv
# NO: source /etc/profile
# NO: module load python

# Just go to work directory and activate venv
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

# Run training
python src/train.py --epochs 10 --batch-size 64 --output-dir outputs
python src/predict.py --checkpoint outputs/model.pt --index 0 --output-file outputs/prediction.txt
