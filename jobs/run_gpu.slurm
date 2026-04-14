#!/bin/bash
#SBATCH --job-name=digits_gpu_demo
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/slurm-%j.out

# If you have multiple allocations, uncomment and set this:
# #SBATCH -A your_account_name

source /etc/profile
module load python

cd $SLURM_SUBMIT_DIR

source .venv/bin/activate

python src/train.py --epochs 15 --batch-size 128 --output-dir outputs
python src/predict.py --checkpoint outputs/model.pt --index 0 --output-file outputs/prediction.txt