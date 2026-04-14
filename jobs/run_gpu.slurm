#!/bin/bash
#SBATCH --job-name=digits_gpu_demo
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gpus=v100:1
#SBATCH --partition=gpu-v100
#SBATCH --output=logs/slurm-%j.out

# Change to submission directory
cd $SLURM_SUBMIT_DIR

# Just activate your venv - don't load system modules
source .venv/bin/activate

# Verify GPU is available
nvidia-smi

# Show PyTorch CUDA info
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Run your scripts
python src/train.py --epochs 10 --batch-size 128 --output-dir outputs
python src/predict.py --checkpoint outputs/model.pt --index 0 --output-file outputs/prediction.txt