# Zaratan ML Demo

A minimal PyTorch demo repository for learning how to use UMD Zaratan.

This project trains a simple neural network on the sklearn digits dataset
and saves:
- a model checkpoint
- evaluation metrics

It is meant for:
- learning GitHub -> Zaratan workflow
- learning Slurm job submission
- validating CPU and GPU execution

## Project structure

- `src/train.py` - trains the model and saves outputs
- `src/predict.py` - loads the saved model and runs one prediction
- `jobs/run_cpu.slurm` - CPU Slurm job
- `jobs/run_gpu.slurm` - GPU Slurm job

## Local usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --epochs 15 --batch-size 64 --output-dir outputs
python src/predict.py --checkpoint outputs/model.pt --index 0