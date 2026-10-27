#!/bin/bash
#SBATCH --partition=artxpro6000
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:rtx_pro_6000_2g.48gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:30:00
#SBATCH --output=/projects/%u/Speech-Alignment/logs/%j.log
#SBATCH --job-name=forced-aligner-test
#SBATCH --account=ucb-general
#SBATCH --mail-type=END,FAIL

export HF_TOKEN=$(grep -v '^#' .env | grep 'HUGGINGFACE_TOKEN=' | cut -d '=' -f2-)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export TMPDIR="/tmp"
export TMP="/tmp"
export TEMP="/tmp"

export SCRATCH="${SCRATCH:-/scratch/alpine/$USER}"

export HF_HOME="/scratch/alpine/$USER/hf_cache"
export HF_DATASETS_CACHE="/scratch/alpine/$USER/hf_cache/datasets"
export TRANSFORMERS_CACHE="/scratch/alpine/$USER/hf_cache/transformers"
export EVALUATE_CACHE_DIR="/scratch/alpine/$USER/hf_cache/evaluate"

export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$EVALUATE_CACHE_DIR"

cd /projects/$USER/Speech-Alignment

module purge
module load ffmpeg
module load uv

# Link NVIDIA NCCL from the .venv and let FFmpeg load its shared libraries via the module environment
export LD_LIBRARY_PATH=".venv/lib/python3.11/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"

uv sync

uv run run.py
