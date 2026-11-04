#!/bin/bash
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:a100_80gb:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=157G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/%u/Speech-Alignment/logs/%j.log
#SBATCH --job-name=forced-aligner-test
#SBATCH --account=ucb-general
#SBATCH --mail-type=END,FAIL

export PYTHONUNBUFFERED=1
cd /projects/$USER/Speech-Alignment

# Load environment variables from .env correctly supporting both HF_TOKEN and HUGGINGFACE_TOKEN
if [ -f .env ]; then
    export HF_TOKEN=$(grep -v '^#' .env | grep -E '^(HF_TOKEN|HUGGINGFACE_TOKEN)=' | head -n 1 | cut -d '=' -f2-)
fi

export SCRATCH="${SCRATCH:-/scratch/alpine/$USER}"
export HF_HOME="/scratch/alpine/$USER/hf_cache"
export HF_DATASETS_CACHE="/scratch/alpine/$USER/huggingface_cache/datasets"
export HF_HUB_CACHE="/scratch/alpine/$USER/huggingface_cache/hub"
export TRANSFORMERS_CACHE="/scratch/alpine/$USER/hf_cache/transformers"
export EVALUATE_CACHE_DIR="/scratch/alpine/$USER/hf_cache/evaluate"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$EVALUATE_CACHE_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export TMPDIR="/tmp"
export TMP="/tmp"
export TEMP="/tmp"

module purge
module load miniforge

eval "$(conda shell.bash hook)"

conda activate audio_env || echo "No virtual environment exists called audio_env"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "Starting Python execution via uv run..."
uv run run.py
