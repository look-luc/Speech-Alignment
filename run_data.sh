#!/bin/bash

module load anaconda python uv
source .venv/bin/activate

export HF_HOME="/scratch/alpine/$USER/hf_cache"
export HF_DATASETS_CACHE="/scratch/alpine/$USER/hf_cache/datasets"

python src/speech_alignment/data.py
