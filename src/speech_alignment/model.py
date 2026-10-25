import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoModelForTokenClassification,
    AutoProcessor,
)

from src.speech_alignment.data import get_data


class Forced_Aligner:
    def __init__(self) -> None:
        # 1. Initialize the 1.7B ASR model and the 0.6B Aligner
        self.asr_model_id = "Qwen/Qwen3-ASR-1.7B-hf"
        self.aligner_model_id = "Qwen/Qwen3-ForcedAligner-0.6B-hf"

        # Processors handle feature extraction and tokenization
        self.asr_processor = AutoProcessor.from_pretrained(self.asr_model_id)
        self.aligner_processor = AutoProcessor.from_pretrained(self.aligner_model_id)

        # Load the 1.7B ASR model for transcription
        self.asr_model = AutoModelForMultimodalLM.from_pretrained(
            self.asr_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Load the Aligner model for timestamping
        self.aligner_model = AutoModelForTokenClassification.from_pretrained(
            self.aligner_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
