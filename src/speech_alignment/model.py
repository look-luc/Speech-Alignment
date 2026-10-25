import torch
import torch.optim as optim
from datasets import concatenate_datasets
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMultimodalLM,
    AutoModelForTokenClassification,
    AutoProcessor,
)

from src.speech_alignment.data import get_data


class Forced_Aligner:
    def __init__(self) -> None:
        self.asr_model_id = "Qwen/Qwen3-ASR-1.7B-hf"
        self.aligner_model_id = "Qwen/Qwen3-ForcedAligner-0.6B-hf"

        self.asr_processor = AutoProcessor.from_pretrained(self.asr_model_id)
        self.aligner_processor = AutoProcessor.from_pretrained(self.aligner_model_id)

        self.asr_model = AutoModelForMultimodalLM.from_pretrained(
            self.asr_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        self.aligner_model = AutoModelForTokenClassification.from_pretrained(
            self.aligner_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        self.data = get_data()
        self.device = "cuda" if torch.cuda.is_available else "mps" if torch.backends.mps.is_available() else "cpu"

    def train(
        self,
        batch_size:int=64,
        sampling_rate:int=16000,
        lr_asr:int|float=2e-5,
        lr_aligner:int|float=5e-5,
        weight_decay:float=0.01,
        num_epochs:int=2
    ):
        train_splits = [
            self.data["facebook"][0],
            self.data["waxal"][0],
            self.data["fleurs"][0]
        ]
        combined_train_data = concatenate_datasets(train_splits)
        dataloader = DataLoader(
            combined_train_data,
            batch_size=batch_size,
            shuffle=True,
        )
        torch.optim.AdamW([
            {"params": filter(lambda p: p.requires_grad, self.asr_model.parameters()), "lr": lr_asr},
            {"params": filter(lambda p: p.requires_grad, self.aligner_model.parameters()), "lr": lr_aligner}
        ], weight_decay=weight_decay)

        def alignment_collate_fn(batch):
            # Extract audio arrays and target text from the dataset batch items
            audio_arrays = [item["audio"]["array"] for item in batch]
            sampling_rates = [item["audio"]["sampling_rate"] for item in batch]

            # Depending on the dataset, text keys might be "text", "sentence", or "transcript"
            texts = [item.get("text") or item.get("sentence") or item.get("transcript") for item in batch]

            return {
                "audio_arrays": audio_arrays,
                "sampling_rates": sampling_rates,
                "texts": texts
            }

        for epoch in range(num_epochs):
            self.aligner_model.train()

            for batches in dataloader:
                audio_arrays = batches["audio_arrays"]
                transcript_texts = batches["texts"]
                sampling_rate = batches["sampling_rates"][0]
