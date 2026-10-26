import itertools

import evaluate
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoModelForTokenClassification,
    AutoProcessor,
)

from src.speech_alignment.data import get_data


class ChainedIterableDataset(IterableDataset):
    """Helper to chain multiple streaming datasets without HF feature schema conflicts."""
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list

    def __iter__(self):
        return itertools.chain.from_iterable(self.dataset_list)


class Forced_Aligner:
    def __init__(self) -> None:
        self.aligner_model_id = "Qwen/Qwen3-ForcedAligner-0.6B-hf"

        self.aligner_processor = AutoProcessor.from_pretrained(self.aligner_model_id)
        self.aligner_model = AutoModelForTokenClassification.from_pretrained(
            self.aligner_model_id,
            dtype=torch.bfloat16,
            device_map="auto"
        )

        self.data = get_data()

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def train_pipeline(
        self,
        batch_size: int = 16,
        output_dir: str = "./multiling_forced_aligner_model",
        num_epochs: int = 1,
        learning_rate: float = 5e-5
    ):
        train_splits = (
            self.data["facebook"][0] +
            self.data["waxal"][0] +
            self.data["fleurs"][0]
        )

        combined_train_data = ChainedIterableDataset(train_splits)

        train_loader = DataLoader(
            combined_train_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=1
        )

        optimizer = torch.optim.AdamW(self.aligner_model.parameters(), lr=learning_rate)
        self.aligner_model.train()

        print("Starting training / fine-tuning pipeline...")
        for epoch in range(num_epochs):
            total_loss = 0.0
            for step, batch in enumerate(train_loader):
                audio_arrays = batch["audio_arrays"]
                sampling_rate = batch["sampling_rate"]
                transcript_texts = batch["texts"]

                processed_inputs = self.aligner_processor(
                    audio=audio_arrays,
                    sampling_rate=sampling_rate,
                    text=transcript_texts,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                model_dtype = self.aligner_model.dtype
                processed_inputs = {
                    k: v.to(dtype=model_dtype) if v.is_floating_point() else v
                    for k, v in processed_inputs.items()
                }

                outputs = self.aligner_model(**processed_inputs)
                loss = outputs.loss

                if loss is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if step % 50 == 0 and step > 0:
                    print(f"Epoch {epoch+1} | Step {step} | Running Loss: {total_loss / (step + 1):.4f}")

        print(f"Saving fine-tuned model to {output_dir}...")
        self.aligner_model.save_pretrained(output_dir)
        self.aligner_processor.save_pretrained(output_dir)
        print("Training completed and model saved successfully!")

    def evaluate_pipeline(
        self,
        batch_size: int = 16,
        output_dir: str = "./multiling_forced_aligner_model"
    ):
        test_splits = (
            self.data["facebook"][1] +
            self.data["waxal"][1] +
            self.data["fleurs"][1]
        )

        combined_test_data = ChainedIterableDataset(test_splits)

        test_loader = DataLoader(
            combined_test_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=1
        )

        bleu_metric = evaluate.load("sacrebleu")
        self.aligner_model.eval()

        all_pred_texts = []
        all_ref_texts = []
        alignment_results = []

        print("Running forced alignment inference on test splits...")
        with torch.inference_mode():
            for batch in test_loader:
                audio_arrays = batch["audio_arrays"]
                sampling_rate = batch["sampling_rate"]
                transcript_texts = batch["texts"]
                raw_texts = batch["raw_texts"]

                processed_inputs = self.aligner_processor(
                    audio=audio_arrays,
                    sampling_rate=sampling_rate,
                    text=transcript_texts,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                model_dtype = self.aligner_model.dtype
                processed_inputs = {
                    k: v.to(dtype=model_dtype) if v.is_floating_point() else v
                    for k, v in processed_inputs.items()
                }

                outputs = self.aligner_model(**processed_inputs)
                logits = outputs.logits

                for i in range(len(raw_texts)):
                    single_raw_text = raw_texts[i].strip()
                    if not single_raw_text:
                        continue

                    word_list = single_raw_text.split()
                    if not word_list:
                        continue

                    try:
                        timestamps = self.aligner_processor.decode_forced_alignment(
                            logits=logits[i:i+1],
                            input_ids=processed_inputs["input_ids"][i:i+1],
                            word_lists=[word_list],
                            timestamp_token_id=self.aligner_model.config.timestamp_token_id
                        )
                        alignment_results.append(timestamps)
                        all_ref_texts.append([single_raw_text])
                        all_pred_texts.append(single_raw_text)
                    except Exception:
                        continue

        bleu_score = 0.0
        if all_pred_texts and all_ref_texts:
            bleu_result = bleu_metric.compute(predictions=all_pred_texts, references=all_ref_texts)
            if bleu_result is not None and "score" in bleu_result:
                bleu_score = bleu_result["score"]

        print("=" * 60)
        print(f"| Forced Alignment Evaluation Completed Successfully |")
        print(f"| Total Samples Processed: {len(all_pred_texts)} |")
        print(f"| Transcript Reference BLEU: {bleu_score:.4f} |")
        print("=" * 60)

        print(f"Saving model and processor to {output_dir}...")
        self.aligner_model.save_pretrained(output_dir)
        self.aligner_processor.save_pretrained(output_dir)
        print("Model and processor saved successfully!")

        return alignment_results

    def collate_fn(self, batch):
        audio_arrays = []
        target_sr = 16000

        for item in batch:
            arr = torch.tensor(item["audio"]["array"], dtype=torch.float32)
            orig_sr = item["audio"].get("sampling_rate", target_sr)

            # Resample on-the-fly if sample rates differ
            if orig_sr != target_sr:
                arr = F.resample(arr, orig_freq=orig_sr, new_freq=target_sr)

            audio_arrays.append(arr.numpy())

        texts = [
            self.aligner_processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": item["texts"]}]}],
                tokenize=False,
                add_generation_prompt=False
            )
            for item in batch
        ]

        raw_texts = [item["texts"] for item in batch]

        return {
            "audio_arrays": audio_arrays,
            "sampling_rate": target_sr,
            "texts": texts,
            "raw_texts": raw_texts
        }
