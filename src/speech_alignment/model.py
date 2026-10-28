import gc
import io

import evaluate
import soundfile as sf
import torch
import torchaudio.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import ConcatDataset, DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoProcessor,
)

from src.speech_alignment.data import get_dataset_group


class Forced_Aligner:
    def __init__(self) -> None:
        self.aligner_model_id = "Qwen/Qwen3-ForcedAligner-0.6B-hf"

        self.aligner_processor = AutoProcessor.from_pretrained(self.aligner_model_id)

        base_model = AutoModelForTokenClassification.from_pretrained(
            self.aligner_model_id,
            dtype=torch.bfloat16,
            device_map="auto"
        )

        base_model.gradient_checkpointing_enable()

        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
        )
        self.aligner_model = get_peft_model(base_model, peft_config)
        self.aligner_model.print_trainable_parameters()

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def train_pipeline(
        self,
        batch_size: int = 1,
        output_dir: str = "./multiling_forced_aligner_model",
        num_epochs: int = 1,
        learning_rate: float = 5e-5
    ):
        optimizer = torch.optim.AdamW(self.aligner_model.parameters(), lr=learning_rate)
        self.aligner_model.train()

        # Removed "facebook" to focus exclusively on low/medium resource languages
        group_names = ["waxal", "fleurs"]

        print("Starting training pipeline with robust audio decoding...")
        for epoch in range(num_epochs):
            print(f"=== Epoch {epoch+1} / {num_epochs} ===")

            for group_name in group_names:
                print(f"Loading dataset group: {group_name}...")
                train_splits, _ = get_dataset_group(group_name)

                if not train_splits:
                    print(f"Warning: No train splits found for group {group_name}")
                    continue

                group_train_data = ConcatDataset(train_splits)

                train_loader = DataLoader(
                    group_train_data,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=self.collate_fn,
                    num_workers=4,
                    pin_memory=True
                )

                total_loss = 0.0
                step_count = 0

                for step, batch in enumerate(train_loader):
                    if batch is None:
                        if step < 5:
                            print(f"[{group_name}] Notice: Batch {step} returned None (skipped).")
                        continue

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
                        step_count += 1

                    if step % 5 == 0 and step_count > 0:
                        print(f"[{group_name}] Epoch {epoch+1} | Step {step} | Running Loss: {total_loss / step_count:.4f}")

                print(f"Finished group {group_name}. Total valid steps executed: {step_count}")
                del train_splits
                del group_train_data
                del train_loader
                gc.collect()
                torch.cuda.empty_cache()

        print(f"Saving fine-tuned model to {output_dir}...")
        self.aligner_model.save_pretrained(output_dir)
        self.aligner_processor.save_pretrained(output_dir)
        print("Training completed and model saved successfully!")

    def collate_fn(self, batch):
        audio_arrays = []
        raw_texts = []
        target_sr = 16000
        min_chunk_samples = int(target_sr * 1.0)

        for item in batch:
            try:
                audio_info = item.get("audio")
                if not audio_info:
                    continue

                audio_bytes = audio_info.get("bytes")
                audio_path = audio_info.get("path")

                if audio_bytes:
                    waveform_np, orig_sr = sf.read(io.BytesIO(audio_bytes))
                elif audio_path:
                    waveform_np, orig_sr = sf.read(audio_path)
                else:
                    continue

                arr = torch.tensor(waveform_np, dtype=torch.float32)

                if arr.ndim == 1:
                    arr = arr.unsqueeze(0)
                else:
                    arr = arr.transpose(0, 1)

                if arr.shape[0] > 1:
                    arr = arr.mean(dim=0, keepdim=True)
                arr = arr.squeeze(0)

                if orig_sr != target_sr:
                    arr = F.resample(arr.unsqueeze(0), orig_freq=orig_sr, new_freq=target_sr).squeeze(0)

            except Exception as e:
                print(f"Audio load exception encountered: {e}")
                continue

            transcript = item.get("texts", "")
            total_samples = arr.numel()

            words = transcript.split()
            total_words = len(words)
            if total_words == 0 or total_samples < min_chunk_samples:
                continue

            samples_per_word = total_samples / total_words

            current_start = 0
            current_word_idx = 0

            while current_start < total_samples and current_word_idx < total_words:
                remaining_words = total_words - current_word_idx
                estimated_words = max(1, min(remaining_words, 15))

                current_end = min(current_start + int(samples_per_word * estimated_words), total_samples)
                audio_chunk = arr[current_start:current_end]

                if audio_chunk.numel() < min_chunk_samples and current_end < total_samples:
                    current_end = min(current_start + min_chunk_samples, total_samples)
                    audio_chunk = arr[current_start:current_end]

                text_chunk = " ".join(words[current_word_idx : current_word_idx + estimated_words])

                if audio_chunk.numel() >= min_chunk_samples and len(text_chunk.strip()) > 0:
                    audio_arrays.append(audio_chunk.numpy())
                    raw_texts.append(text_chunk)

                current_start = current_end
                current_word_idx += estimated_words

        if not audio_arrays:
            return None

        texts = [
            self.aligner_processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": t}]}],
                tokenize=False,
                add_generation_prompt=False
            )
            for t in raw_texts
        ]

        return {
            "audio_arrays": audio_arrays,
            "sampling_rate": target_sr,
            "texts": texts,
            "raw_texts": raw_texts
        }

    def evaluate_pipeline(
        self,
        batch_size: int = 4,
        output_dir: str = "./multiling_forced_aligner_model"
    ):
        test_splits = []
        for group_name in ["waxal", "fleurs"]:
            _, t_splits = get_dataset_group(group_name)
            test_splits.extend(t_splits)

        combined_test_data = ConcatDataset(test_splits)

        test_loader = DataLoader(
            combined_test_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True
        )

        bleu_metric = evaluate.load("sacrebleu")
        self.aligner_model.eval()

        all_pred_texts = []
        all_ref_texts = []
        alignment_results = []

        print("Running forced alignment inference on test splits...")
        with torch.inference_mode():
            for batch in test_loader:
                if batch is None:
                    continue

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
