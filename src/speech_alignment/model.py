import gc
import io

import evaluate
import soundfile as sf
import torch
import torchaudio.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMultimodalLM,
    AutoModelForTokenClassification,
    AutoProcessor,
)

from src.speech_alignment.data import (
    UnifiedSpeechDataset,
    load_all_languages_dataset,
)


class Forced_Aligner:
    def __init__(self) -> None:
        self.asr_model_id = "Qwen/Qwen3-ASR-1.7B-hf"
        print(f"Loading ASR processor and model: {self.asr_model_id}...")
        self.asr_processor = AutoProcessor.from_pretrained(self.asr_model_id)
        self.asr_model = AutoModelForMultimodalLM.from_pretrained(
            self.asr_model_id,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.asr_model.eval()

        # Step 2: Forced Aligner Model for Timestamps
        self.aligner_model_id = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
        print(f"Loading Aligner processor and model: {self.aligner_model_id}...")
        self.aligner_processor = AutoProcessor.from_pretrained(self.aligner_model_id)
        self.aligner_model = AutoModelForTokenClassification.from_pretrained(
            self.aligner_model_id,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.aligner_model.eval()

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def train_pipeline(self, *args, **kwargs):
        print("\n[Notice] Two-Step Pipeline operates in pre-trained zero-shot inference mode.")
        print("Bypassing training pipeline; proceeding directly to ASR transcription and forced alignment...\n")
        return None

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

                waveform_np = None
                orig_sr = target_sr

                if isinstance(audio_info, dict):
                    if "array" in audio_info and audio_info["array"] is not None:
                        waveform_np = audio_info["array"]
                        orig_sr = audio_info.get("sampling_rate", 16000)
                    elif "bytes" in audio_info and audio_info["bytes"] is not None:
                        waveform_np, orig_sr = sf.read(io.BytesIO(audio_info["bytes"]))
                    elif "path" in audio_info and audio_info["path"] is not None:
                        waveform_np, orig_sr = sf.read(audio_info["path"])

                if waveform_np is None:
                    continue

                arr = torch.tensor(waveform_np, dtype=torch.float32)

                if arr.ndim == 1:
                    arr = arr.unsqueeze(0)
                elif arr.ndim == 2:
                    if arr.shape[0] > arr.shape[1]:
                        arr = arr.transpose(0, 1)

                if arr.shape[0] > 1:
                    arr = arr.mean(dim=0, keepdim=True)
                arr = arr.squeeze(0)

                if orig_sr != target_sr:
                    arr = F.resample(arr.unsqueeze(0), orig_freq=orig_sr, new_freq=target_sr).squeeze(0)

            except Exception as e:
                print(f"Audio load exception encountered: {e}")
                continue

            transcript = item.get("transcription", "")
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

        return {
            "audio_arrays": audio_arrays,
            "sampling_rate": target_sr,
            "raw_texts": raw_texts
        }

    def evaluate_pipeline(
        self,
        batch_size: int = 2,
        split: str = "train",
        output_dir: str = "./multiling_forced_aligner_model"
    ):
        bleu_metric = evaluate.load("sacrebleu")
        self.asr_model.eval()
        self.aligner_model.eval()

        all_pred_texts = []
        all_ref_texts = []
        alignment_results = []

        print(f"\n=== Starting Two-Step Pipeline (ASR -> Aligner) ({split} split) ===")

        try:
            raw_ds = load_all_languages_dataset(split=split)
            raw_ds = raw_ds.shuffle(seed=42)
            eval_dataset = UnifiedSpeechDataset(raw_ds, self.asr_processor)

            eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=0,
                pin_memory=True
            )
        except Exception as e:
            print(f"Failed to initialize evaluation dataset: {e}")
            return []

        print(f"Total batches to process: {len(eval_loader)}")

        with torch.inference_mode():
            for step, batch in enumerate(eval_loader):
                if batch is None:
                    continue

                audio_arrays = batch["audio_arrays"]
                sampling_rate = batch["sampling_rate"]
                raw_refs = batch["raw_texts"]

                # --- STEP 1: ASR Transcription ---
                try:
                    asr_inputs = self.asr_processor.apply_transcription_request(
                        audio=audio_arrays,
                        sampling_rate=sampling_rate
                    ).to(self.device)

                    asr_dtype = self.asr_model.dtype
                    asr_inputs = {
                        k: v.to(dtype=asr_dtype) if v.is_floating_point() else v
                        for k, v in asr_inputs.items()
                    }

                    asr_outputs = self.asr_model.generate(**asr_inputs, max_new_tokens=256)
                    generated_ids = asr_outputs[:, asr_inputs["input_ids"].shape[1]:]
                    predicted_texts = self.asr_processor.batch_decode(generated_ids, skip_special_tokens=True)
                except Exception as e:
                    continue

                # --- STEP 2: Forced Alignment ---
                for i in range(len(predicted_texts)):
                    pred_text = predicted_texts[i].strip()
                    ref_text = raw_refs[i].strip()
                    if not pred_text or not ref_text:
                        continue

                    word_list = pred_text.split()
                    if not word_list:
                        continue

                    try:
                        aligner_input_text = self.aligner_processor.apply_chat_template(
                            [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": pred_text}]}],
                            tokenize=False,
                            add_generation_prompt=False
                        )

                        aligner_inputs = self.aligner_processor(
                            audio=[audio_arrays[i]],
                            sampling_rate=sampling_rate,
                            text=[aligner_input_text],
                            padding=True,
                            return_tensors="pt"
                        ).to(self.device)

                        aligner_dtype = self.aligner_model.dtype
                        aligner_inputs = {
                            k: v.to(dtype=aligner_dtype) if v.is_floating_point() else v
                            for k, v in aligner_inputs.items()
                        }

                        outputs = self.aligner_model(**aligner_inputs)
                        logits = outputs.logits

                        timestamps = self.aligner_processor.decode_forced_alignment(
                            logits=logits[0:1],
                            input_ids=aligner_inputs["input_ids"][0:1],
                            word_lists=[word_list],
                            timestamp_token_id=self.aligner_model.config.timestamp_token_id
                        )

                        alignment_results.append({
                            "predicted_transcript": pred_text,
                            "reference_transcript": ref_text,
                            "timestamps": timestamps
                        })
                        all_pred_texts.append(pred_text)
                        all_ref_texts.append([ref_text])
                    except Exception:
                        continue

                if (step + 1) % 50 == 0:
                    print(f"Processed batch {step + 1}/{len(eval_loader)}")

        bleu_score = 0.0
        if all_pred_texts and all_ref_texts:
            bleu_result = bleu_metric.compute(predictions=all_pred_texts, references=all_ref_texts)
            if bleu_result is not None and "score" in bleu_result:
                bleu_score = bleu_result["score"]

        print("=" * 60)
        print("| Two-Step Pipeline (ASR + Forced Aligner) Completed        |")
        print(f"| Total Successfully Aligned Samples: {len(alignment_results)}".ljust(59) + "|")
        print(f"| ASR Transcript BLEU Score vs Reference: {bleu_score:.4f}".ljust(59) + "|")
        print("=" * 60)

        print(f"Saving aligner model and processor configuration to {output_dir}...")
        self.aligner_model.save_pretrained(output_dir)
        self.aligner_processor.save_pretrained(output_dir)
        print("Saved successfully!")

        return alignment_results
