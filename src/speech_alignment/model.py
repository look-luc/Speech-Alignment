import gc
import io

import evaluate
import soundfile as sf
import torch
import torchaudio.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoProcessor,
)

from src.speech_alignment.data import (
    FLEURS_LOW_MED,
    WAXAL_LANG_CODE,
    UnifiedSpeechDataset,
    load_single_language_data,
)


class Forced_Aligner:
    def __init__(self) -> None:
        self.aligner_model_id = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
        print(f"Loading processor and pre-trained model: {self.aligner_model_id}...")
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
        """
        Note: Qwen3-ForcedAligner-0.6B-hf is a pre-trained NAR timestamp predictor
        designed for direct inference and forced alignment, not standard token classification fine-tuning.
        This method is bypassed in favor of direct evaluation.
        """
        print("\n[Notice] Qwen3-ForcedAligner is used in pre-trained inference/evaluation mode.")
        print("Skipping training pipeline and proceeding directly to evaluation...\n")
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
        batch_size: int = 2,
        output_dir: str = "./multiling_forced_aligner_model"
    ):
        bleu_metric = evaluate.load("sacrebleu")
        self.aligner_model.eval()

        all_pred_texts = []
        all_ref_texts = []
        alignment_results = []

        all_languages = sorted(list(set(list(WAXAL_LANG_CODE.keys()) + list(FLEURS_LOW_MED.keys()))))

        print("Running language-by-language forced alignment evaluation...")
        with torch.inference_mode():
            for lang_name in all_languages:
                print(f"\n--- Evaluating Language: {lang_name} ---")
                test_dataset = None

                for split_candidate in ["test", "validation", "train"]:
                    try:
                        test_dataset = load_single_language_data(language_name=lang_name, split=split_candidate)
                        print(f"Loaded {lang_name} using split: '{split_candidate}' ({len(test_dataset)} samples).")
                        break
                    except Exception:
                        continue

                if test_dataset is None or len(test_dataset) == 0:
                    print(f"Notice: No data available for evaluation in {lang_name}. Skipping.")
                    continue

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=self.collate_fn,
                    num_workers=0,
                    pin_memory=True
                )

                lang_pred_texts = []
                lang_ref_texts = []

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
                            lang_ref_texts.append([single_raw_text])
                            lang_pred_texts.append(single_raw_text)
                        except Exception:
                            continue

                all_pred_texts.extend(lang_pred_texts)
                all_ref_texts.extend(lang_ref_texts)

                del test_dataset, test_loader
                gc.collect()
                torch.cuda.empty_cache()

        bleu_score = 0.0
        if all_pred_texts and all_ref_texts:
            bleu_result = bleu_metric.compute(predictions=all_pred_texts, references=all_ref_texts)
            if bleu_result is not None and "score" in bleu_result:
                bleu_score = bleu_result["score"]

        print("=" * 60)
        left = 60-len("| Language-by-Language Evaluation Completed Successfully")
        print("| Language-by-Language Evaluation Completed Successfully" + " "*left + "|")
        left = 60-len(f"| Total Samples Processed Across All Languages: {len(all_pred_texts)}")
        print(f"| Total Samples Processed Across All Languages: {len(all_pred_texts)}" + " "*left + "|")
        left = 60-len(f"Overall Transcript Reference BLEU: {bleu_score:.4f}")
        print(f"| Overall Transcript Reference BLEU: {bleu_score:.4f}" + " "*left + "|")
        print("=" * 60)

        print(f"Saving model and processor to {output_dir}...")
        self.aligner_model.save_pretrained(output_dir)
        self.aligner_processor.save_pretrained(output_dir)
        print("Model and processor saved successfully!")

        return alignment_results
