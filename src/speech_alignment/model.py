import io
import logging
import os

import evaluate
import soundfile as sf
import torch
import torchaudio.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMultimodalLM,
    AutoModelForTokenClassification,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

from src.speech_alignment.data import (
    UnifiedSpeechDataset,
    load_all_languages_dataset,
)

# Suppress harmless transformers processor warnings
logging.getLogger("transformers.processing_utils").setLevel(logging.ERROR)


class Forced_Aligner:
    def __init__(self, model_path: str = None) -> None:
        self.asr_model_id = "Qwen/Qwen3-ASR-1.7B-hf"
        print(f"Loading ASR processor and model: {self.asr_model_id}...")
        self.asr_processor = AutoProcessor.from_pretrained(self.asr_model_id)
        self.asr_model = AutoModelForMultimodalLM.from_pretrained(
            self.asr_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.asr_model.train()

        self.aligner_model_id = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
        print(f"Loading Aligner processor and model: {self.aligner_model_id}...")
        self.aligner_processor = AutoProcessor.from_pretrained(self.aligner_model_id)

        base_aligner_model = AutoModelForTokenClassification.from_pretrained(
            self.aligner_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        if model_path and os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print(f"Loading PEFT LoRA adapter from {model_path}...")
            self.aligner_model = PeftModel.from_pretrained(base_aligner_model, model_path)
        elif os.path.exists("./multiling_forced_aligner_trained/adapter_config.json"):
            print("Loading default local PEFT LoRA adapter from ./multiling_forced_aligner_trained...")
            self.aligner_model = PeftModel.from_pretrained(base_aligner_model, "./multiling_forced_aligner_trained")
        else:
            self.aligner_model = base_aligner_model

        self.aligner_model.train()

        if torch.cuda.is_available():
            self.device = "cuda"
            self.use_cuda = True
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.use_cuda = False
        else:
            self.device = "cpu"
            self.use_cuda = False

    def train_pipeline(
        self,
        output_dir: str = "./multiling_forced_aligner_trained",
        batch_size: int = 16,
        num_train_epochs: int = 1,
        learning_rate: float = 2e-5,
        split: str = "train"
    ):
        """
        Trains the ASR model first on ground-truth transcripts, ensuring it can
        generate valid text before downstream alignment.
        """
        print("\n=== Starting ASR Fine-Tuning Pipeline ===")

        asr_peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        self.asr_model = get_peft_model(self.asr_model, asr_peft_config)
        self.asr_model.print_trainable_parameters()

        try:
            raw_ds = load_all_languages_dataset(split=split)
            train_dataset = raw_ds.shuffle(seed=42)
        except Exception as e:
            print(f"Failed to load dataset for training: {e}")
            return None

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            fp16=False,
            bf16=self.use_cuda,
            use_cpu=not self.use_cuda,
            logging_steps=10,
            save_strategy="epoch",
            dataloader_num_workers=2,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.asr_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=self.asr_train_collate_fn,
        )

        print("Launching ASR PEFT training...")
        train_result = trainer.train()

        trainer.save_model(output_dir)
        self.asr_processor.save_pretrained(output_dir)
        print(f"ASR Training complete! Adapter saved to {output_dir}")

        return train_result

    def asr_train_collate_fn(self, batch):
        """Dedicated collator for ASR training using chat templates and ground-truth text targets."""
        audio_arrays = []
        texts = []
        target_sr = 16000

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
                if arr.ndim > 1:
                    arr = arr.mean(dim=0) if arr.shape[0] > arr.shape[1] else arr.mean(dim=1)
                arr = arr.squeeze()

                if orig_sr != target_sr:
                    arr = F.resample(arr.unsqueeze(0), orig_freq=orig_sr, new_freq=target_sr).squeeze(0)

                transcript = item.get("transcription", "").strip()
                if not transcript or arr.numel() < int(target_sr * 0.1):
                    continue

                audio_arrays.append(arr.numpy())
                texts.append(transcript)
            except Exception:
                continue

        if not audio_arrays:
            return None

        # Format inputs using chat template so audio placeholder tokens are correctly inserted
        asr_chat_texts = [
            self.asr_processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [{"type": "audio"}, {"type": "text", "text": "Use the languages in your training data to transcribe the audio into text using the Latin alphabet."}]
                    },
                    {
                        "role": "assistant",
                        "content": t
                    }
                ],
                tokenize=False,
                add_generation_prompt=False
            ) for t in texts
        ]

        asr_inputs = self.asr_processor(
            audio=audio_arrays,
            sampling_rate=target_sr,
            text=asr_chat_texts,
            padding=True,
            return_tensors="pt"
        )
        asr_labels = asr_inputs["input_ids"].clone()
        asr_pad_id = self.asr_processor.tokenizer.pad_token_id
        if asr_pad_id is not None:
            asr_labels[asr_labels == asr_pad_id] = -100

        input_features = asr_inputs.get("input_features")
        input_features_mask = asr_inputs.get("input_features_mask")
        if input_features_mask is None:
            input_features_mask = asr_inputs.get("feature_attention_mask")
        if input_features_mask is None and input_features is not None:
            input_features_mask = torch.ones(input_features.shape[0], input_features.shape[-1], dtype=torch.long)

        return {
            "input_ids": asr_inputs["input_ids"],
            "attention_mask": asr_inputs["attention_mask"],
            "labels": asr_labels,
            "input_features": input_features,
            "input_features_mask": input_features_mask
        }

    def collate_fn(self, batch):
        """Evaluation collator returning raw arrays and reference texts."""
        audio_arrays = []
        raw_texts = []
        target_sr = 16000

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
                if arr.ndim > 1:
                    arr = arr.mean(dim=0) if arr.shape[0] > arr.shape[1] else arr.mean(dim=1)
                arr = arr.squeeze()

                if orig_sr != target_sr:
                    arr = F.resample(arr.unsqueeze(0), orig_freq=orig_sr, new_freq=target_sr).squeeze(0)

                transcript = item.get("transcription", "").strip()
                if not transcript or arr.numel() < int(target_sr * 0.1):
                    continue

                audio_arrays.append(arr.numpy())
                raw_texts.append(transcript)
            except Exception:
                continue

        if not audio_arrays:
            return None

        return {
            "audio_arrays": audio_arrays,
            "sampling_rate": target_sr,
            "raw_texts": raw_texts
        }

    def evaluate_pipeline(
        self,
        batch_size: int = 16,
        split: str = "train",
        output_dir: str = "./multiling_forced_aligner_model"
    ):
        bleu_metric = evaluate.load("sacrebleu")
        self.asr_model.eval()
        self.aligner_model.eval()

        all_pred_texts = []
        all_ref_texts = []
        alignment_results = []

        print(f"\n=== Starting Two-Step Pipeline Evaluation ({split} split) ===")

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
                    print(f"ASR Generation Error at batch {step}: {e}")
                    predicted_texts = [""] * len(audio_arrays)

                # --- STEP 2: Forced Alignment (Using Ground-Truth Text Anchors) ---
                for i in range(len(audio_arrays)):
                    try:
                        # Safe lookups with index validation
                        pred_text = predicted_texts[i].strip() if i < len(predicted_texts) else ""
                        ref_text = raw_refs[i].strip() if i < len(raw_refs) else ""

                        if not ref_text:
                            continue

                        if pred_text:
                            all_pred_texts.append(pred_text)
                            all_ref_texts.append([ref_text])

                        word_list = ref_text.split()
                        if not word_list:
                            continue

                        aligner_input_text = self.aligner_processor.apply_chat_template(
                            [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": ref_text}]}],
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

                        if logits.shape[1] < aligner_inputs["input_ids"].shape[1]:
                            print(f"Warning: Logits shape mismatch for sample {i} in batch {step}")
                            continue

                        timestamps = self.aligner_processor.decode_forced_alignment(
                            logits=logits[0:1],
                            input_ids=aligner_inputs["input_ids"][0:1],
                            word_lists=[word_list],
                            timestamp_token_id=self.aligner_model.config.timestamp_token_id
                        )

                        # Guard against empty decoding outputs or out-of-bounds indexing
                        if not timestamps or len(timestamps) == 0:
                            print(f"Warning: Empty alignment timestamps for sample {i} in batch {step}")
                            continue

                        alignment_results.append({
                            "predicted_transcript": pred_text,
                            "reference_transcript": ref_text,
                            "timestamps": timestamps
                        })
                    except IndexError as ie:
                        print(f"IndexError caught for sample {i} in batch {step}: {ie}")
                        continue
                    except Exception as e:
                        print(f"Alignment Error for sample {i} in batch {step}: {e}")
                        continue

                if (step + 1) % 10 == 0:
                    print(f"Processed batch {step + 1}/{len(eval_loader)}")

        bleu_score = 0.0
        if all_pred_texts and all_ref_texts:
            bleu_result = bleu_metric.compute(predictions=all_pred_texts, references=all_ref_texts)
            if bleu_result is not None and "score" in bleu_result:
                bleu_score = bleu_result["score"]

        print("=" * 60)
        print("| Pipeline Evaluation Completed                           |")
        print(f"| Total Successfully Aligned Samples: {len(alignment_results)}".ljust(59) + "|")
        print(f"| ASR Transcript BLEU Score vs Reference: {bleu_score:.4f}".ljust(59) + "|")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)
        self.aligner_processor.save_pretrained(output_dir)
        print("Saved successfully!")

        return alignment_results
