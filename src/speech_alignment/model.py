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

# Suppress the harmless transformers v5 processor kwargs warning log spam
logging.getLogger("transformers.processing_utils").setLevel(logging.ERROR)


class Forced_Aligner:
    def __init__(self, model_path: str = None) -> None:
        self.asr_model_id = "Qwen/Qwen3-ASR-1.7B-hf"
        print(f"Loading ASR processor and model: {self.asr_model_id}...")
        self.asr_processor = AutoProcessor.from_pretrained(self.asr_model_id)
        self.asr_model = AutoModelForMultimodalLM.from_pretrained(
            self.asr_model_id,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.asr_model.eval()

        self.aligner_model_id = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
        print(f"Loading Aligner processor and model: {self.aligner_model_id}...")
        self.aligner_processor = AutoProcessor.from_pretrained(self.aligner_model_id)

        # Load base aligner model
        base_aligner_model = AutoModelForTokenClassification.from_pretrained(
            self.aligner_model_id,
            dtype=torch.bfloat16,
            device_map="auto"
        )

        # If a trained LoRA adapter path or directory is provided/exists, load it via PEFT
        if model_path and os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print(f"Loading PEFT LoRA adapter from {model_path}...")
            self.aligner_model = PeftModel.from_pretrained(base_aligner_model, model_path)
        elif os.path.exists("./multiling_forced_aligner_trained/adapter_config.json"):
            print("Loading default local PEFT LoRA adapter from ./multiling_forced_aligner_trained...")
            self.aligner_model = PeftModel.from_pretrained(base_aligner_model, "./multiling_forced_aligner_trained")
        else:
            self.aligner_model = base_aligner_model

        self.aligner_model.eval()

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def train_pipeline(
        self,
        output_dir: str = "./multiling_forced_aligner_trained",
        batch_size: int = 4,
        num_train_epochs: int = 1,
        learning_rate: float = 2e-5,
        split: str = "train"
    ):
        """
        Implements a PEFT (LoRA) training pipeline for the forced aligner model using Hugging Face Trainer.
        """
        print("\n=== Starting PEFT (LoRA) Training Pipeline ===")

        # 1. Configure and apply PEFT LoRA wrappers
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        )

        # Wrap base model with PEFT
        self.aligner_model = get_peft_model(self.aligner_model, peft_config)
        self.aligner_model.print_trainable_parameters()

        # 2. Load dataset
        try:
            raw_ds = load_all_languages_dataset(split=split)
            train_dataset = raw_ds.shuffle(seed=42).select(range(min(10000, len(raw_ds))))
        except Exception as e:
            print(f"Failed to load dataset for training: {e}")
            return None

        # 3. Define training arguments optimized for stability and VRAM efficiency
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            fp16=False,
            bf16=True,  # Use bfloat16 precision
            logging_steps=10,
            save_strategy="epoch",
            dataloader_num_workers=2,
            remove_unused_columns=False,
        )

        # 4. Initialize Trainer
        trainer = Trainer(
            model=self.aligner_model,
            args=training_args,
            train_dataset=train_dataset,
        )

        print("Launching PEFT training...")
        train_result = trainer.train()

        # Save the trained LoRA adapter weights and processor configuration
        trainer.save_model(output_dir)
        self.aligner_processor.save_pretrained(output_dir)
        print(f"Training complete! PEFT model adapter saved to {output_dir}")

        return train_result

    def collate_fn(self, batch):
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

                # Standardize dimensions to 1D mono tensor
                if arr.ndim == 1:
                    arr = arr.unsqueeze(0)
                elif arr.ndim == 2:
                    if arr.shape[0] > arr.shape[1]:
                        arr = arr.transpose(0, 1)

                if arr.shape[0] > 1:
                    arr = arr.mean(dim=0, keepdim=True)
                arr = arr.squeeze(0)

                # Resample if necessary
                if orig_sr != target_sr:
                    arr = F.resample(arr.unsqueeze(0), orig_freq=orig_sr, new_freq=target_sr).squeeze(0)

                transcript = item.get("transcription", "").strip()

                if not transcript or arr.numel() < int(target_sr * 0.1):
                    continue

                audio_arrays.append(arr.numpy())
                raw_texts.append(transcript)

            except Exception as e:
                print(f"Audio load exception encountered: {e}")
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
        batch_size: int = 4,
        split: str = "train",
        output_dir: str = "./multiling_forced_aligner_model"
    ):
        bleu_metric = evaluate.load("sacrebleu")
        self.asr_model.eval()
        self.aligner_model.eval()

        all_pred_texts = []
        all_ref_texts = []
        alignment_results = []

        print(f"\n=== Starting Two-Step Pipeline with PEFT Aligner (ASR -> Aligner) ({split} split) ===")

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
                    except Exception as e:
                        if len(alignment_results) == 0 and step == 0:
                            print(f"Alignment error debug: {e}")
                        continue

                if (step + 1) % 50 == 0:
                    print(f"Processed batch {step + 1}/{len(eval_loader)}")

        bleu_score = 0.0
        if all_pred_texts and all_ref_texts:
            bleu_result = bleu_metric.compute(predictions=all_pred_texts, references=all_ref_texts)
            if bleu_result is not None and "score" in bleu_result:
                bleu_score = bleu_result["score"]

        print("=" * 60)
        print("| Two-Step Pipeline (ASR + PEFT Aligner) Completed          |")
        print(f"| Total Successfully Aligned Samples: {len(alignment_results)}".ljust(59) + "|")
        print(f"| ASR Transcript BLEU Score vs Reference: {bleu_score:.4f}".ljust(59) + "|")
        print("=" * 60)

        print(f"Saving aligner model and processor configuration to {output_dir}...")
        self.aligner_model.save_pretrained(output_dir)
        self.aligner_processor.save_pretrained(output_dir)
        print("Saved successfully!")

        return alignment_results
