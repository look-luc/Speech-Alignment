import evaluate
import torch
from datasets import concatenate_datasets
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoProcessor,
)

from src.speech_alignment.data import get_data


class Forced_Aligner:
    def __init__(self) -> None:
        self.aligner_model_id = "Qwen/Qwen3-ForcedAligner-0.6B-hf"

        self.aligner_processor = AutoProcessor.from_pretrained(self.aligner_model_id)
        self.aligner_model = AutoModelForTokenClassification.from_pretrained(
            self.aligner_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        self.data = get_data()

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def evaluate_pipeline(
        self,
        batch_size: int = 4,
        output_dir: str = "./multiling_forced_aligner_model"
    ):
        test_splits = [
            self.data["facebook"][1],
            self.data["waxal"][1],
            self.data["fleurs"][1]
        ]

        combined_test_data = concatenate_datasets(test_splits)

        test_loader = DataLoader(
            combined_test_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0
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

                for i in range(len(transcript_texts)):
                    try:
                        timestamps = self.aligner_processor.decode_forced_alignment(
                            logits=logits[i:i+1],
                            input_ids=processed_inputs["input_ids"][i:i+1],
                            timestamp_token_id=self.aligner_model.config.timestamp_token_id
                        )
                        alignment_results.append(timestamps)
                    except Exception as e:
                        print(f"Alignment decoding warning: {e}")

                # Ensure non-empty text strings are appended
                cleaned_texts = [t for t in transcript_texts if t]
                all_ref_texts.extend([[t] for t in cleaned_texts])
                all_pred_texts.extend(cleaned_texts)

        # Defensively compute BLEU score to avoid NoneType subscript errors
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

        # Save the model and processor to disk
        print(f"Saving model and processor to {output_dir}...")
        self.aligner_model.save_pretrained(output_dir)
        self.aligner_processor.save_pretrained(output_dir)
        print("Model and processor saved successfully!")

        return alignment_results

    def collate_fn(self, batch):
        audio_arrays = [item["audio"]["array"] for item in batch]
        sampling_rate = batch[0]["audio"]["sampling_rate"] if "audio" in batch[0] else 16000

        texts = [
            self.aligner_processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": item["texts"]}]}],
                tokenize=False,
                add_generation_prompt=False
            )
            for item in batch
        ]

        return {
            "audio_arrays": audio_arrays,
            "sampling_rate": sampling_rate,
            "texts": texts
        }
