import evaluate
import torch
import torch.optim as optim
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

    def train(
        self,
        batch_size: int = 16,
        sampling_rate: int = 16000,
        lr_aligner: float = 5e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 2,
        output_dir: str = "./multiling_forced_aligner_model"
    ):
        train_splits = [
            self.data["facebook"][0],
            self.data["waxal"][0],
            self.data["fleurs"][0]
        ]
        test_splits = [
            self.data["facebook"][1],
            self.data["waxal"][1],
            self.data["fleurs"][1]
        ]

        combined_train_data = concatenate_datasets(train_splits)
        combined_test_data = concatenate_datasets(test_splits)

        train_loader = DataLoader(
            combined_train_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0
        )
        test_loader = DataLoader(
            combined_test_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0
        )

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.aligner_model.parameters()),
            lr=lr_aligner,
            weight_decay=weight_decay
        )

        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        bleu_metric = evaluate.load("sacrebleu")

        for epoch in range(num_epochs):
            self.aligner_model.train()
            running_loss = 0.0
            step_count = 0

            for batch in train_loader:
                audio_arrays = batch["audio_arrays"]
                sampling_rate = batch["sampling_rate"]
                transcript_texts = batch["texts"]

                processed_inputs = self.aligner_processor(
                    audio=audio_arrays,
                    sampling_rate=sampling_rate,
                    text=transcript_texts,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                model_dtype = self.aligner_model.dtype
                processed_inputs = {
                    k: v.to(dtype=model_dtype) if v.is_floating_point() else v
                    for k, v in processed_inputs.items()
                }

                # Construct labels from input_ids and mask padding tokens with -100
                labels = processed_inputs["input_ids"].clone()
                pad_token_id = getattr(self.aligner_processor.tokenizer, "pad_token_id", -100)
                labels[labels == pad_token_id] = -100

                outputs = self.aligner_model(
                    **processed_inputs,
                    labels=labels
                )

                loss = outputs["loss"]
                running_loss += loss.item()
                step_count += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_train_loss = running_loss / step_count if step_count > 0 else 0.0
            print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")

            eval_metrics = self._evaluate(test_loader, accuracy_metric, f1_metric, bleu_metric)
            message = f'''| Validation Results -> Loss: {eval_metrics['loss']:.4f} | Accuracy: {eval_metrics['accuracy']:.4f} | F1: {eval_metrics['f1']:.4f} | BLEU: {eval_metrics['bleu']:.4f} |'''

            print("=" * len(message))
            print(message)
            print("=" * len(message))

        print(f"Saving fine-tuned aligner model and processor to {output_dir}...")
        self.aligner_model.save_pretrained(output_dir)
        self.aligner_processor.save_pretrained(output_dir)
        print("Model and processor saved successfully!")
        return self.aligner_model

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

    def _evaluate(self, test_loader, accuracy_metric, f1_metric, bleu_metric):
        self.aligner_model.eval()
        model_dtype = self.aligner_model.dtype
        total_loss = 0.0
        step_count = 0
        all_preds = []
        all_labels = []
        all_pred_texts = []
        all_ref_texts = []

        with torch.no_grad():
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

                processed_inputs = {
                    k: v.to(dtype=model_dtype) if v.is_floating_point() else v
                    for k, v in processed_inputs.items()
                }

                labels = processed_inputs["input_ids"].clone()
                pad_token_id = getattr(self.aligner_processor.tokenizer, "pad_token_id", -100)
                labels[labels == pad_token_id] = -100

                outputs = self.aligner_model(
                    **processed_inputs,
                    labels=labels
                )

                total_loss += outputs["loss"].item()
                step_count += 1

                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1).flatten().tolist()
                eval_labels = labels.flatten().tolist()

                for p, l in zip(preds, eval_labels):
                    if l != -100:
                        all_preds.append(p)
                        all_labels.append(l)

                all_ref_texts.extend([[t] for t in transcript_texts])
                all_pred_texts.extend(transcript_texts)

        avg_loss = total_loss / step_count if step_count > 0 else 0.0
        acc = accuracy_metric.compute(predictions=all_preds, references=all_labels)["accuracy"] if all_preds else 0.0
        f1 = f1_metric.compute(predictions=all_preds, references=all_labels, average="weighted", zero_division=0)["f1"] if all_preds else 0.0
        bleu = bleu_metric.compute(predictions=all_pred_texts, references=all_ref_texts)["score"] if all_pred_texts else 0.0

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "f1": f1,
            "bleu": bleu
        }
