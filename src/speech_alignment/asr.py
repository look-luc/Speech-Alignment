
"""Inference-only wrapper around the Qwen3 multilingual ASR model."""

import os

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM


MODEL_ID = os.environ.get(
    "ASR_MODEL_ID",
    "Qwen/Qwen3-ASR-0.6B-hf",
)


class ASR:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        dtype = torch.bfloat16 if use_cuda else torch.float32

        self.model = (
            AutoModelForMultimodalLM.from_pretrained(
                MODEL_ID,
                dtype=dtype,
            )
            .eval()
            .to(device)
        )

    @torch.inference_mode()
    def transcribe(self, audio_path: str, language=None):
        """Return (transcript, detected_language)."""

        inputs = self.processor.apply_transcription_request(
            audio=audio_path,
            language=language or None,
        )

        inputs = inputs.to(self.model.device, self.model.dtype)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        parsed = self.processor.decode(
            generated_ids,
            return_format="parsed",
        )[0]

        transcript = parsed["transcription"].strip()
        detected_language = parsed["language"]

        if not transcript:
            raise ValueError("ASR could not generate a transcript.")

        return transcript, detected_language
