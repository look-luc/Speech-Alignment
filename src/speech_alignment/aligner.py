"""Inference-only wrapper around the pretrained Qwen3 forced aligner.
Deliberately does NOT import data.py / peft / datasets, so the worker stays light."""
import os

import soundfile as sf
import torch
import torchaudio.functional as F
from transformers import AutoModelForTokenClassification, AutoProcessor

MODEL_ID = os.environ.get("ALIGNER_MODEL_ID", "Qwen/Qwen3-ForcedAligner-0.6B-hf")
TARGET_SR = 16000


def load_audio(path: str):
    """Any file soundfile can read -> mono float32 numpy array at 16 kHz."""
    wav, sr = sf.read(path)
    wav = torch.tensor(wav, dtype=torch.float32)
    if wav.ndim > 1:
        wav = wav.mean(dim=1)
    if sr != TARGET_SR:
        wav = F.resample(wav.unsqueeze(0), orig_freq=sr, new_freq=TARGET_SR).squeeze(0)
    return wav.numpy()


class Aligner:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        use_cuda = torch.cuda.is_available()
        self.model = (
            AutoModelForTokenClassification.from_pretrained(
                MODEL_ID, dtype=torch.bfloat16 if use_cuda else torch.float32
            )
            .eval()
            .to("cuda" if use_cuda else "cpu")
        )

    @torch.inference_mode()
    def align_raw(self, audio, transcript: str, language=None):
        inputs, word_lists = self.processor.prepare_forced_aligner_inputs(
            audio=audio, transcript=transcript, language=language, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        inputs = {
            k: v.to(dtype=self.model.dtype) if v.is_floating_point() else v
            for k, v in inputs.items()
        }
        inputs.pop("num_audio_tokens", None)  # returned by the processor, not accepted by forward()
        logits = self.model(**inputs).logits

        ts_id = getattr(self.model.config, "timestamp_token_id", None)
        if ts_id is None:
            ts_id = self.processor.tokenizer.convert_tokens_to_ids("<|notimestamp|>")

        return self.processor.decode_forced_alignment(
            logits=logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=ts_id,
        )

    def align(self, audio, transcript: str, language=None):
        """-> [{"word": str, "start": sec, "end": sec}, ...]"""
        raw = self.align_raw(audio, transcript, language)
        return [
            {"word": it["text"], "start": it["start_time"], "end": it["end_time"]}
            for it in raw[0]
        ]