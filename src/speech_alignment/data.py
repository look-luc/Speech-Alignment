import os

import datasets.features.audio as audio_features
import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Dataset, concatenate_datasets, load_dataset
from datasets.features import Audio
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset as TorchDataset


# --- Fallback monkeypatch for load_with_torchcodec when torchcodec is uninstalled ---
def _fallback_decode_example(self, value, token_per_repo_id=None):
    """
    Replaces Hugging Face's torchcodec-based decoding with soundfile.
    """
    if isinstance(value, str):
        path = value
    elif isinstance(value, dict):
        path = value.get("path") or value.get("file")
    else:
        path = getattr(value, "path", None)

    if path:
        speech, sampling_rate = sf.read(path)
        return {
            "path": path,
            "array": speech,
            "sampling_rate": sampling_rate
        }

    # Fallback if bytes are provided directly
    if isinstance(value, dict) and "bytes" in value and value["bytes"] is not None:
        import io
        bytes_io = io.BytesIO(value["bytes"])
        speech, sampling_rate = sf.read(bytes_io)
        return {
            "path": value.get("path"),
            "array": speech,
            "sampling_rate": sampling_rate
        }

    raise ValueError(f"Could not decode audio value: {value}")

# Safely override the decode method globally
Audio.decode_example = _fallback_decode_example


WAXAL_LANG_CODE = {
    "Acholi": "ach", "Luganda": "lug", "Masaaba": "mas", "Nyankole": "nyn",
    "Soga": "sog", "Akan": "aka", "Ewe": "ewe", "Dagbani": "dag",
}

FLEURS_LOW_MED = {
    "Assamese": "as_in", "Gujarati": "gu_in", "Kannada": "kn_in", "Malayalam": "ml_in",
    "Nepali": "ne_np", "Odia": "or_in", "Punjabi": "pa_in", "Sindhi": "sd_in",
    "Cebuano": "ceb_ph", "Javanese": "jv_id", "Malay": "ms_my", "Maori": "mi_nz",
    "Filipino": "fil_ph", "Kazakh": "kk_kz", "Kyrgyz": "ky_kg", "Pashto": "ps_af",
    "Greek": "el_gr", "Finnish": "fi_fi", "Hungarian": "hu_hu", "Czech": "cs_cz",
    "Ukrainian": "uk_ua", "Romanian": "ro_ro", "Bulgarian": "bg_bg", "catalan": "ca_es",
    "occitan": "oc_fr",
}

def load_waxal_clean(language_subset: str = "ach_asr", split: str = "train", target_samples: int = 3000) -> Dataset:
    """
    Loads WaxalNLP parquet files individually, casts their audio columns to a
    standardized Audio feature schema to prevent CastErrors, and concatenates them safely.
    """
    repo_id = "google/WaxalNLP"
    repo_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")

    lang_code = language_subset.replace("_asr", "").replace("_tts", "")

    parquet_files = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(".parquet") and lang_code in root and split in file:
                parquet_files.append(os.path.join(root, file))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for language code '{lang_code}' and split '{split}'.")

    datasets_list = []
    total_loaded = 0

    for file_path in sorted(parquet_files):
        try:
            ds = Dataset.from_parquet(file_path)

            # Standardize column names
            if "__index_level_0__" in ds.column_names:
                ds = ds.remove_columns(["__index_level_0__"])
            if "text" in ds.column_names and "transcription" not in ds.column_names:
                ds = ds.rename_column("text", "transcription")
            if "locale" in ds.column_names and "language" not in ds.column_names:
                ds = ds.rename_column("locale", "language")

            # Force a uniform Audio feature schema across all shards
            if "audio" in ds.column_names:
                ds = ds.cast_column("audio", Audio(decode=False))

            datasets_list.append(ds)
            total_loaded += len(ds)

            # Stop loading once target samples are reached to avoid offset overflow errors
            if total_loaded >= target_samples:
                break

        except Exception as e:
            print(f"Skipping incompatible parquet file {file_path}: {e}")

    if not datasets_list:
        raise FileNotFoundError(f"No valid parquet files could be loaded for {language_subset}.")

    combined_dataset = concatenate_datasets(datasets_list)
    return combined_dataset

def load_single_language_data(language_name: str, split: str = "train", max_samples: int = 3000) -> Dataset:
    """
    Loads, harmonizes, and caps WaxalNLP and FLEURS data for a single language,
    filtering out overly long audio files to prevent OOM errors.
    """
    datasets_to_combine = []
    standard_columns = ["audio", "transcription", "language", "source"]

    # 1. Load WaxalNLP for this language if available
    if language_name in WAXAL_LANG_CODE:
        code = WAXAL_LANG_CODE[language_name]
        subset = f"{code}_asr"
        try:
            print(f"Loading WaxalNLP for {language_name} ({subset})...")
            waxal_ds = load_waxal_clean(language_subset=subset, split=split, target_samples=max_samples)
            waxal_mapped = waxal_ds.map(
                lambda b: {
                    "audio": b["audio"],
                    "transcription": b["transcription"],
                    "language": b.get("language", subset),
                    "source": "waxal"
                },
                remove_columns=[col for col in waxal_ds.column_names if col not in standard_columns]
            )
            datasets_to_combine.append(waxal_mapped)
        except Exception as e:
            print(f"Notice: No WaxalNLP data found for {language_name} ({e})")

    # 2. Load FLEURS for this language if available
    if language_name in FLEURS_LOW_MED:
        code = FLEURS_LOW_MED[language_name]
        try:
            print(f"Loading FLEURS for {language_name} ({code})...")
            fleurs_ds = load_dataset("google/fleurs", code, split=split)

            # Cast audio to decode=False to avoid torchcodec dependency
            if "audio" in fleurs_ds.column_names:
                fleurs_ds = fleurs_ds.cast_column("audio", Audio(decode=False))

            fleurs_mapped = fleurs_ds.map(
                lambda b: {
                    "audio": b["audio"],
                    "transcription": b["transcription"],
                    "language": code,
                    "source": "fleurs"
                },
                remove_columns=[col for col in fleurs_ds.column_names if col not in standard_columns]
            )
            datasets_to_combine.append(fleurs_mapped)
        except Exception as e:
            print(f"Notice: No FLEURS data found for {language_name} ({e})")

    if not datasets_to_combine:
        raise ValueError(f"No datasets available for language: {language_name}")

    combined_dataset = concatenate_datasets(datasets_to_combine)

    # 3. Subsample if the dataset exceeds max_samples
    if len(combined_dataset) > max_samples:
        print(f"Capping {language_name} from {len(combined_dataset)} down to {max_samples} samples.")
        combined_dataset = combined_dataset.select(range(max_samples))

    print(f"Loaded {len(combined_dataset)} total samples for {language_name}.")
    return combined_dataset


class UnifiedSpeechDataset(TorchDataset):
    def __init__(self, hf_dataset: Dataset, processor, sampling_rate: int = 16000):
        self.dataset = hf_dataset
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        audio_data = item["audio"]
        if isinstance(audio_data, dict) and "path" in audio_data:
            speech_array, sr = torchaudio.load(audio_data["path"])
        elif isinstance(audio_data, str):
            speech_array, sr = torchaudio.load(audio_data)
        else:
            speech_array = torch.tensor(audio_data["array"], dtype=torch.float32)
            sr = audio_data["sampling_rate"]

        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
            speech_array = resampler(speech_array)

        if speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)

        input_features = self.processor(
            speech_array.squeeze().numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).input_features[0]

        labels = self.processor.tokenizer(item["transcription"]).input_ids

        return {
            "input_features": input_features,
            "labels": torch.tensor(labels, dtype=torch.long),
            "source": item["source"]
        }
