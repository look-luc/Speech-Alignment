import os

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchaudio
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset as TorchDataset


def load_waxal_clean(language_subset: str = "ewe_asr", split: str = "train") -> Dataset:
    """
    Custom loader for google/WaxalNLP to bypass upstream pandas index cast errors
    caused by rogue '__index_level_0__' columns in parquet shards.
    """
    repo_id = "google/WaxalNLP"
    repo_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")

    parquet_files = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(".parquet") and language_subset in root and split in file:
                parquet_files.append(os.path.join(root, file))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for subset '{language_subset}' and split '{split}'.")

    tables = []
    for file_path in sorted(parquet_files):
        table = pq.read_table(file_path)

        # Drop unexpected pandas index column if present upstream
        if "__index_level_0__" in table.column_names:
            table = table.drop(['__index_level_0__'])

        tables.append(table)

    combined_table = pa.concat_tables(tables)
    return Dataset(combined_table)


def load_combined_speech_data(
    waxal_subset: str = "ewe_asr",
    fleurs_lang: str = "ee_gh",
    split: str = "train"
) -> Dataset:
    """
    Single unified function to load and harmonize Google WaxalNLP and FLEURS datasets.
    Maps both datasets to a standardized schema: ['audio', 'transcription', 'language', 'source']
    """
    print(f"Loading WaxalNLP subset: {waxal_subset} ({split})...")
    waxal_ds = load_waxal_clean(language_subset=waxal_subset, split=split)

    print(f"Loading FLEURS language: {fleurs_lang} ({split})...")
    # FLEURS uses specific language codes like 'ee_gh' for Ewe (Ghana)
    fleurs_ds = load_dataset("google/fleurs", fleurs_lang, split=split)

    # 1. Harmonize WaxalNLP columns
    # Waxal typically has: audio, transcription, language, speaker_id, gender
    def map_waxal(batch):
        return {
            "audio": batch["audio"],
            "transcription": batch["transcription"],
            "language": batch.get("language", waxal_subset),
            "source": "waxal"
        }

    # 2. Harmonize FLEURS columns
    # FLEURS typically has: audio, transcription, raw_transcription, gender, lang_id, language
    def map_fleurs(batch):
        return {
            "audio": batch["audio"],
            "transcription": batch["transcription"],
            "language": fleurs_lang,
            "source": "fleurs"
        }

    # Apply transformations and keep only shared standardized columns
    standard_columns = ["audio", "transcription", "language", "source"]

    waxal_mapped = waxal_ds.map(
        map_waxal,
        remove_columns=[col for col in waxal_ds.column_names if col not in standard_columns]
    )

    fleurs_mapped = fleurs_ds.map(
        map_fleurs,
        remove_columns=[col for col in fleurs_ds.column_names if col not in standard_columns]
    )

    # Concatenate both datasets seamlessly
    combined_dataset = concatenate_datasets([waxal_mapped, fleurs_mapped])
    print(f"Successfully combined datasets! Total samples: {len(combined_dataset)}")

    return combined_dataset


class UnifiedSpeechDataset(TorchDataset):
    """
    PyTorch Dataset wrapper for the combined WaxalNLP + FLEURS dataset.
    Handles audio resampling, channel conversion, and processor feature extraction.
    """
    def __init__(self, hf_dataset: Dataset, processor, sampling_rate: int = 16000):
        self.dataset = hf_dataset
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        # Extract audio payload
        audio_data = item["audio"]
        if isinstance(audio_data, dict) and "path" in audio_data:
            speech_array, sr = torchaudio.load(audio_data["path"])
        elif isinstance(audio_data, str):
            speech_array, sr = torchaudio.load(audio_data)
        else:
            speech_array = torch.tensor(audio_data["array"], dtype=torch.float32)
            sr = audio_data["sampling_rate"]

        # Resample to target rate (e.g., 16kHz for Whisper/Qwen)
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
            speech_array = resampler(speech_array)

        # Convert multi-channel to mono
        if speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)

        # Extract features
        input_features = self.processor(
            speech_array.squeeze().numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).input_features[0]

        # Tokenize labels
        labels = self.processor.tokenizer(item["transcription"]).input_ids

        return {
            "input_features": input_features,
            "labels": torch.tensor(labels, dtype=torch.long),
            "source": item["source"]
        }


if __name__ == "__main__":
    # Test execution check
    combined_ds = load_combined_speech_data(waxal_subset="ewe_asr", fleurs_lang="ee_gh", split="train")
    print("Sample 0 Source:", combined_ds[0]["source"])
    print("Sample -1 Source:", combined_ds[-1]["source"])
