import gc
import os

import pandas as pd
import torch
import torchaudio
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset as TorchDataset

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


def load_waxal_clean(language_subset: str = "ach_asr", split: str = "train") -> Dataset:
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
    for file_path in sorted(parquet_files):
        # Read into Pandas
        df = pd.read_parquet(file_path)

        if "__index_level_0__" in df.columns:
            df = df.drop(columns=["__index_level_0__"])

        if "text" in df.columns and "transcription" not in df.columns:
            df = df.rename(columns={"text": "transcription"})
        if "locale" in df.columns and "language" not in df.columns:
            df = df.rename(columns={"locale": "language"})

        ds = Dataset.from_pandas(df)
        datasets_list.append(ds)

        # Explicitly delete the intermediate dataframe and free memory
        del df
        gc.collect()

    combined_dataset = concatenate_datasets(datasets_list)

    # Clean up the list references
    del datasets_list
    gc.collect()

    return combined_dataset


def load_single_language_data(language_name: str, split: str = "train") -> Dataset:
    """
    Loads and harmonizes WaxalNLP and FLEURS data for a single specified language.
    """
    datasets_to_combine = []
    standard_columns = ["audio", "transcription", "language", "source"]

    # 1. Load WaxalNLP for this language if available
    if language_name in WAXAL_LANG_CODE:
        code = WAXAL_LANG_CODE[language_name]
        subset = f"{code}_asr"
        try:
            print(f"Loading WaxalNLP for {language_name} ({subset})...")
            waxal_ds = load_waxal_clean(language_subset=subset, split=split)
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


if __name__ == "__main__":
    combined_ds = load_all_speech_data(split="train")
    print("Sample 0 Source:", combined_ds[0]["source"])
    print("Sample -1 Source:", combined_ds[-1]["source"])
