import torch
import torchaudio
from datasets import Dataset, concatenate_datasets, load_dataset
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

def load_single_language_data(dataset_name=None, language=None, split="train", language_name=None):
    """
    Loads and standardizes dataset mapping and preprocessing
    for either WaxalNLP or FLEURS datasets.
    """
    lang = language or language_name or dataset_name
    if not lang:
        raise ValueError("Must provide a language or dataset name.")

    # 1. Check and Standardize Dataset Mapping for WaxalNLP
    if lang in WAXAL_LANG_CODE:
        lang_code = WAXAL_LANG_CODE[lang]
        dataset = load_dataset("google/WaxalNLP", f"{lang_code}_asr", split=split)

        if "__index_level_0__" in dataset.column_names:
            dataset = dataset.remove_columns(["__index_level_0__"])

        if "text" in dataset.column_names:
            dataset = dataset.rename_column("text", "transcription")

        if "locale" in dataset.column_names:
            dataset = dataset.rename_column("locale", "language")

        dataset = dataset.map(
            lambda example: {
                "audio": example["audio"],
                "transcription": example["transcription"],
                "language": example.get("language", lang),
                "source": "waxal"
            },
            remove_columns=[col for col in dataset.column_names if col not in ["audio", "transcription", "language", "source"]]
        )

    # 2. Check and Standardize Dataset Mapping for FLEURS
    elif lang in FLEURS_LOW_MED:
        lang_code = FLEURS_LOW_MED[lang]
        dataset = load_dataset("google/fleurs", lang_code, split=split)

        if "raw_transcription" in dataset.column_names and "transcription" not in dataset.column_names:
            dataset = dataset.rename_column("raw_transcription", "transcription")

        dataset = dataset.map(
            lambda example: {
                "audio": example["audio"],
                "transcription": example["transcription"],
                "language": lang,
                "source": "fleurs"
            },
            remove_columns=[col for col in dataset.column_names if col not in ["audio", "transcription", "language", "source"]]
        )
    else:
        # Fallback for custom or direct loading
        dataset = load_dataset(dataset_name or "google/fleurs", lang, split=split)
        if "raw_transcription" in dataset.column_names and "transcription" not in dataset.column_names:
            dataset = dataset.rename_column("raw_transcription", "transcription")
        if "text" in dataset.column_names and "transcription" not in dataset.column_names:
            dataset = dataset.rename_column("text", "transcription")

        dataset = dataset.map(
            lambda example: {
                "audio": example.get("audio"),
                "transcription": example.get("transcription", ""),
                "language": lang,
                "source": "custom"
            },
            remove_columns=[col for col in dataset.column_names if col not in ["audio", "transcription", "language", "source"]]
        )

    return dataset


class UnifiedSpeechDataset(TorchDataset):
    def __init__(self, hf_dataset: Dataset, processor=None, sampling_rate: int = 16000):
        self.dataset = hf_dataset
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        # Return raw dataset item so collate_fn can handle audio decoding, chunking, and text formatting.
        return self.dataset[idx]
