from datasets import Audio, concatenate_datasets, load_dataset
from torch.utils.data import Dataset as TorchDataset

# Expanded Stratified Subset featuring Occitan, diverse Asian languages, and comprehensive African coverage
STRATIFIED_FLEURS = {
    # High-Resource Anchors
    "English": "en_us",
    "Spanish": "es_419",

    # European Romance & Regional Minority Language
    "occitan": "oc_fr",
    "French": "fr_fr",

    # Asian Languages (East Asian, Indic, & Austronesian)
    "Mandarin": "cmn_hans_cn",
    "Japanese": "ja_jp",
    "Hindi": "hi_in",
    "Kannada": "kn_in",
    "Filipino": "fil_ph",

    # African Languages (FLEURS Global tier)
    "Swahili": "sw_ke",
    "Hausa": "ha_ng",
    "Yoruba": "yo_ng",
}

STRATIFIED_WAXAL = {
    # Regional African Low-Resource (WaxalNLP tier)
    "Acholi": "ach",
    "Luganda": "lug",
    "Dagbani": "dag",
}

def load_single_language_data(dataset_name=None, language=None, split="train", language_name=None):
    """
    Loads and standardizes dataset mapping and preprocessing
    for either WaxalNLP or FLEURS datasets, with robust fallback for schema/CastErrors[cite: 26].
    """
    lang = language or language_name or dataset_name
    if not lang:
        raise ValueError("Must provide a language or dataset name.")

    if lang in STRATIFIED_WAXAL:
        lang_code = STRATIFIED_WAXAL[lang]
        try:
            dataset = load_dataset("google/WaxalNLP", f"{lang_code}_asr", split=split)
        except Exception as e:
            print(f"Encountered schema issue loading {lang} ({e}), attempting reload with verification bypass...")
            dataset = load_dataset("google/WaxalNLP", f"{lang_code}_asr", split=split, verification_mode="no_checks")

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

    elif lang in STRATIFIED_FLEURS:
        lang_code = STRATIFIED_FLEURS[lang]
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

    # Unify audio column features across all datasets so concatenation succeeds
    try:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    except Exception:
        pass

    return dataset

def load_all_languages_dataset(split="train"):
    """
    Loads, standardizes, and concatenates the stratified language subset
    into a single unified Hugging Face dataset for efficient evaluation.
    """
    all_languages = sorted(list(set(list(STRATIFIED_WAXAL.keys()) + list(STRATIFIED_FLEURS.keys()))))
    dataset_list = []

    print("Loading and combining datasets for stratified language subset...")
    for lang_name in all_languages:
        try:
            if lang_name in STRATIFIED_WAXAL:
                lang_code = STRATIFIED_WAXAL[lang_name]
            elif lang_name in STRATIFIED_FLEURS:
                lang_code = STRATIFIED_FLEURS[lang_name]
            ds = load_single_language_data(language_name=lang_code, split=split)
            if ds is not None and len(ds) > 0:
                dataset_list.append(ds)
                print(f"Successfully loaded {lang_name} ({len(ds)} samples).")
        except Exception as e:
            print(f"Skipping {lang_name} due to loading error: {e}")
            continue

    if not dataset_list:
        raise ValueError("No language datasets could be loaded successfully.")

    combined_dataset = concatenate_datasets(dataset_list)
    return combined_dataset


class UnifiedSpeechDataset(TorchDataset):
    def __init__(self, hf_dataset, processor=None, sampling_rate: int = 16000):
        self.dataset = hf_dataset
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[idx]
