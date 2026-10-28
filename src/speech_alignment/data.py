from datasets import load_dataset

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

def get_dataset_group(group_name: str):
    """Loads and returns train/test splits using official Hugging Face dataset structures."""
    if group_name == "waxal":
        train_splits, test_splits = [], []
        for code in WAXAL_LANG_CODE.values():
            # Load Waxal ASR dataset configuration
            train_splits.append(
                load_dataset("google/WaxalNLP", f"{code}_asr", split="train", trust_remote_code=True)
                .select_columns(["audio", "transcription"]).rename_column("transcription", "texts")
            )
            test_splits.append(
                load_dataset("google/WaxalNLP", f"{code}_asr", split="test", trust_remote_code=True)
                .select_columns(["audio", "transcription"]).rename_column("transcription", "texts")
            )
        return train_splits, test_splits

    elif group_name == "fleurs":
        train_splits = [
            load_dataset("google/fleurs", code, split="train", trust_remote_code=True)
            .select_columns(["audio", "transcription"]).rename_column("transcription", "texts")
            for code in FLEURS_LOW_MED.values()
        ]
        test_splits = [
            load_dataset("google/fleurs", code, split="test", trust_remote_code=True)
            .select_columns(["audio", "transcription"]).rename_column("transcription", "texts")
            for code in FLEURS_LOW_MED.values()
        ]
        return train_splits, test_splits

    raise ValueError(f"Unknown dataset group: {group_name}")
