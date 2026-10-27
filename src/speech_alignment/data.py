from datasets import Audio, Features, Value, load_dataset

FACEBOOK_LANGS = [
    "dutch", "french", "german", "italian", "polish", "portuguese", "spanish"
]

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
    """Loads and returns train/test splits with decode=False for robust audio loading."""
    if group_name == "facebook":
        MAX_MLS_SAMPLES = 3000
        train_splits = [
            load_dataset("facebook/multilingual_librispeech", lang, split="train", streaming=True)
            .shuffle(seed=42).take(MAX_MLS_SAMPLES)
            .select_columns(["audio", "transcript"]).rename_column("transcript", "texts")
            .cast_column("audio", Audio(sampling_rate=16000, decode=False))
            for lang in FACEBOOK_LANGS
        ]
        test_splits = [
            load_dataset("facebook/multilingual_librispeech", lang, split="test", streaming=True)
            .select_columns(["audio", "transcript"]).rename_column("transcript", "texts")
            .cast_column("audio", Audio(sampling_rate=16000, decode=False))
            for lang in FACEBOOK_LANGS
        ]
        return train_splits, test_splits

    elif group_name == "waxal":
        waxal_features = Features({
            "id": Value("string"), "speaker_id": Value("string"), "transcription": Value("string"),
            "language": Value("string"), "gender": Value("string"), "audio": Audio(decode=False), "__index_level_0__": Value("int64"),
        })
        train_splits, test_splits = [], []
        for code in WAXAL_LANG_CODE.values():
            train_splits.append(
                load_dataset("google/WaxalNLP", f"{code}_asr", split="train", streaming=True, features=waxal_features)
                .select_columns(["audio", "transcription"]).rename_column("transcription", "texts")
                .cast_column("audio", Audio(sampling_rate=16000, decode=False))
            )
            test_splits.append(
                load_dataset("google/WaxalNLP", f"{code}_asr", split="test", streaming=True, features=waxal_features)
                .select_columns(["audio", "transcription"]).rename_column("transcription", "texts")
                .cast_column("audio", Audio(sampling_rate=16000, decode=False))
            )
        return train_splits, test_splits

    elif group_name == "fleurs":
        train_splits = [
            load_dataset("google/fleurs", code, split="train", streaming=True)
            .select_columns(["audio", "transcription"]).rename_column("transcription", "texts")
            .cast_column("audio", Audio(sampling_rate=16000, decode=False))
            for code in FLEURS_LOW_MED.values()
        ]
        test_splits = [
            load_dataset("google/fleurs", code, split="test", streaming=True)
            .select_columns(["audio", "transcription"]).rename_column("transcription", "texts")
            .cast_column("audio", Audio(sampling_rate=16000, decode=False))
            for code in FLEURS_LOW_MED.values()
        ]
        return train_splits, test_splits

    raise ValueError(f"Unknown dataset group: {group_name}")
