from datasets import Audio, Features, Value, load_dataset

FACEBOOK_LANGS = [
    "dutch",
    "french",
    "german",
    "italian",
    "polish",
    "portuguese",
    "spanish"
]

WAXAL_LANG_CODE = {
    "Acholi": "ach",
    "Luganda": "lug",
    "Masaaba": "mas",
    "Nyankole": "nyn",
    "Soga": "sog",
    "Akan": "aka",
    "Ewe": "ewe",
    "Dagbani": "dag",
}

FLEURS_LOW_MED = {
    "Assamese": "as_in",
    "Gujarati": "gu_in",
    "Kannada": "kn_in",
    "Malayalam": "ml_in",
    "Nepali": "ne_np",
    "Odia": "or_in",
    "Punjabi": "pa_in",
    "Sindhi": "sd_in",
    "Cebuano": "ceb_ph",
    "Javanese": "jv_id",
    "Malay": "ms_my",
    "Maori": "mi_nz",
    "Filipino": "fil_ph",
    "Kazakh": "kk_kz",
    "Kyrgyz": "ky_kg",
    "Pashto": "ps_af",
    "Greek": "el_gr",
    "Finnish": "fi_fi",
    "Hungarian": "hu_hu",
    "Czech": "cs_cz",
    "Ukrainian": "uk_ua",
    "Romanian": "ro_ro",
    "Bulgarian": "bg_bg",
    "catalan": "ca_es",
    "occitan": "oc_fr",
}

def get_data():
    MAX_MLS_SAMPLES = 3000

    fb_train = [
        load_dataset(
            "facebook/multilingual_librispeech",
            lang,
            split="train",
            streaming=True
        ).shuffle(seed=42).take(MAX_MLS_SAMPLES).select_columns(
            ["audio", "transcript"]
        ).rename_column(
            "transcript", "texts"
        ).cast_column(
            "audio", Audio(sampling_rate=16000)
        ) for lang in FACEBOOK_LANGS
    ]
    fb_test = [
        load_dataset(
            "facebook/multilingual_librispeech",
            lang,
            split="test",
            streaming=True
        ).select_columns(
            ["audio", "transcript"]
        ).rename_column(
            "transcript", "texts"
        ).cast_column(
            "audio", Audio(sampling_rate=16000)
        ) for lang in FACEBOOK_LANGS
    ]

    # 2. WaxalNLP (Explicitly define features including __index_level_0__ to prevent CastError)
    waxal_features = Features({
        "id": Value("string"),
        "speaker_id": Value("string"),
        "transcription": Value("string"),
        "language": Value("string"),
        "gender": Value("string"),
        "audio": Audio(),
        "__index_level_0__": Value("int64"),
    })

    waxal_train, waxal_test = [], []
    for code in WAXAL_LANG_CODE.values():
        ds_train = load_dataset(
            "google/WaxalNLP",
            f"{code}_asr",
            split="train",
            streaming=True,
            features=waxal_features
        ).select_columns(
            ["audio", "transcription"]
        ).rename_column(
            "transcription", "texts"
        ).cast_column(
            "audio", Audio(sampling_rate=16000)
        )

        ds_test = load_dataset(
            "google/WaxalNLP",
            f"{code}_asr",
            split="test",
            streaming=True,
            features=waxal_features
        ).select_columns(
            ["audio", "transcription"]
        ).rename_column(
            "transcription", "texts"
        ).cast_column(
            "audio", Audio(sampling_rate=16000)
        )

        waxal_train.append(ds_train)
        waxal_test.append(ds_test)

    # 3. Google FLEURS
    fleurs_train = [
        load_dataset(
            "google/fleurs",
            code,
            split="train",
            streaming=True
        ).select_columns(
            ["audio", "transcription"]
        ).rename_column(
            "transcription", "texts"
        ).cast_column(
            "audio", Audio(sampling_rate=16000)
        ) for code in FLEURS_LOW_MED.values()
    ]
    fleurs_test = [
        load_dataset(
            "google/fleurs",
            code,
            split="test",
            streaming=True
        ).select_columns(
            ["audio", "transcription"]
        ).rename_column(
            "transcription", "texts"
        ).cast_column(
            "audio", Audio(sampling_rate=16000)
        ) for code in FLEURS_LOW_MED.values()
    ]

    return {
        "facebook": (fb_train, fb_test),
        "waxal": (waxal_train, waxal_test),
        "fleurs": (fleurs_train, fleurs_test)
    }
