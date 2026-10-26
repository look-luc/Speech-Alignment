from datasets import concatenate_datasets, load_dataset

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
    fb_train = [
        load_dataset(
            "facebook/multilingual_librispeech",
            lang,
            split="train",
            streaming=True
        ).select_columns(
            ["audio", "transcript"]
        ).rename_column(
            "transcript", "texts"
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
        ) for lang in FACEBOOK_LANGS
    ]

    facebook_concat_ds_train = concatenate_datasets(fb_train)
    facebook_concat_ds_test = concatenate_datasets(fb_test)

    waxal_train, waxal_test = [], []
    for code in WAXAL_LANG_CODE.values():
        ds_train = load_dataset(
            "google/WaxalNLP",
            f"{code}_asr",
            split="train",
            streaming=True
        ).select_columns(
            ["audio", "transcript"]
        ).rename_column(
            "transcript", "texts"
        )
        ds_test = load_dataset(
            "google/WaxalNLP",
            f"{code}_asr",
            split="test",
            streaming=True
        ).select_columns(
            ["audio", "transcript"]
        ).rename_column(
            "transcript", "texts"
        )

        waxal_train.append(ds_train)
        waxal_test.append(ds_test)

    waxal_ds_train = concatenate_datasets(waxal_train)
    waxal_ds_test = concatenate_datasets(waxal_test)

    fleurs_train = [
        load_dataset(
            "google/fleurs",
            code,
            split="train",
            streaming=True
        ).select_columns(
            ["audio", "transcript"]
        ).rename_column(
            "transcript", "texts"
        ) for code in FLEURS_LOW_MED.values()
    ]
    fleurs_test = [
        load_dataset(
            "google/fleurs",
            code,
            split="test",
            streaming=True
        ).select_columns(
            ["audio", "transcript"]
        ).rename_column(
            "transcript", "texts"
        ) for code in FLEURS_LOW_MED.values()
    ]

    fleurs_concat_ds_train = concatenate_datasets(fleurs_train)
    fleurs_concat_ds_test = concatenate_datasets(fleurs_test)

    return {
        "facebook": (facebook_concat_ds_train, facebook_concat_ds_test),
        "waxal": (waxal_ds_train, waxal_ds_test),
        "fleurs": (fleurs_concat_ds_train, fleurs_concat_ds_test)
    }
