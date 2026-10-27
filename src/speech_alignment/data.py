import io

import soundfile as sf
import torch
import torchaudio.functional as ta_func
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
    """Loads and returns train/test splits with decode=False and trust_remote_code=True for offline execution."""
    if group_name == "facebook":
        MAX_MLS_SAMPLES = 3000
        train_splits = [
            load_dataset("facebook/multilingual_librispeech", lang, split="train", streaming=False, trust_remote_code=True)
            .shuffle(seed=42).select(range(min(MAX_MLS_SAMPLES, 50000)))
            .select_columns(["audio", "transcript"]).rename_column("transcript", "texts")
            .cast_column("audio", Audio(sampling_rate=16000, decode=False))
            for lang in FACEBOOK_LANGS
        ]
        test_splits = [
            load_dataset("facebook/multilingual_librispeech", lang, split="test", streaming=False, trust_remote_code=True)
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
                load_dataset("google/WaxalNLP", f"{code}_asr", split="train", streaming=False, features=waxal_features, trust_remote_code=True)
                .select_columns(["audio", "transcription"]).rename_column("transcription", "texts")
                .cast_column("audio", Audio(sampling_rate=16000, decode=False))
            )
            test_splits.append(
                load_dataset("google/WaxalNLP", f"{code}_asr", split="test", streaming=False, features=waxal_features, trust_remote_code=True)
                .select_columns(["audio", "transcription"]).rename_column("transcription", "texts")
                .cast_column("audio", Audio(sampling_rate=16000, decode=False))
            )
        return train_splits, test_splits

    elif group_name == "fleurs":
        train_splits = [
            load_dataset("google/fleurs", code, split="train", streaming=False, trust_remote_code=True)
            .select_columns(["audio", "transcription"]).rename_column("transcription", "texts")
            .cast_column("audio", Audio(sampling_rate=16000, decode=False))
            for code in FLEURS_LOW_MED.values()
        ]
        test_splits = [
            load_dataset("google/fleurs", code, split="test", streaming=False, trust_remote_code=True)
            .select_columns(["audio", "transcription"]).rename_column("transcription", "texts")
            .cast_column("audio", Audio(sampling_rate=16000, decode=False))
            for code in FLEURS_LOW_MED.values()
        ]
        return train_splits, test_splits

    raise ValueError(f"Unknown dataset group: {group_name}")

def load_and_preprocess_dataset(dataset_name, dataset_config=None, split="train", target_sr=16000, streaming=False):
    """
    Loads a Hugging Face speech dataset with decoding disabled and processes
    audio files using soundfile to completely bypass torchcodec and FFmpeg.
    """
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=streaming, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=streaming, trust_remote_code=True)

    dataset = dataset.cast_column("audio", Audio(decode=False))

    def process_batch(batch):
        waveforms = []
        sample_rates = []

        for audio_item in batch["audio"]:
            if "bytes" in audio_item and audio_item["bytes"] is not None:
                audio_source = io.BytesIO(audio_item["bytes"])
            else:
                audio_source = audio_item["path"]

            waveform_np, sr = sf.read(audio_source)
            waveform = torch.tensor(waveform_np, dtype=torch.float32)

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.transpose(0, 1)

            if sr != target_sr:
                waveform = ta_func.resample(waveform, orig_freq=sr, new_freq=target_sr)
                sr = target_sr

            waveforms.append(waveform)
            sample_rates.append(sr)

        batch["input_values"] = waveforms
        batch["sampling_rate"] = sample_rates
        return batch

    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=32,
        remove_columns=["audio"]
    )

    return processed_dataset
