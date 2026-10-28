import os

# Point Hugging Face cache to your persistent project space on CURC
project_dir = f"/scratch/alpine/{os.environ.get('USER', 'lude4390')}"
os.environ["HF_HOME"] = f"{project_dir}/.cache/huggingface"

from datasets import load_dataset

FACEBOOK_LANGS = [
    "dutch", "french", "german", "italian", "polish", "portuguese", "spanish"
]

WAXAL_LANG_CODES = [
    "ach", "lug", "mas", "nyn", "sog", "aka", "ewe", "dag"
]

FLEURS_LOW_MED = [
    "as_in", "gu_in", "kn_in", "ml_in", "ne_np", "or_in", "pa_in", "sd_in",
    "ceb_ph", "jv_id", "ms_my", "mi_nz", "fil_ph", "kk_kz", "ky_kg", "ps_af",
    "el_gr", "fi_fi", "hu_hu", "cs_cz", "uk_ua", "ro_ro", "bg_bg", "ca_es", "oc_fr"
]

def download_and_cache():
    print("=== Caching Facebook Multilingual LibriSpeech ===")
    for lang in FACEBOOK_LANGS:
        for split in ["train", "test"]:
            print(f"Caching MLS: {lang} [{split}]")
            load_dataset("facebook/multilingual_librispeech", lang, split=split, trust_remote_code=True)

    print("\n=== Caching Google WaxalNLP ===")
    for code in WAXAL_LANG_CODES:
        for split in ["train", "test"]:
            print(f"Caching Waxal: {code}_asr [{split}]")
            load_dataset("google/WaxalNLP", f"{code}_asr", split=split, trust_remote_code=True)

    print("\n=== Caching Google Fleurs ===")
    for code in FLEURS_LOW_MED:
        for split in ["train", "test"]:
            print(f"Caching Fleurs: {code} [{split}]")
            load_dataset("google/fleurs", code, split=split, trust_remote_code=True)

    print("\nAll datasets successfully downloaded and cached to disk!")

if __name__ == "__main__":
    download_and_cache()
