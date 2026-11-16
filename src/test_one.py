"""Run from repo root:  uv run python -m src.test_one
Aligns one English FLEURS test clip, prints the words, and writes test_output.TextGrid + test_output.wav"""
import soundfile as sf

from src.speech_alignment.aligner import Aligner
from src.speech_alignment.data import load_single_language_data
from src.speech_alignment.to_textgrid import words_to_textgrid

ds = load_single_language_data(language_name="en_us", split="test")
ex = ds[0]
audio, text = ex["audio"]["array"], ex["transcription"]
duration = len(audio) / 16000
print("DURATION:", duration, "s")
print("TRANSCRIPT:", text)

words = Aligner().align(audio, text)
for w in words:
    print(f'{w["start"]:6.2f} {w["end"]:6.2f}  {w["word"]}')

sf.write("test_output.wav", audio, 16000)
words_to_textgrid(words, duration, "test_output.TextGrid")
print("Wrote test_output.wav and test_output.TextGrid in the repo root")
