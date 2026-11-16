from src.speech_alignment.asr import ASR

audio_path = "data/in/04d70c0c-ea87-41d8-958b-e869b1665232_test_output.wav"

print("Loading ASR model...")
asr = ASR()

print("Transcribing audio...")
transcript, detected_language = asr.transcribe(audio_path)

print("\nDetected language:", detected_language)
print("Transcript:", transcript)