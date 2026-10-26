import argparse

from src.speech_alignment.model import Forced_Aligner


def main():
    parser = argparse.ArgumentParser(description="A multilingual forced aligner")
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=False, required=True, help="Run model")

    args = parser.parse_args()
    if args:
        model = Forced_Aligner()
        final_model = model.evaluate_pipeline()

if __name__ == "__main__":
    main()
    print("ran")
