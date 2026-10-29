import argparse

from src.speech_alignment.model import Forced_Aligner


def main():
    model = Forced_Aligner()
    print("training model")
    train = model.train_pipeline()
    print("evaluating model")
    final_model = model.evaluate_pipeline()

if __name__ == "__main__":
    main()
    print("ran")
