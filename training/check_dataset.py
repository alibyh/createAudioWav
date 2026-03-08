"""
Quick check that the dataset loads and mel spectrograms have the expected shape.
Run from project root: python training/check_dataset.py
"""

from pathlib import Path

# Project root = parent of training/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"


def main():
    from training.dataset import PlaceNameDataset, SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH

    print("Loading dataset from", DATASET_DIR)
    ds = PlaceNameDataset(
        DATASET_DIR,
        labels_save_path=PROJECT_ROOT / "training" / "labels.json",
    )
    print(f"  Samples: {len(ds)}")
    print(f"  Classes: {ds.n_classes}")
    print(f"  Labels: {ds.ordered_labels}")

    if len(ds) == 0:
        print("No samples found. Check that metadata.csv and WAV paths exist.")
        return

    mel, label = ds[0]
    print(f"  Mel spectrogram shape: {mel.shape}  (n_mels, time)")
    print(f"  Label: {label} -> {ds.ordered_labels[label]}")

    # Time steps for 5s at 16kHz: (target_samples - n_fft) // hop_length + 1
    target_samples = SAMPLE_RATE * 5
    expected_time = (target_samples - N_FFT) // HOP_LENGTH + 1
    print(f"  Expected shape: (n_mels={N_MELS}, time={expected_time})")
    print("Done.")


if __name__ == "__main__":
    main()
