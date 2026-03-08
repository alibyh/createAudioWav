# Place-name audio classifier (PyTorch)

Dataset loader and (later) training for recognizing place names from short WAV recordings.

## Setup

From the project root:

```bash
pip install -r requirements.txt -r requirements-training.txt
```

**Note:** PyTorch does not yet provide official wheels for Python 3.13. Use Python 3.10–3.12 for the training environment, or create a separate venv with e.g. `python3.11 -m venv venv_train` and install there.

## Dataset loader

- **`dataset.py`** – `PlaceNameDataset`: loads `dataset/metadata.csv` and `dataset/places_names.json`, builds `location → label` from the place list, loads WAVs from `dataset/<folder>/<file>.wav`, and returns **(mel spectrogram, label)**.
- **Mel settings:** 16 kHz, 64 mels, FFT 1024, hop 512. Clips are padded/cropped to 5 seconds.
- **`labels.json`** – Saved under `training/` when you create the dataset with `labels_save_path`; use it at inference to map class index → place name.

## Check the dataset

```bash
python training/check_dataset.py
```

Prints number of samples, classes, label list, and shape of one mel spectrogram.

## Next step

Add `train.py`: build a small CNN (e.g. Conv2D → ReLU → MaxPool → … → Dense → Softmax), train with cross-entropy and Adam, save the model and use `training/labels.json` for inference.
