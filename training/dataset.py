"""
PyTorch Dataset for place-name audio classification.
Loads WAVs from dataset/, converts to mel spectrograms, and returns (spectrogram, label).
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from torch.utils.data import Dataset
import torchaudio


# Default mel spectrogram settings (speech commands style)
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
# Fixed length: 5 seconds of audio (pad/crop to this)
TARGET_SAMPLES = SAMPLE_RATE * 5


def location_to_folder_name(location: str) -> str:
    """Convert location display name to folder name (spaces -> underscores)."""
    return location.strip().replace(" ", "_") if location else ""


def load_label_mapping(places_path: Path) -> Tuple[List[str], Dict[str, int]]:
    """
    Load ordered list of place names and mapping name -> index from places_names.json.
    Returns (ordered_names, name_to_index).
    """
    with open(places_path, "r", encoding="utf-8") as f:
        places = json.load(f)
    ordered = []
    for item in places:
        if isinstance(item, dict) and item.get("name"):
            name = item["name"].strip()
            if name and name not in ordered:
                ordered.append(name)
    name_to_index = {name: i for i, name in enumerate(ordered)}
    return ordered, name_to_index


class PlaceNameDataset(Dataset):
    """
    Dataset of (mel spectrogram, place label) for place-name audio.
    Reads metadata.csv and places_names.json; loads WAVs and converts to mel spec.
    """

    def __init__(
        self,
        dataset_dir: Union[Path, str],
        metadata_path: Union[Path, str, None] = None,
        places_path: Union[Path, str, None] = None,
        target_length_sec: float = 5.0,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        labels_save_path: Union[Path, str, None] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        if metadata_path is None:
            metadata_path = self.dataset_dir / "metadata.csv"
        self.metadata_path = Path(metadata_path)
        if places_path is None:
            places_path = self.dataset_dir / "places_names.json"
        self.places_path = Path(places_path)

        self.sample_rate = sample_rate
        self.target_samples = int(target_length_sec * sample_rate)
        self.ordered_labels, self.name_to_index = load_label_mapping(self.places_path)
        self.n_classes = len(self.ordered_labels)

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        # Build list of (filepath, label_index), skipping missing files and duplicates
        self.samples: List[Tuple[Path, int]] = []
        seen_paths: set[Path] = set()
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = (row.get("filename") or "").strip()
                location = (row.get("location") or "").strip()
                if not filename or not location:
                    continue
                if location not in self.name_to_index:
                    continue
                folder_name = location_to_folder_name(location)
                filepath = self.dataset_dir / folder_name / filename
                if not filepath.exists():
                    continue
                filepath = filepath.resolve()
                if filepath in seen_paths:
                    continue
                seen_paths.add(filepath)
                label = self.name_to_index[location]
                self.samples.append((filepath, label))

        if labels_save_path is not None:
            self._save_labels(Path(labels_save_path))

    def _save_labels(self, path: Path) -> None:
        """Save index -> place_name mapping for inference."""
        path.parent.mkdir(parents=True, exist_ok=True)
        mapping = {str(i): name for i, name in enumerate(self.ordered_labels)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]
        waveform, sr = torchaudio.load(str(filepath))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        # Pad or crop to target length
        n = waveform.shape[1]
        if n >= self.target_samples:
            waveform = waveform[:, : self.target_samples]
        else:
            padding = torch.zeros(1, self.target_samples - n)
            waveform = torch.cat([waveform, padding], dim=1)
        mel = self.mel_spec(waveform)
        mel = mel.squeeze(0)
        return mel, label


def get_dataloaders(
    dataset_dir: Path | str,
    batch_size: int = 16,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    **dataset_kwargs,
):
    """
    Build train and validation DataLoaders with an 80/20 split.
    """
    from torch.utils.data import DataLoader, random_split

    full = PlaceNameDataset(dataset_dir, **dataset_kwargs)
    n = len(full)
    n_val = max(0, int(n * val_ratio))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=gen)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, full.n_classes, full.ordered_labels
