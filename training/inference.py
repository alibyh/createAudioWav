"""
Predict place name from a single WAV file using the trained model.
Run from project root: python training/inference.py path/to/recording.wav
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchaudio

from training.dataset import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS
from training.model import PlaceCNN


def wav_to_mel(wav_path: Path, target_length_sec: float = 5.0) -> torch.Tensor:
    """Load WAV and convert to mel spectrogram (same as PlaceNameDataset)."""
    target_samples = int(target_length_sec * SAMPLE_RATE)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    n = waveform.shape[1]
    if n >= target_samples:
        waveform = waveform[:, :target_samples]
    else:
        padding = torch.zeros(1, target_samples - n)
        waveform = torch.cat([waveform, padding], dim=1)
    mel = mel_spec(waveform).squeeze(0)
    return mel


def main():
    if len(sys.argv) < 2:
        print("Usage: python training/inference.py <path/to/audio.wav> [--checkpoint training/checkpoints/best.pt]")
        sys.exit(1)

    wav_path = Path(sys.argv[1])
    if not wav_path.is_absolute():
        wav_path = (PROJECT_ROOT / wav_path).resolve()
    if not wav_path.exists():
        print(f"File not found: {wav_path}")
        sys.exit(1)

    checkpoint_path = PROJECT_ROOT / "training" / "checkpoints" / "best.pt"
    if "--checkpoint" in sys.argv:
        i = sys.argv.index("--checkpoint")
        if i + 1 < len(sys.argv):
            checkpoint_path = Path(sys.argv[i + 1])

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}. Train first with: python training/train.py")
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    n_classes = ckpt["n_classes"]
    ordered_labels = ckpt.get("ordered_labels")
    if ordered_labels is None:
        labels_path = PROJECT_ROOT / "training" / "labels.json"
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                ordered_labels = [json.load(f)[str(i)] for i in range(n_classes)]
        else:
            ordered_labels = [str(i) for i in range(n_classes)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlaceCNN(n_classes=n_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    mel = wav_to_mel(wav_path).unsqueeze(0)
    mel = mel.to(device)
    with torch.no_grad():
        logits = model(mel)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()

    place_name = ordered_labels[pred]
    confidence = probs[0, pred].item()
    print(f"Predicted: {place_name}  (confidence: {confidence:.2%})")
    return place_name


if __name__ == "__main__":
    main()
