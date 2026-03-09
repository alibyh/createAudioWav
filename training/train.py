"""
Train the place-name classifier.
Run from project root: python training/train.py [--epochs 30] [--batch_size 16]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio

from training.dataset import PlaceNameDataset, get_dataloaders
from training.model import PlaceCNN


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    freq_mask_param: int = 0,
    time_mask_param: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param) if freq_mask_param > 0 else None
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param) if time_mask_param > 0 else None
    for mel, labels in loader:
        if freq_mask is not None:
            mel = freq_mask(mel)
        if time_mask is not None:
            mel = time_mask(mel)
        mel = mel.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(mel)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mel.size(0)
        n += mel.size(0)
    return total_loss / n if n else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for mel, labels in loader:
        mel = mel.to(device)
        labels = labels.to(device)
        logits = model(mel)
        loss = criterion(logits, labels)
        pred = logits.argmax(dim=1)
        total_loss += loss.item() * mel.size(0)
        correct += (pred == labels).sum().item()
        n += mel.size(0)
    avg_loss = total_loss / n if n else 0.0
    acc = correct / n if n else 0.0
    return avg_loss, acc


def build_class_weights(train_loader: DataLoader, n_classes: int, device: torch.device) -> torch.Tensor | None:
    """
    Build inverse-frequency class weights from the train split to reduce class imbalance.
    Returns None if labels cannot be inferred.
    """
    ds = train_loader.dataset
    labels = []
    if hasattr(ds, "indices") and hasattr(ds, "dataset") and hasattr(ds.dataset, "samples"):
        labels = [ds.dataset.samples[i][1] for i in ds.indices]
    elif hasattr(ds, "samples"):
        labels = [label for _, label in ds.samples]
    if not labels:
        return None

    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(n_classes):
        c = counts.get(i, 0)
        w = (total / (n_classes * c)) if c > 0 else 0.0
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser(description="Train place-name audio classifier")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to dataset/ (default: project dataset/)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--freq_mask_param", type=int, default=8, help="SpecAugment frequency mask size")
    parser.add_argument("--time_mask_param", type=int, default=12, help="SpecAugment time mask size")
    parser.add_argument("--save_dir", type=str, default=None, help="Where to save checkpoints (default: training/checkpoints)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else PROJECT_ROOT / "dataset"
    save_dir = Path(args.save_dir) if args.save_dir else PROJECT_ROOT / "training" / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, n_classes, ordered_labels = get_dataloaders(
        dataset_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        labels_save_path=PROJECT_ROOT / "training" / "labels.json",
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Classes: {n_classes}")

    if n_classes == 0 or len(train_loader) == 0:
        print("No data to train on. Add WAVs and metadata first.")
        return

    model = PlaceCNN(n_classes=n_classes).to(device)
    class_weights = build_class_weights(train_loader, n_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            freq_mask_param=args.freq_mask_param,
            time_mask_param=args.time_mask_param,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.2%}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ckpt = save_dir / "best.pt"
            torch.save({"state_dict": model.state_dict(), "n_classes": n_classes, "ordered_labels": ordered_labels}, ckpt)
            print(f"  -> saved {ckpt}")

    print(f"Done. Best val accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    main()
