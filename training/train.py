"""
Train the place-name classifier.
Run from project root: python training/train.py [--epochs 30] [--batch_size 16]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from training.dataset import PlaceNameDataset, get_dataloaders
from training.model import PlaceCNN


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for mel, labels in loader:
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


def main():
    parser = argparse.ArgumentParser(description="Train place-name audio classifier")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to dataset/ (default: project dataset/)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
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
