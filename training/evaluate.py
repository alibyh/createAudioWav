"""
Evaluate a trained checkpoint with per-class accuracy and confusion matrix.

Run from project root, for example:
  python training/evaluate.py --split val
  python training/evaluate.py --split full
"""

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from training.dataset import PlaceNameDataset, get_dataloaders
from training.model import PlaceCNN


def format_label(label: str, width: int = 20) -> str:
    """Shorten long labels for table display."""
    if len(label) <= width:
        return label
    return label[: width - 3] + "..."


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    n_classes: int,
    device: torch.device,
):
    confusion = torch.zeros((n_classes, n_classes), dtype=torch.int64)
    total = 0
    correct = 0

    model.eval()
    for mel, labels in loader:
        mel = mel.to(device)
        labels = labels.to(device)
        logits = model(mel)
        preds = logits.argmax(dim=1)

        total += labels.numel()
        correct += (preds == labels).sum().item()
        for t, p in zip(labels.cpu(), preds.cpu()):
            confusion[int(t), int(p)] += 1

    overall_acc = (correct / total) if total else 0.0
    return confusion, overall_acc, total


def print_report(confusion: torch.Tensor, ordered_labels: List[str]) -> None:
    n_classes = len(ordered_labels)
    row_sums = confusion.sum(dim=1)
    per_class_acc = []

    print("\nPer-class accuracy:")
    for i in range(n_classes):
        total_i = int(row_sums[i].item())
        correct_i = int(confusion[i, i].item())
        acc_i = (correct_i / total_i) if total_i else 0.0
        per_class_acc.append(acc_i)
        print(f"- {ordered_labels[i]}: {correct_i}/{total_i} = {acc_i:.2%}")

    macro_acc = sum(per_class_acc) / n_classes if n_classes else 0.0
    print(f"\nMacro average accuracy: {macro_acc:.2%}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    short = [format_label(lbl, width=18) for lbl in ordered_labels]
    header = "true\\pred".ljust(20) + " ".join(s.ljust(6) for s in short)
    print(header)
    for i in range(n_classes):
        row_vals = " ".join(str(int(confusion[i, j].item())).ljust(6) for j in range(n_classes))
        print(short[i].ljust(20) + row_vals)


def main():
    parser = argparse.ArgumentParser(description="Evaluate place-name audio classifier")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to dataset/ (default: project dataset/)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "training" / "checkpoints" / "best.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--val_ratio", type=float, default=0.25, help="Validation ratio for --split val")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic split")
    parser.add_argument(
        "--split",
        choices=["val", "full"],
        default="val",
        help="Evaluate on validation split or full dataset",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else PROJECT_ROOT / "dataset"
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    ordered_labels = ckpt.get("ordered_labels")
    n_classes = ckpt.get("n_classes")
    if ordered_labels is None or n_classes is None:
        print("Invalid checkpoint: missing ordered_labels or n_classes")
        sys.exit(1)

    model = PlaceCNN(n_classes=n_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])

    if args.split == "full":
        ds = PlaceNameDataset(dataset_dir)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        split_name = "full dataset"
    else:
        _, val_loader, _, _ = get_dataloaders(
            dataset_dir,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        loader = val_loader
        split_name = "validation split"

    confusion, overall_acc, total = evaluate_loader(model, loader, n_classes, device)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Evaluated on: {split_name}")
    print(f"Samples: {total}")
    print(f"Overall accuracy: {overall_acc:.2%}")
    print_report(confusion, ordered_labels)


if __name__ == "__main__":
    main()
