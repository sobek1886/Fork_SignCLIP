#!/usr/bin/env python3
"""
smplx_sequence_classifier.py  –  Sequence-aware signer-identity probe.

Uses a minimal 1D CNN that operates directly on the raw SMPL-X hand pose
sequence, without any temporal aggregation (no mean/std pooling).

Model
  Input  : (T, 90)  –  T frames of lhand_pose (45) + rhand_pose (45)
  Layer 1: Conv1d(in=90, out=32, kernel=5, padding=2)  → ReLU
            Detects local temporal patterns over ~5-frame windows.
            At 30 fps that is roughly a 1/6-second neighbourhood.
  Pool   : Global average over time  →  (32,)
            Collapses variable-length sequences without padding.
  Layer 2: Linear(32 → n_signers)

Total trainable parameters: 90 × 32 × 5 + 32 + 32 × n_signers ≈ 14 500

The model is intentionally tiny so that any above-chance result cannot be
attributed to model capacity; it must come from genuine sequence-level signal.

Evaluation: stratified 5-fold cross-validation.  All sequences are loaded
into memory once; each fold trains from scratch on 4/5 of the data and
evaluates on the held-out fifth.

Usage
  python smplx_sequence_classifier.py --dataset nsa --content_control
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Reuse all data-loading code from the aggregation-based script
sys.path.insert(0, str(Path(__file__).parent))
from smplx_signer_classifier import (
    NSA_DEFAULT, SA_DEFAULT,
    discover_nsa, discover_signavatars,
    load_nsa_clip, load_signavatars_clip,
    apply_content_control,
)


# ---------------------------------------------------------------------------
# Data loading  –  returns raw (T, 90) sequences, no aggregation
# ---------------------------------------------------------------------------

def load_hand_sequences(records, dataset):
    """
    Load raw hand pose sequences.

    Returns
    -------
    sequences : list of (T, 90) float32 arrays
        T varies per clip.  90 = lhand_pose (45) + rhand_pose (45).
        Pose layout inside the 159-dim SMPL-X pose vector:
          lhand : dims 66:111
          rhand : dims 111:156
    labels    : list of int
    label_names : list of str  (signer IDs, sorted)
    """
    signer_index = {}
    sequences, labels = [], []
    failed = 0

    for rec in tqdm(records, desc="Loading sequences"):
        if dataset == "nsa":
            result = load_nsa_clip(rec["path"])
        else:
            result = load_signavatars_clip(rec["path"])

        if result is None:
            failed += 1
            continue

        poses, _, _ = result          # poses: (T, 159)
        if len(poses) == 0:
            failed += 1
            continue

        # Slice out finger joints only – no wrist, no arm, no body
        lhand = poses[:, 66:111]      # (T, 45)
        rhand = poses[:, 111:156]     # (T, 45)
        hands = np.concatenate([lhand, rhand], axis=1).astype(np.float32)  # (T, 90)

        sid = rec["signer_id"]
        if sid not in signer_index:
            signer_index[sid] = len(signer_index)

        sequences.append(hands)
        labels.append(signer_index[sid])

    if failed:
        print(f"  (skipped {failed} clips due to errors)")

    label_names = sorted(signer_index, key=signer_index.get)
    return sequences, np.array(labels, dtype=np.int64), label_names


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class HandSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences    # list of (T_i, 90) arrays
        self.labels = labels          # (N,) int array

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.sequences[idx])   # (T, 90)
        y = int(self.labels[idx])
        return x, y


def collate_pad(batch):
    """Pad sequences in a batch to the same length for batch processing."""
    xs, ys = zip(*batch)
    lengths = [x.shape[0] for x in xs]
    max_len = max(lengths)
    feat_dim = xs[0].shape[1]

    padded = torch.zeros(len(xs), max_len, feat_dim)
    for i, (x, length) in enumerate(zip(xs, lengths)):
        padded[i, :length] = x

    lengths = torch.tensor(lengths, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.long)
    return padded, lengths, ys


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SignerCNN(nn.Module):
    """
    Single-layer 1D CNN with global average pooling.

    The convolution scans along the time axis with a kernel_size-frame
    receptive field, learning to detect local temporal patterns in the
    90-dim hand pose signal.  Global average pooling then summarises
    the entire (variable-length) sequence into a fixed 32-dim vector.
    """

    def __init__(self, input_dim: int = 90, hidden: int = 32,
                 kernel_size: int = 5, n_classes: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,   # same-length output
        )
        self.relu = nn.ReLU()
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, T, 90)
        lengths : (B,)  actual frame counts (rest is padding)
        returns : (B, n_classes) logits
        """
        # Conv1d expects (B, C, T)
        x = x.permute(0, 2, 1)           # (B, 90, T)
        x = self.relu(self.conv(x))       # (B, 32, T)

        # Masked global average pool – average only over real frames
        mask = torch.arange(x.shape[2], device=x.device).unsqueeze(0) \
               < lengths.unsqueeze(1)     # (B, T)
        mask = mask.unsqueeze(1).float()  # (B, 1, T)
        x = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)  # (B, 32)

        return self.head(x)               # (B, n_classes)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        optimiser.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, n = 0, 0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits = model(x, lengths)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return correct / n


def run_cv(sequences, labels, n_classes, n_splits=5, epochs=50,
           batch_size=16, lr=1e-3, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        train_seqs = [sequences[i] for i in train_idx]
        val_seqs   = [sequences[i] for i in val_idx]
        train_lbls = labels[train_idx]
        val_lbls   = labels[val_idx]

        train_loader = DataLoader(
            HandSequenceDataset(train_seqs, train_lbls),
            batch_size=batch_size, shuffle=True, collate_fn=collate_pad,
        )
        val_loader = DataLoader(
            HandSequenceDataset(val_seqs, val_lbls),
            batch_size=batch_size, shuffle=False, collate_fn=collate_pad,
        )

        model = SignerCNN(input_dim=90, hidden=32, kernel_size=5,
                          n_classes=n_classes).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            train_epoch(model, train_loader, optimiser, criterion, device)

        val_acc = evaluate(model, val_loader, device)
        fold_accs.append(val_acc)
        print(f"  Fold {fold + 1}/{n_splits}  val_acc={val_acc:.4f}")

    return np.array(fold_accs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["nsa", "signavatars"], default="nsa")
    parser.add_argument("--pose_dir", type=Path, default=None)
    parser.add_argument("--content_control", action="store_true")
    parser.add_argument("--min_signers_per_content", type=int, default=2)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--n_splits",   type=int, default=5)
    parser.add_argument("--epochs",     type=int, default=50,
                        help="Training epochs per fold (default: 50)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    if args.pose_dir is None:
        args.pose_dir = NSA_DEFAULT if args.dataset == "nsa" else SA_DEFAULT
    if not args.pose_dir.exists():
        print(f"ERROR: directory not found: {args.pose_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover clips
    print(f"Scanning {args.pose_dir} ...")
    records = discover_nsa(args.pose_dir) if args.dataset == "nsa" \
              else discover_signavatars(args.pose_dir)

    if not records:
        print("ERROR: no clips matched the expected naming convention.", file=sys.stderr)
        sys.exit(1)

    signers = set(r["signer_id"] for r in records)
    print(f"Found {len(records)} clips from {len(signers)} signer(s): {sorted(signers)}")

    if args.content_control:
        records = apply_content_control(records, args.min_signers_per_content)

    if args.max_clips:
        records = records[:args.max_clips]

    # Load raw sequences
    sequences, labels, label_names = load_hand_sequences(records, args.dataset)
    n_classes = len(label_names)
    lengths = [s.shape[0] for s in sequences]

    print(f"\nLoaded {len(sequences)} sequences, {n_classes} signers: {label_names}")
    print(f"Sequence length  min={min(lengths)}  max={max(lengths)}  "
          f"mean={np.mean(lengths):.0f}")

    n_params = 90 * 32 * 5 + 32 + 32 * n_classes
    print(f"\nModel: Conv1d(90→32, k=5) → ReLU → GlobalAvgPool → Linear(32→{n_classes})")
    print(f"Trainable parameters: {n_params:,}")
    print(f"Training: {args.epochs} epochs × {args.n_splits} folds, "
          f"batch={args.batch_size}, lr={args.lr}")

    # Cross-validation
    print()
    fold_accs = run_cv(
        sequences, labels, n_classes,
        n_splits=args.n_splits, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, seed=args.seed,
    )

    chance = 1.0 / n_classes
    mean_acc = fold_accs.mean()
    ratio = mean_acc / chance

    print("\n" + "=" * 60)
    print("SMPL-X SEQUENCE CLASSIFIER (1D CNN) – RESULTS")
    print("=" * 60)
    print(f"  Samples   : {len(sequences)}")
    print(f"  Signers   : {n_classes}  ({', '.join(label_names)})")
    print(f"  Features  : raw lhand_pose + rhand_pose, (T, 90), no aggregation")
    print(f"  Model     : Conv1d(90→32, k=5) → ReLU → GlobalAvgPool → Linear")
    print(f"  Params    : {n_params:,}")
    print(f"  CV        : {args.n_splits}-fold stratified, {args.epochs} epochs/fold")
    print()
    print(f"  Accuracy (per fold) : {' '.join(f'{a:.3f}' for a in fold_accs)}")
    print(f"  Mean accuracy       : {mean_acc:.4f}  ± {fold_accs.std():.4f}")
    print(f"  Chance baseline     : {chance:.4f}")
    print(f"  Accuracy / chance   : {ratio:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
