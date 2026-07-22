#!/usr/bin/env python3
"""
smplx_signer_classifier.py  –  Linear signer-identity probe on SMPL-X parameters.

Mirrors signer_classifier.py but operates on SMPL-X pose parameters instead of
raw MediaPipe keypoints, letting us compare how much signer identity leaks from
each representation.

Supported datasets
  nsa          –  Neural Sign Actors: one subdirectory per clip, per-frame PKL files
                  Path:  /Users/piotr/Projects/Thesis/Data/Neural Sign Actors/poses/
  signavatars  –  SignAvatars How2Sign: one PKL per clip (whole-sequence tensor)
                  Path:  /Users/piotr/Projects/Thesis/Data/SignAvatars/how2sign_pkls_cropTrue_shapeTrue/

Clip naming convention (same as How2Sign):
  {video_id}_{sent_idx}-{signer_id}-rgb_front[__blender]

SMPL-X components included by default (pose only, 159 dims/frame):
  root_pose   3   global orientation
  body_pose  63   21 body joints (axis-angle)
  lhand_pose 45   15 left-hand joints
  rhand_pose 45   15 right-hand joints
  jaw_pose    3   jaw rotation

Optional additions:
  --include_shape   adds betas (10 dims, mean over frames)   – anatomical confound
  --include_expr    adds expression (10 dims, mean+std)      – facial dynamics

Aggregation: mean + std pooling over time → 318-dim vector (pose only).

Usage
  python smplx_signer_classifier.py --dataset nsa
  python smplx_signer_classifier.py --dataset nsa --content_control
  python smplx_signer_classifier.py --dataset nsa --include_shape
  python smplx_signer_classifier.py --dataset signavatars
"""

import io
import os
import re
import sys
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
NSA_DEFAULT     = Path("/Users/piotr/Projects/Thesis/Data/Neural Sign Actors/poses")
SA_DEFAULT      = Path("/Users/piotr/Projects/Thesis/Data/SignAvatars/how2sign_pkls_cropTrue_shapeTrue")

# ---------------------------------------------------------------------------
# Pickle helper: handle PyTorch tensors saved on CUDA
# ---------------------------------------------------------------------------

class _CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return _CPU_Unpickler(f).load()


def to_np(v, n: int) -> np.ndarray:
    """Convert tensor/array to flat float32 of length n."""
    if v is None:
        return np.zeros(n, dtype=np.float32)
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    if a.shape[0] >= n:
        return a[:n]
    return np.pad(a, (0, n - a.shape[0]))


def frame_dict_to_row(d: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract one frame's (pose_159, shape_10, expr_10) from a per-frame dict.
    Pose layout: root(3) body(63) lhand(45) rhand(45) jaw(3) = 159
    """
    root  = to_np(d.get("smplx_root_pose"),  3)
    body  = to_np(d.get("smplx_body_pose"),  63)
    lh    = to_np(d.get("smplx_lhand_pose"), 45)
    rh    = to_np(d.get("smplx_rhand_pose"), 45)
    jaw   = to_np(d.get("smplx_jaw_pose"),   3)
    pose  = np.concatenate([root, body, lh, rh, jaw])   # (159,)
    shape = to_np(d.get("smplx_shape"), 10)
    expr  = to_np(d.get("smplx_expr"),  10)
    return pose, shape, expr


# ---------------------------------------------------------------------------
# Clip naming – same regex as How2Sign / signer_classifier.py
# ---------------------------------------------------------------------------

# Matches  {video_id}_{sent_idx}-{signer_id}-rgb_front[__blender]
_CLIP_RE = re.compile(r'^(.+)_(\d+)-(\d+)-rgb_front(?:__blender)?$')


def parse_clip_name(name: str) -> dict | None:
    m = _CLIP_RE.match(name)
    if not m:
        return None
    return {
        "video_id":    m.group(1),
        "sent_idx":    m.group(2),
        "signer_id":   m.group(3),
        "content_key": (m.group(1), m.group(2)),
    }


# ---------------------------------------------------------------------------
# NSA loader: per-frame PKL files inside clip subdirectories
# ---------------------------------------------------------------------------

_NSA_FRAME_RE = re.compile(r'_(\d+)_3D\.pkl$')


def load_nsa_clip(clip_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Load all per-frame PKL files in a clip directory.

    Returns (poses, shapes, exprs) arrays of shape (T, 159), (T, 10), (T, 10)
    or None on failure.
    """
    frame_files = sorted(
        clip_dir.glob("*.pkl"),
        key=lambda p: int(m.group(1)) if (m := _NSA_FRAME_RE.search(p.name)) else -1,
    )
    # Filter out files that don't match the frame pattern
    frame_files = [p for p in frame_files if _NSA_FRAME_RE.search(p.name)]
    if not frame_files:
        return None

    poses, shapes, exprs = [], [], []
    for fp in frame_files:
        try:
            d = load_pkl(fp)
            pose, shape, expr = frame_dict_to_row(d)
            poses.append(pose)
            shapes.append(shape)
            exprs.append(expr)
        except Exception as exc:
            pass  # skip corrupt frame silently

    if not poses:
        return None
    return (
        np.stack(poses,  axis=0).astype(np.float32),   # (T, 159)
        np.stack(shapes, axis=0).astype(np.float32),   # (T, 10)
        np.stack(exprs,  axis=0).astype(np.float32),   # (T, 10)
    )


def discover_nsa(poses_dir: Path) -> list[dict]:
    records = []
    for d in sorted(poses_dir.iterdir()):
        if not d.is_dir():
            continue
        info = parse_clip_name(d.name)
        if info is None:
            continue
        if d.name.endswith("__blender"):
            continue   # skip Blender variants
        info["path"] = d
        records.append(info)
    return records


# ---------------------------------------------------------------------------
# SignAvatars loader: one PKL per clip (whole-sequence tensor / dict)
# ---------------------------------------------------------------------------

def load_signavatars_clip(pkl_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    SignAvatars How2Sign PKLs store the full sequence.

    Observed formats:
      A) dict with per-frame keys → stack manually (same as NSA frames)
      B) dict with keys mapping to (T, D) tensors
      C) a single (T, 165+) tensor  (layout from convert_pkl_amass_NSA.py)

    We try each in turn.
    """
    try:
        d = load_pkl(pkl_path)
    except Exception as exc:
        return None

    # Format C: bare tensor
    if isinstance(d, torch.Tensor):
        arr = d.cpu().numpy().astype(np.float32)    # (T, ?)
        pose  = arr[:, :159]
        shape = arr[:, 175:185] if arr.shape[1] >= 185 else np.zeros((arr.shape[0], 10), dtype=np.float32)
        expr  = arr[:, 165:175] if arr.shape[1] >= 175 else np.zeros((arr.shape[0], 10), dtype=np.float32)
        return pose, shape, expr

    if not isinstance(d, dict):
        return None

    # Format A/B: dict
    first_val = next(iter(d.values()))
    if isinstance(first_val, (torch.Tensor, np.ndarray)):
        first_arr = np.asarray(first_val, dtype=np.float32)
        if first_arr.ndim == 1:
            # Format A: per-frame dict (shouldn't happen for whole-clip PKLs, but handle it)
            pose, shape, expr = frame_dict_to_row(d)
            return (
                pose[None],   # (1, 159)
                shape[None],
                expr[None],
            )
        # Format B: (T, D) tensors per key
        T = first_arr.shape[0]
        def get_seq(key, n):
            v = d.get(key)
            if v is None:
                return np.zeros((T, n), dtype=np.float32)
            a = np.asarray(v, dtype=np.float32)
            if a.ndim == 1:
                a = np.tile(a, (T, 1))
            return a[:, :n] if a.shape[1] >= n else np.pad(a, ((0,0),(0, n - a.shape[1])))

        root  = get_seq("smplx_root_pose",  3)
        body  = get_seq("smplx_body_pose",  63)
        lh    = get_seq("smplx_lhand_pose", 45)
        rh    = get_seq("smplx_rhand_pose", 45)
        jaw   = get_seq("smplx_jaw_pose",   3)
        pose  = np.concatenate([root, body, lh, rh, jaw], axis=1)  # (T, 159)
        shape = get_seq("smplx_shape", 10)
        expr  = get_seq("smplx_expr",  10)
        return pose, shape, expr

    return None


def discover_signavatars(pkl_dir: Path) -> list[dict]:
    records = []
    for p in sorted(pkl_dir.glob("*.pkl")):
        info = parse_clip_name(p.stem)
        if info is None:
            continue
        info["path"] = p
        records.append(info)
    return records


# ---------------------------------------------------------------------------
# Content control  (same logic as signer_classifier.py)
# ---------------------------------------------------------------------------

def apply_content_control(records: list[dict], min_signers: int) -> list[dict]:
    by_content: dict = defaultdict(set)
    for r in records:
        by_content[r["content_key"]].add(r["signer_id"])
    valid = {k for k, s in by_content.items() if len(s) >= min_signers}
    filtered = [r for r in records if r["content_key"] in valid]
    print(f"Content control: {len(valid)} content keys with ≥{min_signers} signers → "
          f"{len(filtered)}/{len(records)} clips retained")
    return filtered


# ---------------------------------------------------------------------------
# Feature aggregation
# ---------------------------------------------------------------------------

def aggregate(
    poses: np.ndarray,    # (T, 159)
    shapes: np.ndarray,   # (T, 10)
    exprs: np.ndarray,    # (T, 10)
    include_shape: bool,
    include_expr: bool,
    hands_only: bool,
) -> np.ndarray:
    """Mean + std pooling; optionally restrict to hand joints or append shape/expr.

    Pose layout inside the 159-dim vector:
      root_pose   0:3    (3)  global orientation
      body_pose   3:66  (63)  21 body joints
      lhand_pose  66:111 (45) 15 left-hand finger joints
      rhand_pose 111:156 (45) 15 right-hand finger joints
      jaw_pose   156:159  (3) jaw
    """
    if hands_only:
        # 45 dims lhand + 45 dims rhand = 90 dims/frame → 180 after mean+std
        lhand = poses[:, 66:111]
        rhand = poses[:, 111:156]
        hands = np.concatenate([lhand, rhand], axis=1)   # (T, 90)
        parts = [hands.mean(0), hands.std(0)]
    else:
        parts = [poses.mean(0), poses.std(0)]              # 318 dims

    if include_shape:
        parts.append(shapes.mean(0))                   # +10 dims (nearly constant per clip)
    if include_expr:
        parts += [exprs.mean(0), exprs.std(0)]         # +20 dims
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Main feature extraction loop
# ---------------------------------------------------------------------------

def extract_features(
    records: list[dict],
    dataset: str,
    include_shape: bool,
    include_expr: bool,
    hands_only: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    signer_index: dict[str, int] = {}
    X_list, y_list = [], []
    failed = 0

    for rec in tqdm(records, desc="Loading SMPL-X"):
        if dataset == "nsa":
            result = load_nsa_clip(rec["path"])
        else:
            result = load_signavatars_clip(rec["path"])

        if result is None:
            failed += 1
            continue

        poses, shapes, exprs = result
        if len(poses) == 0:
            failed += 1
            continue

        vec = aggregate(poses, shapes, exprs, include_shape, include_expr, hands_only)

        sid = rec["signer_id"]
        if sid not in signer_index:
            signer_index[sid] = len(signer_index)

        X_list.append(vec)
        y_list.append(signer_index[sid])

    if failed:
        print(f"  (skipped {failed} clips due to errors)")

    if not X_list:
        print("ERROR: no clips loaded.", file=sys.stderr)
        sys.exit(1)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    label_names = sorted(signer_index, key=signer_index.get)
    return X, y, label_names


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def run_classification(X, y, n_splits=5, seed=42):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=2000, solver="lbfgs",
                                      multi_class="auto", random_state=seed, C=1.0)),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    acc     = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    bal_acc = cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy")
    return {
        "accuracy_per_fold":          acc,
        "balanced_accuracy_per_fold": bal_acc,
        "mean_accuracy":              acc.mean(),
        "std_accuracy":               acc.std(),
        "mean_balanced_accuracy":     bal_acc.mean(),
        "std_balanced_accuracy":      bal_acc.std(),
        "chance":                     1.0 / len(np.unique(y)),
        "n_classes":                  len(np.unique(y)),
        "n_samples":                  len(y),
        "n_splits":                   n_splits,
    }


def print_results(res, label_names, feature_desc):
    acc   = res["mean_accuracy"]
    bal   = res["mean_balanced_accuracy"]
    chance = res["chance"]
    nc    = res["n_classes"]
    ratio = acc / chance

    print("\n" + "=" * 60)
    print("SMPL-X SIGNER IDENTITY PROBE – RESULTS")
    print("=" * 60)
    print(f"  Samples   : {res['n_samples']}")
    print(f"  Signers   : {nc}  ({', '.join(label_names)})")
    print(f"  Features  : {feature_desc}")
    print(f"  Classifier: logistic regression (linear, L2)")
    print(f"  CV        : {res['n_splits']}-fold stratified")
    print()
    print(f"  Accuracy (per fold)          : {' '.join(f'{s:.3f}' for s in res['accuracy_per_fold'])}")
    print(f"  Mean accuracy                : {acc:.4f}  ± {res['std_accuracy']:.4f}")
    print(f"  Mean balanced accuracy       : {bal:.4f}  ± {res['std_balanced_accuracy']:.4f}")
    print(f"  Chance baseline (1/{nc})     : {chance:.4f}")
    print()

    if ratio > 2.0:
        verdict = "STRONG bias: features carry clear signer identity."
    elif ratio > 1.3:
        verdict = "MODERATE bias: some signer-specific information."
    elif ratio > 1.05:
        verdict = "WEAK bias: marginal improvement over chance."
    else:
        verdict = "NO detectable bias: at chance level."

    print(f"  Accuracy / chance            : {ratio:.2f}x")
    print(f"  Verdict: {verdict}")
    print("=" * 60)


def save_confusion_matrix(X, y, label_names, out_path="smplx_signer_confusion.png", seed=42):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=2000, solver="lbfgs",
                                          multi_class="auto", random_state=seed, C=1.0)),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        y_pred = cross_val_predict(pipe, X, y, cv=cv)
        cm = confusion_matrix(y, y_pred, normalize="true")

        fig, ax = plt.subplots(figsize=(max(6, len(label_names)), max(5, len(label_names) - 1)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot(ax=ax, colorbar=True, xticks_rotation="vertical")
        ax.set_title("SMPL-X signer identity probe – normalised confusion matrix")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"\nConfusion matrix saved to: {out_path}")
    except Exception as exc:
        print(f"\n(Could not save confusion matrix: {exc})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["nsa", "signavatars"], default="nsa",
                        help="Which dataset to use (default: nsa)")
    parser.add_argument("--pose_dir", type=Path, default=None,
                        help="Override the default pose directory")
    parser.add_argument("--content_control", action="store_true",
                        help="Restrict to clips where ≥ --min_signers_per_content "
                             "signers produced the same content key")
    parser.add_argument("--min_signers_per_content", type=int, default=2)
    parser.add_argument("--hands_only", action="store_true",
                        help="Use only finger joint rotations: lhand_pose (45 dims) + "
                             "rhand_pose (45 dims) = 90 dims/frame → 180 after mean+std. "
                             "Excludes body posture, arm trajectory, and root orientation.")
    parser.add_argument("--include_shape", action="store_true",
                        help="Include SMPL-X shape (betas, 10 dims) in features. "
                             "WARNING: encodes body anatomy, will trivially inflate accuracy.")
    parser.add_argument("--include_expr", action="store_true",
                        help="Include facial expression coefficients (mean+std, 20 dims)")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--max_clips", type=int, default=None,
                        help="Cap clips loaded (for quick testing)")
    parser.add_argument("--confusion_matrix", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve directory
    if args.pose_dir is None:
        args.pose_dir = NSA_DEFAULT if args.dataset == "nsa" else SA_DEFAULT
    if not args.pose_dir.exists():
        print(f"ERROR: directory not found: {args.pose_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover clips
    print(f"Scanning {args.pose_dir} (dataset={args.dataset}) ...")
    if args.dataset == "nsa":
        records = discover_nsa(args.pose_dir)
    else:
        records = discover_signavatars(args.pose_dir)

    if not records:
        print("ERROR: no clips matched the expected naming convention.", file=sys.stderr)
        sys.exit(1)

    signers = set(r["signer_id"] for r in records)
    print(f"Found {len(records)} clips from {len(signers)} signer(s): {sorted(signers)}")

    if len(signers) < 2:
        print("ERROR: need at least 2 signers.", file=sys.stderr)
        sys.exit(1)

    if args.content_control:
        records = apply_content_control(records, args.min_signers_per_content)
        if not records:
            print("ERROR: no clips remain after content control.", file=sys.stderr)
            sys.exit(1)

    if args.max_clips:
        records = records[:args.max_clips]
        print(f"(limited to {args.max_clips} clips)")

    # Feature description string
    if args.hands_only:
        dims = 180
        parts = ["lhand_pose (15 finger joints × 3) + rhand_pose (15 finger joints × 3) "
                 "= 90 dims/frame, mean+std → 180 dims"]
    else:
        dims = 318
        parts = ["pose (root+body+lhand+rhand+jaw) mean+std → 318 dims"]
    if args.include_shape:
        parts.append("+shape mean → 10 dims")
        dims += 10
    if args.include_expr:
        parts.append("+expr mean+std → 20 dims")
        dims += 20
    feature_desc = ", ".join(parts) + f" = {dims} total"

    # Extract features
    X, y, label_names = extract_features(
        records, args.dataset, args.include_shape, args.include_expr, args.hands_only
    )
    print(f"\nFeature matrix: {X.shape}  –  labels: {y.shape}")

    unique, counts = np.unique(y, return_counts=True)
    print("Samples per signer:")
    for uid, cnt in zip(unique, counts):
        print(f"  signer {label_names[uid]}: {cnt} clips")

    min_count = counts.min()
    if min_count < args.n_splits:
        print(f"\nWARNING: signer with only {min_count} clips — reducing CV folds to {min_count}.")
        args.n_splits = min_count

    # Classify
    print(f"\nRunning {args.n_splits}-fold logistic regression ...")
    res = run_classification(X, y, n_splits=args.n_splits, seed=args.seed)

    print_results(res, label_names, feature_desc)

    if args.confusion_matrix:
        save_confusion_matrix(X, y, label_names, seed=args.seed)


if __name__ == "__main__":
    main()
