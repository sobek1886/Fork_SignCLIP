#!/usr/bin/env python3
"""
signer_classifier.py  –  Linear signer-identity probe on raw MediaPipe pose features.

Tests whether pose features retain signer-dependent bias (signing style, body
proportions, movement habits) after shoulder-width normalisation.

Method:
  1. Load .pose files from a directory; parse signer labels from filenames.
  2. Preprocess each pose with the standard SignCLIP pipeline:
       select components → normalise (shoulder width = 1) → zero legs
       → (T, 609) feature matrix per clip.
  3. Aggregate frames via mean + std pooling → 1218-dim fixed-size vector.
  4. Train logistic regression (linear) inside a stratified k-fold loop.
  5. Compare accuracy to the chance baseline (1 / n_signers).

Supported datasets
  how2sign  –  filename: {video_id}_{sent_idx}-{signer_id}-rgb_front.pose
  rwth      –  filename: {signer}_{letter}_{seq}_{camera}.pose

Content-confound note
  Because different signers may have signed different content, an above-chance
  result does not conclusively prove *style* bias; content confound is possible.
  Use --content_control to restrict to clips where at least --min_signers_per_content
  different signers produced the same content key – this controls for content.

Usage
  python signer_classifier.py --pose_dir /path/to/pose_cache --dataset how2sign
  python signer_classifier.py --pose_dir /path/to/rwth/poses  --dataset rwth
  python signer_classifier.py --pose_dir /path/to/pose_cache  \\
      --dataset how2sign --content_control --min_signers_per_content 2
"""

import os
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay


# ---------------------------------------------------------------------------
# Hardcoded face-contour points (avoids importing mediapipe)
# ---------------------------------------------------------------------------
FACEMESH_CONTOURS_POINTS = [
    '0', '7', '10', '13', '14', '17', '21', '33', '37', '39', '40', '46',
    '52', '53', '54', '55', '58', '61', '63', '65', '66', '67', '70', '78',
    '80', '81', '82', '84', '87', '88', '91', '93', '95', '103', '105', '107',
    '109', '127', '132', '133', '136', '144', '145', '146', '148', '149', '150',
    '152', '153', '154', '155', '157', '158', '159', '160', '161', '162', '163',
    '172', '173', '176', '178', '181', '185', '191', '234', '246', '249', '251',
    '263', '267', '269', '270', '276', '282', '283', '284', '285', '288', '291',
    '293', '295', '296', '297', '300', '308', '310', '311', '312', '314', '317',
    '318', '321', '323', '324', '332', '334', '336', '338', '356', '361', '362',
    '365', '373', '374', '375', '377', '378', '379', '380', '381', '382', '384',
    '385', '386', '387', '388', '389', '390', '397', '398', '400', '402', '405',
    '409', '415', '454', '466',
]


# ---------------------------------------------------------------------------
# Pose preprocessing  (mirrors demo_sign.py without loading any model)
# ---------------------------------------------------------------------------

def _normalization_info(pose_header):
    name = pose_header.components[0].name
    if name == "POSE_LANDMARKS":
        return pose_header.normalization_info(
            p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
            p2=("POSE_LANDMARKS", "LEFT_SHOULDER"),
        )
    if name == "BODY_135":
        return pose_header.normalization_info(
            p1=("BODY_135", "RShoulder"),
            p2=("BODY_135", "LShoulder"),
        )
    if name == "pose_keypoints_2d":
        return pose_header.normalization_info(
            p1=("pose_keypoints_2d", "RShoulder"),
            p2=("pose_keypoints_2d", "LShoulder"),
        )
    raise ValueError(f"Unknown pose schema: {name}")


def _hide_legs(pose):
    if pose.header.components[0].name == "POSE_LANDMARKS":
        leg_pts = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        indices = [
            pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
            for n in leg_pts
            for side in ["LEFT", "RIGHT"]
        ]
        pose.body.confidence[:, :, indices] = 0
        pose.body.data[:, :, indices, :] = 0
        return pose
    raise ValueError("Cannot hide legs for this pose schema")


def pose_to_features(pose) -> np.ndarray:
    """
    Apply the standard SignCLIP preprocessing and return a (T, 609) float32 array.
    Returns None if preprocessing fails.
    """
    try:
        pose = pose.get_components(
            ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
            {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS},
        )
        pose = pose.normalize(_normalization_info(pose.header))
        pose = _hide_legs(pose)
        feat = np.nan_to_num(pose.body.data)          # (T, 1, n_pts, 3)
        feat = feat[:, 0, :, :]                        # (T, n_pts, 3)
        feat = feat.reshape(feat.shape[0], -1)         # (T, 609)
        return feat.astype(np.float32)
    except Exception as exc:
        return None


def aggregate_features(feat: np.ndarray) -> np.ndarray:
    """Mean + std pooling over time → (1218,) vector."""
    return np.concatenate([feat.mean(axis=0), feat.std(axis=0)])


# ---------------------------------------------------------------------------
# Dataset-specific filename parsers
# ---------------------------------------------------------------------------

# How2Sign pose cache:  {video_id}_{sent_idx}-{signer_id}-rgb_front.pose
_H2S_RE = re.compile(r'^(.+)_(\d+)-(\d+)-rgb_front\.pose$')

# RWTH Fingerspelling:  {signer}_{letter}_{seq}_{camera}.pose
_RWTH_RE = re.compile(r'^(\d+)_(\d+)_(\d+)_(cam\d+)\.pose$')


def parse_how2sign(pose_dir: Path) -> list[dict]:
    records = []
    for p in sorted(pose_dir.glob("*.pose")):
        m = _H2S_RE.match(p.name)
        if not m:
            continue
        records.append({
            "path": p,
            "signer_id": m.group(3),
            "content_key": (m.group(1), m.group(2)),   # (video_id, sent_idx)
        })
    return records


def parse_rwth(pose_dir: Path) -> list[dict]:
    records = []
    for p in sorted(pose_dir.glob("*.pose")):
        m = _RWTH_RE.match(p.name)
        if not m:
            continue
        records.append({
            "path": p,
            "signer_id": m.group(1),
            "content_key": m.group(2),   # letter/gesture ID
        })
    return records


# ---------------------------------------------------------------------------
# Content control
# ---------------------------------------------------------------------------

def apply_content_control(records: list[dict], min_signers: int) -> list[dict]:
    """
    Retain only clips whose content_key was produced by at least `min_signers`
    different signers.  This rules out content-signer confound.
    """
    by_content: dict = defaultdict(set)
    for r in records:
        by_content[r["content_key"]].add(r["signer_id"])

    valid_keys = {k for k, signers in by_content.items() if len(signers) >= min_signers}
    filtered = [r for r in records if r["content_key"] in valid_keys]
    print(f"Content control: {len(valid_keys)} content keys with ≥{min_signers} signers → "
          f"{len(filtered)}/{len(records)} clips retained")
    return filtered


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_all_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess all pose files.

    Returns
    -------
    X : (N, 1218) float32
    y : (N,) int   – integer-encoded signer IDs
    """
    from pose_format import Pose

    signer_index: dict[str, int] = {}
    X_list, y_list = [], []
    failed = 0

    for rec in tqdm(records, desc="Loading poses"):
        try:
            with open(rec["path"], "rb") as fh:
                pose = Pose.read(fh.read())
        except Exception as exc:
            print(f"  WARNING: could not read {rec['path'].name}: {exc}")
            failed += 1
            continue

        feat = pose_to_features(pose)
        if feat is None or len(feat) == 0:
            failed += 1
            continue

        vec = aggregate_features(feat)

        sid = rec["signer_id"]
        if sid not in signer_index:
            signer_index[sid] = len(signer_index)

        X_list.append(vec)
        y_list.append(signer_index[sid])

    if failed:
        print(f"  (skipped {failed} files due to errors)")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    label_names = sorted(signer_index, key=signer_index.get)
    return X, y, label_names


# ---------------------------------------------------------------------------
# Classification and reporting
# ---------------------------------------------------------------------------

def run_classification(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                       max_iter: int = 1000, seed: int = 42) -> dict:
    """
    Stratified k-fold logistic regression.  Returns a dict of results.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=max_iter, solver="lbfgs",
                                      multi_class="auto", random_state=seed,
                                      C=1.0)),
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    bal_scores = cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy")

    n_classes = len(np.unique(y))
    chance = 1.0 / n_classes

    return {
        "accuracy_per_fold":          scores,
        "balanced_accuracy_per_fold": bal_scores,
        "mean_accuracy":              scores.mean(),
        "std_accuracy":               scores.std(),
        "mean_balanced_accuracy":     bal_scores.mean(),
        "std_balanced_accuracy":      bal_scores.std(),
        "chance":                     chance,
        "n_classes":                  n_classes,
        "n_samples":                  len(y),
        "n_splits":                   n_splits,
    }


def print_results(res: dict, label_names: list[str]):
    n = res["n_samples"]
    k = res["n_splits"]
    nc = res["n_classes"]
    chance = res["chance"]
    acc = res["mean_accuracy"]
    bal = res["mean_balanced_accuracy"]

    print("\n" + "=" * 60)
    print("SIGNER IDENTITY PROBE – RESULTS")
    print("=" * 60)
    print(f"  Samples   : {n}")
    print(f"  Signers   : {nc}  ({', '.join(label_names)})")
    print(f"  Features  : mean + std pooling → 1218 dims")
    print(f"  Classifier: logistic regression (linear, L2)")
    print(f"  CV        : {k}-fold stratified")
    print()
    print(f"  Accuracy (per fold)          : {' '.join(f'{s:.3f}' for s in res['accuracy_per_fold'])}")
    print(f"  Mean accuracy                : {acc:.4f}  ± {res['std_accuracy']:.4f}")
    print(f"  Mean balanced accuracy       : {bal:.4f}  ± {res['std_balanced_accuracy']:.4f}")
    print(f"  Chance baseline (1/{nc})     : {chance:.4f}")
    print()

    ratio = acc / chance
    if ratio > 2.0:
        verdict = "STRONG bias: classifier far exceeds chance → features carry clear signer identity."
    elif ratio > 1.3:
        verdict = "MODERATE bias: accuracy notably above chance → some signer-specific information."
    elif ratio > 1.05:
        verdict = "WEAK bias: marginal improvement over chance."
    else:
        verdict = "NO detectable bias: classifier at chance level."

    print(f"  Accuracy / chance            : {ratio:.2f}x")
    print(f"  Verdict: {verdict}")
    print("=" * 60)


def save_confusion_matrix(X, y, label_names, seed=42, out_path="signer_confusion.png"):
    """Fit on full data and save a normalised confusion matrix."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
        from sklearn.model_selection import cross_val_predict

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, solver="lbfgs",
                                          multi_class="auto", random_state=seed, C=1.0)),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        y_pred = cross_val_predict(pipe, X, y, cv=cv)

        cm = confusion_matrix(y, y_pred, normalize="true")
        fig, ax = plt.subplots(figsize=(max(6, len(label_names)), max(5, len(label_names) - 1)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot(ax=ax, colorbar=True, xticks_rotation="vertical")
        ax.set_title("Signer identity probe – normalised confusion matrix")
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
    parser.add_argument(
        "--pose_dir", type=Path,
        default=Path("/Users/piotr/Projects/Thesis/Data/How2Sign/pose_cache"),
        help="Directory containing .pose files",
    )
    parser.add_argument(
        "--dataset", choices=["how2sign", "rwth"], default="how2sign",
        help="Filename convention to use when parsing signer IDs (default: how2sign)",
    )
    parser.add_argument(
        "--content_control", action="store_true",
        help="Restrict to clips where ≥ --min_signers_per_content different signers "
             "produced the same content key (controls for content confound)",
    )
    parser.add_argument(
        "--min_signers_per_content", type=int, default=2,
        help="Minimum number of distinct signers per content key for content control (default: 2)",
    )
    parser.add_argument(
        "--n_splits", type=int, default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--max_clips", type=int, default=None,
        help="Cap the number of clips loaded (for quick testing)",
    )
    parser.add_argument(
        "--confusion_matrix", action="store_true",
        help="Save a normalised confusion matrix to signer_confusion.png",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    if not args.pose_dir.is_dir():
        print(f"ERROR: pose_dir does not exist: {args.pose_dir}", file=sys.stderr)
        sys.exit(1)

    # 1. Discover files
    print(f"Scanning {args.pose_dir} (dataset={args.dataset}) ...")
    if args.dataset == "how2sign":
        records = parse_how2sign(args.pose_dir)
    else:
        records = parse_rwth(args.pose_dir)

    if not records:
        print("ERROR: no .pose files matched the expected filename pattern.", file=sys.stderr)
        sys.exit(1)

    signers = set(r["signer_id"] for r in records)
    print(f"Found {len(records)} clips from {len(signers)} signer(s): {sorted(signers)}")

    if len(signers) < 2:
        print("ERROR: need at least 2 signers to train a classifier.", file=sys.stderr)
        sys.exit(1)

    # 2. Content control
    if args.content_control:
        records = apply_content_control(records, args.min_signers_per_content)
        if not records:
            print("ERROR: no clips remain after content control.", file=sys.stderr)
            sys.exit(1)

    # 3. Cap for quick testing
    if args.max_clips:
        records = records[:args.max_clips]
        print(f"(limited to {args.max_clips} clips)")

    # 4. Extract features
    X, y, label_names = extract_all_features(records)
    print(f"\nFeature matrix: {X.shape}  –  labels: {y.shape}")

    unique, counts = np.unique(y, return_counts=True)
    print("Samples per signer:")
    for uid, cnt in zip(unique, counts):
        print(f"  signer {label_names[uid]}: {cnt} clips")

    min_count = counts.min()
    if min_count < args.n_splits:
        print(f"\nWARNING: signer with only {min_count} clips – reducing CV folds to {min_count}.")
        args.n_splits = min_count

    # 5. Classify
    print(f"\nRunning {args.n_splits}-fold logistic regression ...")
    res = run_classification(X, y, n_splits=args.n_splits, seed=args.seed)

    # 6. Report
    print_results(res, label_names)

    # 7. Optional confusion matrix
    if args.confusion_matrix:
        save_confusion_matrix(X, y, label_names, seed=args.seed)


if __name__ == "__main__":
    main()
