#!/usr/bin/env python3
"""
Counterfactual analysis: does SignCLIP encode sign content or signer style?

Conditions:
  positive: same sentence, different signer  (content matches, style differs)
  negative: different sentence, same signer  (content differs, style matches)
  control:  different sentence, different signer (both differ — baseline)

Interpretation:
  sim(positive) > sim(negative) → model is content/linguistics-focused
  sim(negative) > sim(positive) → model is kinematic-style biased

NOTE: How2Sign is sentence-level continuous signing, not isolated signs.
This tests sentence-level content vs. kinematic idiolect, not individual sign identity.

Usage:
  python counterfactual_analysis.py [--n_pairs 100] [--model] [--max_clips N]
"""

import os
import sys
import re
import random
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_VIDEOS_DIR = Path("/Users/piotr/Projects/Thesis/Data/How2Sign/val_rgb_front_clips")
POSE_CACHE_DIR = Path("/Users/piotr/Projects/Thesis/Data/How2Sign/pose_cache")
RESULTS_PATH = Path("counterfactual_results.pkl")
SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_raw_videos(raw_videos_dir: Path) -> list[dict]:
    """Parse filenames into structured clip records.

    Filename format: {VIDEO_ID}_{SENT_IDX}-{SIGNER_ID}-rgb_front.mp4
    """
    pattern = re.compile(r'^(.+)_(\d+)-(\d+)-rgb_front\.mp4$')
    clips = []
    for fname in sorted(os.listdir(raw_videos_dir)):
        m = pattern.match(fname)
        if m:
            clips.append({
                "path": raw_videos_dir / fname,
                "filename": fname,
                "video_id": m.group(1),
                "sent_idx": m.group(2),
                "signer_id": m.group(3),
                "sentence_key": (m.group(1), m.group(2)),  # (video_id, sent_idx)
            })
    return clips


def build_pairs(clips: list[dict], n_pairs: int, seed: int) -> list[tuple]:
    """Build positive, negative, and control pairs.

    Returns list of (clip_a, clip_b, condition) tuples.
    """
    rng = random.Random(seed)

    by_sentence = defaultdict(list)   # (video_id, sent_idx) → [clips]
    by_signer = defaultdict(list)     # signer_id → [clips]
    for c in clips:
        by_sentence[c["sentence_key"]].append(c)
        by_signer[c["signer_id"]].append(c)

    # -- Positive: same sentence, different signer --
    positive_pairs = []
    for sent_clips in by_sentence.values():
        if len(sent_clips) >= 2:
            for i in range(len(sent_clips)):
                for j in range(i + 1, len(sent_clips)):
                    a, b = sent_clips[i], sent_clips[j]
                    if a["signer_id"] != b["signer_id"]:
                        positive_pairs.append((a, b, "positive"))

    # -- Negative: different sentence, same signer --
    negative_pairs = []
    for signer_clips in by_signer.values():
        if len(signer_clips) >= 2:
            shuffled = signer_clips[:]
            rng.shuffle(shuffled)
            for i in range(0, len(shuffled) - 1, 2):
                a, b = shuffled[i], shuffled[i + 1]
                if a["sentence_key"] != b["sentence_key"]:
                    negative_pairs.append((a, b, "negative"))

    # -- Control: different sentence, different signer --
    all_shuffled = clips[:]
    rng.shuffle(all_shuffled)
    control_pairs = []
    for i in range(0, len(all_shuffled) - 1, 2):
        a, b = all_shuffled[i], all_shuffled[i + 1]
        if a["signer_id"] != b["signer_id"] and a["sentence_key"] != b["sentence_key"]:
            control_pairs.append((a, b, "control"))

    # Sample to n_pairs each
    positive_pairs = rng.sample(positive_pairs, min(n_pairs, len(positive_pairs)))
    negative_pairs = rng.sample(negative_pairs, min(n_pairs, len(negative_pairs)))
    control_pairs  = rng.sample(control_pairs,  min(n_pairs, len(control_pairs)))

    print(f"Pairs — positive: {len(positive_pairs)}, negative: {len(negative_pairs)}, control: {len(control_pairs)}")
    return positive_pairs + negative_pairs + control_pairs


# ---------------------------------------------------------------------------
# Pose extraction
# ---------------------------------------------------------------------------

def extract_pose_from_video(video_path: Path):
    """Extract pose from an mp4 using MediaPipe Holistic via pose_format."""
    from pose_format.utils.holistic import load_holistic

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return None

    return load_holistic(frames, fps=fps, width=width, height=height, progress=False)


def get_pose(clip: dict, cache_dir: Path):
    """Return pose for a clip, using cache if available."""
    from pose_format import Pose

    cache_path = cache_dir / clip["filename"].replace(".mp4", ".pose")
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return Pose.read(f.read())

    pose = extract_pose_from_video(clip["path"])
    if pose is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pose.write(f)
    return pose


# ---------------------------------------------------------------------------
# Embedding and similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def embed_clips(clips_needed: list[dict], model_name: str, cache_dir: Path,
                max_frames: int) -> dict[str, np.ndarray]:
    """Embed all required clips; return {filename → embedding}."""
    # Import here to avoid slow startup when not needed
    sys.path.insert(0, str(Path(__file__).parent))
    from demo_sign import embed_pose, preprocess_pose

    embeddings = {}
    for clip in tqdm(clips_needed, desc="Extracting poses & embedding"):
        fname = clip["filename"]
        if fname in embeddings:
            continue

        pose = get_pose(clip, cache_dir)
        if pose is None:
            print(f"  WARNING: could not extract pose for {fname}")
            continue

        try:
            emb = embed_pose(pose, model_name=model_name)  # shape (1, 768)
            embeddings[fname] = emb[0]
        except Exception as e:
            print(f"  WARNING: embedding failed for {fname}: {e}")

    return embeddings


# ---------------------------------------------------------------------------
# Statistics and reporting
# ---------------------------------------------------------------------------

def report(condition: str, sims: list[float]) -> dict:
    arr = np.array(sims)
    return {
        "condition": condition,
        "n": len(arr),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def print_results(results_by_condition: dict[str, list[float]]):
    pos = np.array(results_by_condition["positive"])
    neg = np.array(results_by_condition["negative"])
    ctrl = np.array(results_by_condition["control"])

    print("\n" + "=" * 60)
    print("COUNTERFACTUAL ANALYSIS RESULTS")
    print("=" * 60)

    for cond, sims in results_by_condition.items():
        r = report(cond, sims)
        print(f"\n{cond.upper()} (n={r['n']})")
        print(f"  mean ± std : {r['mean']:.4f} ± {r['std']:.4f}")
        print(f"  median     : {r['median']:.4f}")
        print(f"  range      : [{r['min']:.4f}, {r['max']:.4f}]")

    print("\n" + "-" * 60)
    print("STATISTICAL TESTS (Wilcoxon rank-sum / Mann-Whitney U)")

    if len(pos) >= 2 and len(neg) >= 2:
        stat, p = stats.mannwhitneyu(pos, neg, alternative='two-sided')
        print(f"\n  positive vs negative : U={stat:.0f}, p={p:.4f}", end="")
        if p < 0.001:
            print(" ***")
        elif p < 0.01:
            print(" **")
        elif p < 0.05:
            print(" *")
        else:
            print(" (n.s.)")

    if len(pos) >= 2 and len(ctrl) >= 2:
        stat, p = stats.mannwhitneyu(pos, ctrl, alternative='two-sided')
        print(f"  positive vs control  : U={stat:.0f}, p={p:.4f}", end="")
        if p < 0.001:
            print(" ***")
        elif p < 0.01:
            print(" **")
        elif p < 0.05:
            print(" *")
        else:
            print(" (n.s.)")

    if len(neg) >= 2 and len(ctrl) >= 2:
        stat, p = stats.mannwhitneyu(neg, ctrl, alternative='two-sided')
        print(f"  negative vs control  : U={stat:.0f}, p={p:.4f}", end="")
        if p < 0.001:
            print(" ***")
        elif p < 0.01:
            print(" **")
        elif p < 0.05:
            print(" *")
        else:
            print(" (n.s.)")

    print("\n" + "-" * 60)
    print("INTERPRETATION")

    if len(pos) > 0 and len(neg) > 0:
        diff = np.mean(pos) - np.mean(neg)
        if diff > 0.02:
            verdict = "Content-focused: same sentence embeddings are more similar across signers."
        elif diff < -0.02:
            verdict = "Style-biased: same-signer embeddings cluster regardless of content."
        else:
            verdict = "Ambiguous: no strong preference for content or signer style."
        print(f"\n  mean(positive) - mean(negative) = {diff:+.4f}")
        print(f"  → {verdict}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_pairs",    type=int,   default=100,
                        help="Max pairs per condition (default: 100)")
    parser.add_argument("--model",      type=str,   default="asl_finetune",
                        choices=["default", "asl_citizen", "asl_finetune", "suisse"],
                        help="SignCLIP model to use (default: asl_finetune)")
    parser.add_argument("--max_frames", type=int,   default=256,
                        help="Truncate pose sequences to this many frames (default: 256)")
    parser.add_argument("--max_clips",  type=int,   default=None,
                        help="Limit total clips processed (for quick testing)")
    parser.add_argument("--cache_dir",  type=Path,  default=POSE_CACHE_DIR,
                        help="Directory for cached .pose files")
    parser.add_argument("--results",    type=Path,  default=RESULTS_PATH,
                        help="Output pickle path for raw similarity scores")
    parser.add_argument("--seed",       type=int,   default=SEED)
    args = parser.parse_args()

    # 1. Parse clip filenames
    print(f"Scanning {RAW_VIDEOS_DIR} ...")
    clips = parse_raw_videos(RAW_VIDEOS_DIR)
    print(f"Found {len(clips)} clips, {len(set(c['signer_id'] for c in clips))} signers, "
          f"{len(set(c['sentence_key'] for c in clips))} unique sentences")

    if args.max_clips:
        clips = clips[:args.max_clips]
        print(f"  (limited to first {args.max_clips} clips for testing)")

    # 2. Build pairs
    all_pairs = build_pairs(clips, n_pairs=args.n_pairs, seed=args.seed)

    # 3. Identify unique clips needed
    clips_needed = {}
    for a, b, _ in all_pairs:
        clips_needed[a["filename"]] = a
        clips_needed[b["filename"]] = b
    clips_needed = list(clips_needed.values())
    print(f"\nUnique clips to embed: {len(clips_needed)}")

    # 4. Extract poses and embed
    print(f"\nUsing model: {args.model}")
    embeddings = embed_clips(clips_needed, model_name=args.model,
                             cache_dir=args.cache_dir, max_frames=args.max_frames)
    print(f"Successfully embedded {len(embeddings)} / {len(clips_needed)} clips")

    # 5. Compute similarities per condition
    results_by_condition: dict[str, list[float]] = defaultdict(list)
    skipped = 0
    for a, b, condition in all_pairs:
        emb_a = embeddings.get(a["filename"])
        emb_b = embeddings.get(b["filename"])
        if emb_a is None or emb_b is None:
            skipped += 1
            continue
        sim = cosine_similarity(emb_a, emb_b)
        results_by_condition[condition].append(sim)

    if skipped:
        print(f"Skipped {skipped} pairs due to missing embeddings")

    # 6. Save raw results
    with open(args.results, "wb") as f:
        pickle.dump(dict(results_by_condition), f)
    print(f"\nRaw results saved to {args.results}")

    # 7. Print stats and interpretation
    print_results(dict(results_by_condition))


if __name__ == "__main__":
    main()
