#!/usr/bin/env python3
"""
Counterfactual analysis on WLASL: does SignCLIP encode sign content or signer style?

WLASL is an isolated ASL dataset (one sign per video) that SignCLIP was NOT trained on,
making it a clean out-of-domain probe.

Conditions:
  positive: same gloss, different signer  (content matches, style differs)
  negative: different gloss, same signer  (content differs, style matches)
  control:  different gloss, different signer  (both differ — baseline)

Interpretation:
  sim(positive) > sim(negative)  → model encodes sign content
  sim(negative) > sim(positive)  → model is signer-style biased

Usage:
  python wlasl_counterfactual_analysis.py [--n_pairs 200] [--model asl_finetune]
"""

import sys
import json
import random
import argparse
import pickle
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
from tqdm import tqdm
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WLASL_JSON = Path("/Users/piotr/Projects/Thesis/Data/WLASL/WLASL_v0.3.json")
POSE_CACHE = Path("/Users/piotr/Projects/Thesis/Data/WLASL/pose_cache")
RESULTS_PATH = Path("wlasl_counterfactual_results.pkl")
SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_clips(wlasl_json: Path, pose_cache: Path) -> list[dict]:
    """Build a list of clip records from WLASL metadata, keeping only cached poses."""
    with open(wlasl_json) as f:
        data = json.load(f)

    clips = []
    for entry in data:
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            if "video_id" not in inst:
                continue
            vid_id = int(inst["video_id"])
            pose_path = pose_cache / f"{vid_id:05d}.pose"
            if not pose_path.exists():
                continue
            clips.append({
                "video_id":  vid_id,
                "filename":  f"{vid_id:05d}.pose",
                "pose_path": pose_path,
                "signer_id": inst["signer_id"],
                "gloss":     gloss,
            })

    return clips


# ---------------------------------------------------------------------------
# Pair building
# ---------------------------------------------------------------------------

def build_pairs(clips: list[dict], n_pairs: int, seed: int) -> list[tuple]:
    """Build positive, negative, and control pairs.

    Returns list of (clip_a, clip_b, condition) tuples.
    """
    rng = random.Random(seed)

    by_gloss  = defaultdict(list)   # gloss → [clips]
    by_signer = defaultdict(list)   # signer_id → [clips]
    for c in clips:
        by_gloss[c["gloss"]].append(c)
        by_signer[c["signer_id"]].append(c)

    # -- Positive: same gloss, different signer --
    positive_pairs = []
    for gloss_clips in by_gloss.values():
        for a, b in combinations(gloss_clips, 2):
            if a["signer_id"] != b["signer_id"]:
                positive_pairs.append((a, b, "positive"))

    # -- Negative: different gloss, same signer --
    negative_pairs = []
    for signer_clips in by_signer.values():
        if len(signer_clips) < 2:
            continue
        shuffled = signer_clips[:]
        rng.shuffle(shuffled)
        for i in range(0, len(shuffled) - 1, 2):
            a, b = shuffled[i], shuffled[i + 1]
            if a["gloss"] != b["gloss"]:
                negative_pairs.append((a, b, "negative"))

    # -- Control: different gloss, different signer --
    all_shuffled = clips[:]
    rng.shuffle(all_shuffled)
    control_pairs = []
    for i in range(0, len(all_shuffled) - 1, 2):
        a, b = all_shuffled[i], all_shuffled[i + 1]
        if a["signer_id"] != b["signer_id"] and a["gloss"] != b["gloss"]:
            control_pairs.append((a, b, "control"))

    # Sample to n_pairs each
    positive_pairs = rng.sample(positive_pairs, min(n_pairs, len(positive_pairs)))
    negative_pairs = rng.sample(negative_pairs, min(n_pairs, len(negative_pairs)))
    control_pairs  = rng.sample(control_pairs,  min(n_pairs, len(control_pairs)))

    print(f"Pairs — positive: {len(positive_pairs)}, "
          f"negative: {len(negative_pairs)}, control: {len(control_pairs)}")
    return positive_pairs + negative_pairs + control_pairs


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def embed_clips(clips_needed: list[dict], model_name: str) -> dict[str, np.ndarray]:
    """Load poses from cache and embed; returns {filename → embedding}."""
    sys.path.insert(0, str(Path(__file__).parent))
    from pose_format import Pose
    from demo_sign import embed_pose

    embeddings = {}
    for clip in tqdm(clips_needed, desc="Embedding"):
        fname = clip["filename"]
        if fname in embeddings:
            continue
        try:
            with open(clip["pose_path"], "rb") as f:
                pose = Pose.read(f.read())
            emb = embed_pose(pose, model_name=model_name)   # (1, 768)
            embeddings[fname] = emb[0]
        except Exception as e:
            print(f"  WARNING: embedding failed for {fname}: {e}")

    return embeddings


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "(n.s.)"


def print_results(results_by_condition: dict[str, list[float]]):
    pos  = np.array(results_by_condition.get("positive", []))
    neg  = np.array(results_by_condition.get("negative", []))
    ctrl = np.array(results_by_condition.get("control",  []))

    W = 62
    print("\n" + "=" * W)
    print("WLASL COUNTERFACTUAL ANALYSIS RESULTS")
    print("=" * W)

    for cond, arr in [("positive", pos), ("negative", neg), ("control", ctrl)]:
        if len(arr) == 0:
            continue
        print(f"\n{cond.upper()} (n={len(arr)})")
        print(f"  mean ± std : {arr.mean():.4f} ± {arr.std():.4f}")
        print(f"  median     : {float(np.median(arr)):.4f}")
        print(f"  range      : [{arr.min():.4f}, {arr.max():.4f}]")

    print("\n" + "-" * W)
    print("STATISTICAL TESTS (Mann-Whitney U)")

    comparisons = [
        (pos,  neg,  "positive vs negative"),
        (pos,  ctrl, "positive vs control "),
        (neg,  ctrl, "negative vs control "),
    ]
    for a, b, label in comparisons:
        if len(a) < 2 or len(b) < 2:
            continue
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"\n  {label} : U={stat:.0f}, p={p:.4f} {sig_stars(p)}")

    print("\n" + "-" * W)
    print("INTERPRETATION")
    if len(pos) > 0 and len(neg) > 0:
        diff = float(pos.mean() - neg.mean())
        if diff > 0.02:
            verdict = "Content-focused: same-gloss embeddings are more similar across signers."
        elif diff < -0.02:
            verdict = "Style-biased: same-signer embeddings cluster regardless of gloss."
        else:
            verdict = "Ambiguous: no strong preference for content or signer style."
        print(f"\n  mean(positive) - mean(negative) = {diff:+.4f}")
        print(f"  → {verdict}")

    print("=" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n_pairs",   type=int,  default=1000,
                        help="Max pairs per condition (default: 200)")
    parser.add_argument("--model",     type=str,  default="asl_finetune",
                        choices=["default", "asl_citizen", "asl_finetune", "suisse"],
                        help="SignCLIP model to use (default: asl_finetune)")
    parser.add_argument("--results",   type=Path, default=RESULTS_PATH,
                        help="Output pickle path for raw similarity scores")
    parser.add_argument("--seed",      type=int,  default=SEED)
    parser.add_argument("--min_signers", type=int, default=2,
                        help="Only include glosses with at least this many cached signers (default: 2)")
    args = parser.parse_args()

    # 1. Load clips from cache
    print(f"Loading WLASL metadata and scanning pose cache...")
    clips = load_clips(WLASL_JSON, POSE_CACHE)

    # Filter to glosses with enough cached signers
    by_gloss = defaultdict(list)
    for c in clips:
        by_gloss[c["gloss"]].append(c)
    clips = [c for g, gc in by_gloss.items()
             if len(set(x["signer_id"] for x in gc)) >= args.min_signers
             for c in gc]

    n_glosses  = len(set(c["gloss"]     for c in clips))
    n_signers  = len(set(c["signer_id"] for c in clips))
    print(f"Found {len(clips)} cached clips, {n_glosses} glosses, {n_signers} signers")

    # 2. Build pairs
    all_pairs = build_pairs(clips, n_pairs=args.n_pairs, seed=args.seed)

    # 3. Identify unique clips needed
    clips_needed = {c["filename"]: c
                    for a, b, _ in all_pairs
                    for c in (a, b)}
    print(f"\nUnique clips to embed: {len(clips_needed)}")

    # 4. Embed
    print(f"Using model: {args.model}")
    embeddings = embed_clips(list(clips_needed.values()), model_name=args.model)
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
        results_by_condition[condition].append(cosine_similarity(emb_a, emb_b))

    if skipped:
        print(f"Skipped {skipped} pairs due to missing embeddings")

    # 6. Save raw results
    with open(args.results, "wb") as f:
        pickle.dump(dict(results_by_condition), f)
    print(f"\nRaw results saved to {args.results}")

    # 7. Report
    print_results(dict(results_by_condition))


if __name__ == "__main__":
    main()
