#!/usr/bin/env python3
"""
Cross-view analysis: is SignCLIP invariant to camera viewing angle?

Six conditions, varying view (front-side vs side-side), signer, and content:

  cv_pos          front-side  same signer   same content   (view changes, rest fixed)
  cv_neg          front-side  same signer   diff content   (view + content change)
  cv_ctrl         front-side  diff signer   diff content   (everything changes)
  cv_diff_signer  front-side  diff signer   same content   (view + signer change, content fixed)
  ss_same_content side-side   diff signer   same content   (signer changes within side view)
  ss_diff_content side-side   diff signer   diff content   (side-view random baseline)

Key comparisons:
  cv_pos  vs cv_neg          content effect within cross-view, same signer
  cv_pos  vs cv_diff_signer  signer effect within cross-view, same content
  cv_diff_signer vs ss_same_content  cross-view gap at diff-signer / same-content level
  ss_same_content vs ss_diff_content content effect within side-view
  ss_diff_content vs cv_ctrl         view clustering: are side-side random pairs inflated?

Usage:
  python crossview_analysis.py [--n_pairs 100] [--model asl_finetune]
"""

import os
import sys
import re
import random
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
from tqdm import tqdm

FRONT_DIR   = Path("/Users/piotr/Projects/Thesis/Data/How2Sign/val_rgb_front_clips")
SIDE_DIR    = Path("/Users/piotr/Projects/Thesis/Data/How2Sign/val_rgb_side_clips")
FRONT_CACHE = Path("/Users/piotr/Projects/Thesis/Data/How2Sign/pose_cache")       # reuse existing
SIDE_CACHE  = Path("/Users/piotr/Projects/Thesis/Data/How2Sign/pose_cache_side")
RESULTS_PATH = Path("crossview_results.pkl")
SEED = 42

# Ordered for display
CONDITIONS = [
    "cv_pos",
    "cv_neg",
    "cv_ctrl",
    "cv_diff_signer",
    "ss_same_content",
    "ss_diff_content",
]

CONDITION_LABELS = {
    "cv_pos":          "front-side | same signer | same content",
    "cv_neg":          "front-side | same signer | diff content",
    "cv_ctrl":         "front-side | diff signer | diff content",
    "cv_diff_signer":  "front-side | diff signer | same content  [NEW]",
    "ss_same_content": "side-side  | diff signer | same content  [NEW]",
    "ss_diff_content": "side-side  | diff signer | diff content  [NEW]",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_dir(directory: Path, view: str) -> dict:
    """Returns {(video_id, sent_idx, signer_id): clip_dict}."""
    pattern = re.compile(rf'^(.+)_(\d+)-(\d+)-rgb_{view}\.mp4$')
    clips = {}
    for fname in sorted(os.listdir(directory)):
        m = pattern.match(fname)
        if m:
            key = (m.group(1), m.group(2), m.group(3))
            clips[key] = {
                "path": directory / fname,
                "filename": fname,
                "video_id": m.group(1),
                "sent_idx": m.group(2),
                "signer_id": m.group(3),
                "view": view,
            }
    return clips


# ---------------------------------------------------------------------------
# Pair building
# ---------------------------------------------------------------------------

def build_all_pairs(front: dict, side: dict, n_pairs: int, seed: int) -> list[tuple]:
    rng = random.Random(seed)
    matched_keys = sorted(set(front) & set(side))

    # Indexes
    by_signer   = defaultdict(list)   # signer_id  → [key]
    by_sentence = defaultdict(list)   # (vid, sent) → [key]
    for key in matched_keys:
        by_signer[key[2]].append(key)
        by_sentence[(key[0], key[1])].append(key)

    pairs = defaultdict(list)

    # ------------------------------------------------------------------
    # CROSS-VIEW pairs (one front clip, one side clip)
    # ------------------------------------------------------------------

    # cv_pos: same content, same signer
    for key in matched_keys:
        pairs["cv_pos"].append((front[key], side[key]))

    # cv_neg: different content, same signer
    for signer_id, keys in by_signer.items():
        if len(keys) < 2:
            continue
        shuffled = keys[:]
        rng.shuffle(shuffled)
        for i in range(0, len(shuffled) - 1, 2):
            ka, kb = shuffled[i], shuffled[i + 1]
            if (ka[0], ka[1]) != (kb[0], kb[1]):
                pairs["cv_neg"].append((front[ka], side[kb]))

    # cv_ctrl: different content, different signer
    all_shuf = matched_keys[:]
    rng.shuffle(all_shuf)
    for i in range(0, len(all_shuf) - 1, 2):
        ka, kb = all_shuf[i], all_shuf[i + 1]
        if ka[2] != kb[2] and (ka[0], ka[1]) != (kb[0], kb[1]):
            pairs["cv_ctrl"].append((front[ka], side[kb]))

    # cv_diff_signer: same content, different signer (view + signer change, content fixed)
    for sent_keys in by_sentence.values():
        if len(sent_keys) < 2:
            continue
        for i in range(len(sent_keys)):
            for j in range(len(sent_keys)):
                ka, kb = sent_keys[i], sent_keys[j]
                if ka[2] != kb[2]:   # different signers
                    pairs["cv_diff_signer"].append((front[ka], side[kb]))

    # ------------------------------------------------------------------
    # SAME-VIEW SIDE pairs (both side clips)
    # ------------------------------------------------------------------

    # ss_same_content: same content, different signer
    for sent_keys in by_sentence.values():
        if len(sent_keys) < 2:
            continue
        for i in range(len(sent_keys)):
            for j in range(i + 1, len(sent_keys)):
                ka, kb = sent_keys[i], sent_keys[j]
                if ka[2] != kb[2]:
                    pairs["ss_same_content"].append((side[ka], side[kb]))

    # ss_diff_content: different content, different signer
    rng.shuffle(all_shuf)
    for i in range(0, len(all_shuf) - 1, 2):
        ka, kb = all_shuf[i], all_shuf[i + 1]
        if ka[2] != kb[2] and (ka[0], ka[1]) != (kb[0], kb[1]):
            pairs["ss_diff_content"].append((side[ka], side[kb]))

    # Sample and label
    result = []
    for cond in CONDITIONS:
        pool = pairs[cond]
        sampled = rng.sample(pool, min(n_pairs, len(pool)))
        for a, b in sampled:
            result.append((a, b, cond))
        print(f"  {cond:<22} {len(sampled):>4} pairs  (pool: {len(pool)})")

    return result


# ---------------------------------------------------------------------------
# Pose / embedding
# ---------------------------------------------------------------------------

def get_pose(clip: dict, cache_dir: Path):
    from pose_format import Pose
    from counterfactual_analysis import extract_pose_from_video

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


def embed_all_clips(clips_needed: list[dict], model_name: str) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(Path(__file__).parent))
    from demo_sign import embed_pose

    embeddings = {}
    for clip in tqdm(clips_needed, desc="Extracting poses & embedding"):
        fname = clip["filename"]
        if fname in embeddings:
            continue
        cache_dir = FRONT_CACHE if clip["view"] == "front" else SIDE_CACHE
        pose = get_pose(clip, cache_dir)
        if pose is None:
            print(f"  WARNING: pose extraction failed for {fname}")
            continue
        try:
            emb = embed_pose(pose, model_name=model_name)
            embeddings[fname] = emb[0]
        except Exception as e:
            print(f"  WARNING: embedding failed for {fname}: {e}")

    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def cohens_d(a, b):
    pooled = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled


def mean_ci(a, b):
    diff = np.mean(a) - np.mean(b)
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    return diff, diff - 1.96 * se, diff + 1.96 * se


def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def print_results(r: dict[str, list[float]]):
    arrays = {c: np.array(r[c]) for c in CONDITIONS if c in r}

    W = 62
    print("\n" + "=" * W)
    print("CROSS-VIEW ANALYSIS — ALL CONDITIONS")
    print("=" * W)
    print(f"\n{'Condition':<22}  {'n':>4}  {'mean':>7}  {'std':>7}  {'median':>7}")
    print("-" * W)
    for cond in CONDITIONS:
        if cond not in arrays:
            continue
        a = arrays[cond]
        print(f"  {cond:<20}  {len(a):>4}  {a.mean():>7.4f}  {a.std():>7.4f}  {np.median(a):>7.4f}")

    print("\n" + "=" * W)
    print("KEY COMPARISONS")
    print("=" * W)

    comparisons = [
        ("cv_pos",         "cv_neg",          "content effect, cross-view, same signer"),
        ("cv_pos",         "cv_diff_signer",  "signer effect, cross-view, same content"),
        ("cv_pos",         "cv_ctrl",         "content+signer effect, cross-view"),
        ("cv_diff_signer", "ss_same_content", "cross-view gap (diff-signer, same content)"),
        ("ss_same_content","ss_diff_content", "content effect within side-side"),
        ("ss_diff_content","cv_ctrl",         "view clustering: side-side vs cross-view random"),
    ]

    for ca, cb, label in comparisons:
        if ca not in arrays or cb not in arrays:
            continue
        a, b = arrays[ca], arrays[cb]
        d = cohens_d(a, b)
        diff, lo, hi = mean_ci(a, b)
        _, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        print(f"\n  {label}")
        print(f"    {ca} ({a.mean():.4f})  vs  {cb} ({b.mean():.4f})")
        print(f"    d={d:+.3f}  Δ={diff:+.4f} [{lo:+.4f}, {hi:+.4f}]  p={p:.4f} {sig_stars(p)}")

    # Reference numbers from previous experiment
    prev_path = Path("counterfactual_results.pkl")
    if prev_path.exists():
        with open(prev_path, "rb") as f:
            prev = pickle.load(f)
        ff_same = np.array(prev["positive"])  # front-front, diff signer, same content
        ff_diff = np.array(prev["control"])   # front-front, diff signer, diff content

        print("\n" + "-" * W)
        print("REFERENCE (Experiment 1 — front-front, diff signer)")
        print(f"  front-front same content: {ff_same.mean():.4f} ± {ff_same.std():.4f}  (n={len(ff_same)})")
        print(f"  front-front diff content: {ff_diff.mean():.4f} ± {ff_diff.std():.4f}  (n={len(ff_diff)})")

        if "ss_same_content" in arrays and "cv_diff_signer" in arrays:
            ss = arrays["ss_same_content"]
            cv_ds = arrays["cv_diff_signer"]
            print("\n  View degradation for same-content, diff-signer pairs:")
            print(f"    front-front  → {ff_same.mean():.4f}")
            print(f"    side-side    → {ss.mean():.4f}   gap = {ff_same.mean()-ss.mean():+.4f}")
            print(f"    front-side   → {cv_ds.mean():.4f}   gap = {ff_same.mean()-cv_ds.mean():+.4f}")

    print("\n" + "=" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_pairs",    type=int,  default=100)
    parser.add_argument("--model",      type=str,  default="asl_finetune",
                        choices=["default", "asl_citizen", "asl_finetune", "suisse"])
    parser.add_argument("--max_frames", type=int,  default=256)
    parser.add_argument("--results",    type=Path, default=RESULTS_PATH)
    parser.add_argument("--seed",       type=int,  default=SEED)
    args = parser.parse_args()

    print("Parsing clip directories...")
    front = parse_dir(FRONT_DIR, "front")
    side  = parse_dir(SIDE_DIR,  "side")
    matched = len(set(front) & set(side))
    print(f"  Frontal: {len(front)}  |  Side: {len(side)}  |  Matched: {matched}\n")

    print("Building pairs...")
    all_pairs = build_all_pairs(front, side, n_pairs=args.n_pairs, seed=args.seed)

    clips_needed = {}
    for a, b, _ in all_pairs:
        clips_needed[a["filename"]] = a
        clips_needed[b["filename"]] = b
    n_side_new = sum(1 for c in clips_needed.values()
                     if c["view"] == "side" and not
                     (SIDE_CACHE / c["filename"].replace(".mp4", ".pose")).exists())
    print(f"\nUnique clips to embed: {len(clips_needed)}  "
          f"(~{n_side_new} new side-view poses to extract)")

    print(f"\nUsing model: {args.model}")
    embeddings = embed_all_clips(list(clips_needed.values()), model_name=args.model)
    print(f"Successfully embedded {len(embeddings)} / {len(clips_needed)} clips")

    results: dict[str, list] = defaultdict(list)
    skipped = 0
    for a, b, cond in all_pairs:
        ea = embeddings.get(a["filename"])
        eb = embeddings.get(b["filename"])
        if ea is None or eb is None:
            skipped += 1
            continue
        results[cond].append(cosine_similarity(ea, eb))

    if skipped:
        print(f"Skipped {skipped} pairs due to missing embeddings")

    with open(args.results, "wb") as f:
        pickle.dump(dict(results), f)
    print(f"Raw results saved to {args.results}")

    print_results(dict(results))


if __name__ == "__main__":
    main()
