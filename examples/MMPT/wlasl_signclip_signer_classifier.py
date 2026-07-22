#!/usr/bin/env python3
"""
wlasl_signclip_signer_classifier.py  –  Cross-gloss signer retrieval probe on WLASL
using SignCLIP embeddings.

The original per-gloss LOO-CV design (used in wlasl_mediapipe_signer_classifier.py)
breaks for WLASL because most glosses have exactly one video per signer: the correct
signer label is removed from the training set at test time, so 1-NN always predicts
wrong by construction.

This script uses a cross-gloss LOO retrieval design instead:

  For each video v (signer s, gloss g) as a query:
    - Build a prototype for every signer t as the mean embedding of all their
      videos EXCEPT those of gloss g (to avoid content contamination).
    - 1-NN: predict the signer whose prototype is most similar to v.
    - Correct if the nearest prototype belongs to signer s.

  Chance level = 1 / (number of signers with a valid prototype).

This is content-controlled by construction: the prototypes exclude the query
gloss, so a correct match must rely on cross-gloss kinematic style.

Usage
  python wlasl_signclip_signer_classifier.py
  python wlasl_signclip_signer_classifier.py --model asl_finetune --min_signer_videos 10
  python wlasl_signclip_signer_classifier.py --save_embeddings embs.pkl
  python wlasl_signclip_signer_classifier.py --load_embeddings embs.pkl
"""

import sys
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WLASL_JSON = Path("/Users/piotr/Projects/Thesis/Data/WLASL/WLASL_v0.3.json")
POSE_CACHE = Path("/Users/piotr/Projects/Thesis/Data/WLASL/pose_cache")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _load_pose(args):
    """Load one .pose file; returns (video_id, Pose) or (video_id, None)."""
    vid_id, cache_path = args
    try:
        from pose_format import Pose
        with open(cache_path, "rb") as f:
            return vid_id, Pose.read(f.read())
    except Exception:
        return vid_id, None


def embed_all(video_ids: list[int], model_name: str, cache_dir: Path,
              batch_size: int = 64, io_workers: int = 8) -> dict[int, np.ndarray]:
    """Embed all cached videos; returns {video_id → 768-dim embedding}.

    Loads .pose files in parallel (threads, I/O bound) then embeds in batches
    (single model instance, avoids MPS contention from multiple processes).
    """
    import concurrent.futures
    sys.path.insert(0, str(Path(__file__).parent))
    from demo_sign import embed_pose

    tasks = [(vid_id, cache_dir / f"{vid_id:05d}.pose")
             for vid_id in video_ids
             if (cache_dir / f"{vid_id:05d}.pose").exists()]

    # -- Load poses in parallel (I/O bound) --
    poses: list[tuple[int, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as ex:
        for vid_id, pose in tqdm(
            ex.map(_load_pose, tasks), total=len(tasks), desc="Loading poses"
        ):
            if pose is not None:
                poses.append((vid_id, pose))

    # -- Embed in batches (single model forward pass per batch) --
    embeddings = {}
    for i in tqdm(range(0, len(poses), batch_size), desc="Embedding batches"):
        batch = poses[i : i + batch_size]
        ids   = [vid_id for vid_id, _ in batch]
        ps    = [pose   for _,      pose in batch]
        try:
            embs = embed_pose(ps, model_name=model_name)   # (B, 768)
            for vid_id, emb in zip(ids, embs):
                embeddings[vid_id] = emb
        except Exception as e:
            tqdm.write(f"  WARNING: batch starting {ids[0]:05d}: {e}")

    return embeddings


# ---------------------------------------------------------------------------
# Cross-gloss LOO retrieval
# ---------------------------------------------------------------------------

def cosine_sim_matrix(queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """(Q, D) × (K, D) → (Q, K) cosine similarity matrix."""
    q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    k = keys   / (np.linalg.norm(keys,    axis=1, keepdims=True) + 1e-8)
    return q @ k.T


def run_retrieval(
    videos: list[dict],           # [{"video_id", "signer_id", "gloss", "emb"}]
    min_signer_videos: int,
) -> list[dict]:
    """
    For each query video v (signer s, gloss g):
      - prototype_t = mean embedding of all videos by signer t where gloss != g
      - predict = argmax_t cosine_sim(v, prototype_t)
      - record hit / miss / rank

    Returns list of result dicts, one per query.
    """
    # Index: signer_id → list of video dicts
    by_signer: dict[int, list[dict]] = defaultdict(list)
    for v in videos:
        by_signer[v["signer_id"]].append(v)

    # Only signers with enough videos can serve as queries
    query_signers = {sid for sid, vlist in by_signer.items()
                     if len(vlist) >= min_signer_videos}

    results = []

    for v in tqdm(videos, desc="Retrieval"):
        sid = v["signer_id"]
        gloss = v["gloss"]

        if sid not in query_signers:
            continue

        # Build per-signer prototypes excluding gloss g
        prototypes = {}   # signer_id → mean embedding
        for t, vlist in by_signer.items():
            other = [x["emb"] for x in vlist if x["gloss"] != gloss]
            if other:
                prototypes[t] = np.mean(other, axis=0)

        if sid not in prototypes:
            # Query signer has no other-gloss videos after exclusion
            continue

        proto_ids  = sorted(prototypes.keys())
        proto_mat  = np.stack([prototypes[t] for t in proto_ids])   # (T, D)
        query_emb  = v["emb"][np.newaxis, :]                        # (1, D)

        sims       = cosine_sim_matrix(query_emb, proto_mat)[0]     # (T,)
        pred_idx   = int(np.argmax(sims))
        pred_signer = proto_ids[pred_idx]
        correct    = int(pred_signer == sid)

        # Rank of the correct signer (1 = best)
        true_idx  = proto_ids.index(sid)
        rank      = int(np.sum(sims >= sims[true_idx]))

        results.append({
            "video_id":   v["video_id"],
            "signer_id":  sid,
            "gloss":      gloss,
            "pred_signer": pred_signer,
            "correct":    correct,
            "rank":       rank,
            "n_pool":     len(proto_ids),
            "sim_correct": float(sims[true_idx]),
            "sim_pred":   float(sims[pred_idx]),
        })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results: list[dict], min_signer_videos: int):
    if not results:
        print("No results.")
        return

    n_pool = results[0]["n_pool"]   # approximately constant
    correct = [r["correct"] for r in results]
    ranks   = [r["rank"]    for r in results]

    overall_acc   = np.mean(correct)
    mean_rank     = np.mean(ranks)
    chance        = 1.0 / n_pool
    ratio         = overall_acc / chance

    # MRR
    mrr = np.mean([1.0 / r for r in ranks])

    # Per-signer breakdown
    by_signer: dict[int, list] = defaultdict(list)
    for r in results:
        by_signer[r["signer_id"]].append(r["correct"])

    W = 62
    print("\n" + "=" * W)
    print("WLASL SIGNCLIP CROSS-GLOSS SIGNER RETRIEVAL PROBE")
    print("=" * W)
    print(f"\n  Design     : LOO cross-gloss 1-NN prototype retrieval")
    print(f"  Pool size  : {n_pool} signers")
    print(f"  Chance     : 1/{n_pool} = {chance:.4f}")
    print(f"  Queries    : {len(results)} videos  "
          f"(signers with ≥{min_signer_videos} videos)")
    print()
    print(f"  Top-1 accuracy : {overall_acc:.4f}  ({ratio:.2f}x chance)")
    print(f"  Mean rank      : {mean_rank:.2f}  (chance = {n_pool / 2:.1f})")
    print(f"  MRR            : {mrr:.4f}  (chance = {1 / n_pool:.4f})")

    print(f"\n{'Signer':>8}  {'Queries':>8}  {'Accuracy':>10}  {'Ratio':>7}")
    print("-" * 42)
    for sid, hits in sorted(by_signer.items(), key=lambda x: -len(x[1])):
        acc = np.mean(hits)
        print(f"  {sid:>6}  {len(hits):>8}  {acc:>10.3f}  {acc/chance:>6.2f}x")

    print("=" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="asl_finetune",
                        choices=["default", "asl_citizen", "asl_finetune", "suisse"])
    parser.add_argument("--min_signer_videos", type=int, default=5,
                        help="Minimum cached videos for a signer to serve as a query "
                             "(default: 5)")
    parser.add_argument("--pose_cache", type=Path, default=POSE_CACHE)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Poses per model forward pass (default: 64)")
    parser.add_argument("--io_workers", type=int, default=8,
                        help="Threads for parallel .pose file loading (default: 8)")
    parser.add_argument("--save_embeddings", type=Path, default=None,
                        help="Save computed embeddings to this pickle file for reuse")
    parser.add_argument("--load_embeddings", type=Path, default=None,
                        help="Load pre-computed embeddings instead of re-embedding")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    print(f"Loading WLASL metadata ...")
    with open(WLASL_JSON) as f:
        data = json.load(f)

    vid_to_meta: dict[int, dict] = {}
    for entry in data:
        for inst in entry["instances"]:
            if "video_id" in inst:
                vid_to_meta[int(inst["video_id"])] = {
                    "gloss":     entry["gloss"],
                    "signer_id": inst["signer_id"],
                }

    # Keep only cached videos
    cached_ids = sorted(
        vid for vid in vid_to_meta
        if (args.pose_cache / f"{vid:05d}.pose").exists()
    )
    print(f"Cached videos: {len(cached_ids)}  across "
          f"{len(set(vid_to_meta[v]['signer_id'] for v in cached_ids))} signers, "
          f"{len(set(vid_to_meta[v]['gloss'] for v in cached_ids))} glosses")

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------
    if args.load_embeddings and args.load_embeddings.exists():
        print(f"Loading embeddings from {args.load_embeddings} ...")
        with open(args.load_embeddings, "rb") as f:
            emb_dict = pickle.load(f)
        print(f"Loaded {len(emb_dict)} embeddings")
    else:
        print(f"\nEmbedding {len(cached_ids)} videos with model '{args.model}' ...")
        emb_dict = embed_all(cached_ids, model_name=args.model,
                             cache_dir=args.pose_cache,
                             batch_size=args.batch_size,
                             io_workers=args.io_workers)
        print(f"Successfully embedded {len(emb_dict)} / {len(cached_ids)} videos")

        if args.save_embeddings:
            with open(args.save_embeddings, "wb") as f:
                pickle.dump(emb_dict, f)
            print(f"Embeddings saved to {args.save_embeddings}")

    # ------------------------------------------------------------------
    # Build video list
    # ------------------------------------------------------------------
    videos = []
    for vid_id, emb in emb_dict.items():
        meta = vid_to_meta.get(vid_id)
        if meta is None:
            continue
        videos.append({
            "video_id":  vid_id,
            "signer_id": meta["signer_id"],
            "gloss":     meta["gloss"],
            "emb":       emb,
        })

    # ------------------------------------------------------------------
    # Retrieval probe
    # ------------------------------------------------------------------
    print(f"\nRunning cross-gloss signer retrieval "
          f"(min_signer_videos={args.min_signer_videos}) ...")
    results = run_retrieval(videos, min_signer_videos=args.min_signer_videos)
    print(f"Evaluated {len(results)} queries")

    print_results(results, min_signer_videos=args.min_signer_videos)


if __name__ == "__main__":
    main()
