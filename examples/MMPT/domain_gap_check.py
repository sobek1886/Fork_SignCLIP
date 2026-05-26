"""Quick domain-gap sanity check for the NGT pair manifest.

For each matched pair: load both .npy files, mean-pool over clips → (768,),
compute cosine similarity.  Prints summary statistics and a small histogram.

Usage:
    python domain_gap_check.py --manifest /home/psobecki/ngt_pair_manifest.json
"""

import argparse
import json
import numpy as np
from pathlib import Path


def load_mean(path: str) -> np.ndarray:
    """Load .npy (N_clips, 768), return mean-pooled (768,) L2-normalised vector."""
    feat = np.load(path).astype(np.float32)   # (N, 768)
    v = feat.mean(axis=0)                      # (768,)
    v /= np.linalg.norm(v) + 1e-8
    return v


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def histogram(values, bins=10, width=40):
    counts, edges = np.histogram(values, bins=bins)
    for i, c in enumerate(counts):
        bar = '#' * int(c / max(counts) * width)
        print(f'  [{edges[i]:.2f}, {edges[i+1]:.2f})  {bar} {c}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True,
                        help='Path to ngt_pair_manifest.json')
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Limit to first N pairs (default: all)')
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    keys = sorted(manifest.keys())
    if args.max_pairs:
        keys = keys[:args.max_pairs]

    print(f'Checking {len(keys)} pairs ...')
    sims = []
    errors = []
    for key in keys:
        try:
            real   = load_mean(manifest[key]['real'])
            unreal = load_mean(manifest[key]['unreal'])
            sims.append(cosine_sim(real, unreal))
        except Exception as e:
            errors.append((key, str(e)))

    if errors:
        print(f'\nERRORS ({len(errors)}):')
        for k, e in errors[:5]:
            print(f'  {k}: {e}')

    sims = np.array(sims)
    print(f'\nCosine similarity  real ↔ unreal (N={len(sims)})')
    print(f'  mean:   {sims.mean():.4f}')
    print(f'  median: {np.median(sims):.4f}')
    print(f'  std:    {sims.std():.4f}')
    print(f'  min:    {sims.min():.4f}')
    print(f'  max:    {sims.max():.4f}')
    print(f'  frac > 0.15:  {(sims > 0.15).mean():.1%}')
    print(f'  frac > 0.30:  {(sims > 0.30).mean():.1%}')
    print()
    print('Distribution:')
    histogram(sims)


if __name__ == '__main__':
    main()
