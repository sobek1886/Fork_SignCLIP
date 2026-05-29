"""Analyse ASL-Citizen training data and test predictions to select which
training videos to augment for appearance-invariance training.

Selection priority (sorted ascending, so worst-first):
  1. n_train_signers  — 1-signer classes benefit most from augmentation
  2. test_accuracy    — hardest classes first within same signer count

Outputs:
  aug_priority.csv   – ranked table: gloss | train_count | train_signers | test_accuracy
  aug_video_list.txt – video_ids (without .mp4) to augment, capped at --budget

Video filename convention expected in train.csv:
  {signer_id}-{GLOSS}.mp4   e.g.  7410564482917914-CRACK.mp4
  Signer ID = everything before the first '-'.

Usage (on Snellius, login node is fine — no GPU needed):
  python analyze_asl_citizen_for_aug.py \\
      --train_csv  /home/psobecki/ASL_Citizen/splits/train.csv \\
      --test_csv   /home/psobecki/ASL_Citizen/splits/test.csv \\
      --eval_dir   runs/retri_asl/asl_citizen_cnn_scratch_logos_nowd/eval/test \\
      --budget     5400 \\
      --output_dir /home/psobecki/ASL_Citizen/aug_selection
"""

import argparse
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            video_file = row['Video file']
            gloss = row['Gloss']
            video_id = os.path.splitext(video_file)[0]
            # Participant ID column (P1–P52) is the actual signer identity.
            # Falls back to the numeric video-UID prefix if the column is absent
            # (older Snellius CSVs that were stripped to Video file + Gloss only).
            signer_id = row.get('Participant ID') or video_id.split('-')[0]
            rows.append({'video_id': video_id, 'gloss': gloss, 'signer_id': signer_id})
    return rows


def load_embeddings(eval_dir):
    files = sorted(
        glob.glob(os.path.join(eval_dir, 'video_embeddings_*.npy')),
        key=lambda p: int(Path(p).stem.split('_')[-1]),
    )
    if not files:
        raise FileNotFoundError(f'No video_embeddings_*.npy found in {eval_dir}')
    parts = [np.load(f) for f in files]
    embeddings = np.concatenate(parts, axis=0)
    print(f'Loaded {len(files)} batch files → {embeddings.shape} embeddings')
    return embeddings


# ── Per-class accuracy via nearest-centroid ───────────────────────────────────

def compute_per_class_accuracy(embeddings, feat_ids, test_rows):
    """Nearest-centroid classifier in video-embedding space.

    For each test video: find which class centroid it is closest to.
    Returns {gloss: accuracy}.

    This is a proxy for V2T accuracy — it measures how well video embeddings
    cluster by class, which degrades when models overfit to signer identity.
    """
    # feat_id → gloss  (feat_id looks like "asl_citizen_7410...-CRACK")
    id_to_gloss = {}
    for row in test_rows:
        feat_id = f"asl_citizen_{row['video_id']}"
        id_to_gloss[feat_id] = row['gloss']

    # Build class centroids from all test embeddings for that class
    class_sum = defaultdict(lambda: np.zeros(embeddings.shape[1], dtype=np.float64))
    class_count = defaultdict(int)
    for i, fid in enumerate(feat_ids):
        gloss = id_to_gloss.get(fid)
        if gloss is None:
            continue
        class_sum[gloss] += embeddings[i]
        class_count[gloss] += 1

    classes = sorted(class_sum.keys())
    centroids = np.stack([class_sum[g] / class_count[g] for g in classes]).astype(np.float32)

    # L2-normalise
    norms = np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-8)
    centroids = centroids / norms
    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)

    # Similarity: (N, C)
    sims = emb_norm @ centroids.T

    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for i, fid in enumerate(feat_ids):
        true_gloss = id_to_gloss.get(fid)
        if true_gloss is None:
            continue
        pred_gloss = classes[int(np.argmax(sims[i]))]
        class_total[true_gloss] += 1
        if pred_gloss == true_gloss:
            class_correct[true_gloss] += 1

    return {g: class_correct[g] / class_total[g] for g in class_total}


# ── Per-signer accuracy ───────────────────────────────────────────────────────

def compute_per_signer_accuracy(embeddings, feat_ids, test_rows, class_to_centroid_idx, classes):
    """Returns {signer_id: (n_correct, n_total)}."""
    id_to_row = {f"asl_citizen_{r['video_id']}": r for r in test_rows}
    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)

    # Reconstruct centroids (already computed in compute_per_class_accuracy, but cheaper to redo)
    class_sum = defaultdict(lambda: np.zeros(embeddings.shape[1], dtype=np.float64))
    class_count = defaultdict(int)
    for i, fid in enumerate(feat_ids):
        row = id_to_row.get(fid)
        if row:
            class_sum[row['gloss']] += embeddings[i]
            class_count[row['gloss']] += 1
    centroids = np.stack([class_sum[g] / class_count[g] for g in classes]).astype(np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-8)
    centroids = centroids / norms

    sims = emb_norm @ centroids.T

    signer_correct = defaultdict(int)
    signer_total = defaultdict(int)
    for i, fid in enumerate(feat_ids):
        row = id_to_row.get(fid)
        if not row:
            continue
        pred_gloss = classes[int(np.argmax(sims[i]))]
        signer_total[row['signer_id']] += 1
        if pred_gloss == row['gloss']:
            signer_correct[row['signer_id']] += 1

    return {s: (signer_correct[s], signer_total[s]) for s in signer_total}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--test_csv', required=True)
    parser.add_argument('--eval_dir', required=True,
                        help='Directory containing video_embeddings_*.npy, ids.txt, texts.txt')
    parser.add_argument('--budget', type=int, default=5400,
                        help='Max training videos to select for augmentation (per aug type)')
    parser.add_argument('--output_dir', default='.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load splits ──────────────────────────────────────────────────────────
    train_rows = load_split(args.train_csv)
    test_rows = load_split(args.test_csv)
    print(f'Train: {len(train_rows)} videos | Test: {len(test_rows)} videos')

    # Training stats per class
    train_by_class = defaultdict(list)
    for row in train_rows:
        train_by_class[row['gloss']].append(row)

    class_stats = {}
    for gloss, rows in train_by_class.items():
        signers = sorted({r['signer_id'] for r in rows})
        class_stats[gloss] = {
            'train_count': len(rows),
            'train_signers': len(signers),
            'train_videos': [r['video_id'] for r in rows],
        }

    n_single = sum(1 for s in class_stats.values() if s['train_signers'] == 1)
    print(f'Train classes: {len(class_stats)} | 1-signer classes: {n_single}')

    # ── Load eval embeddings ─────────────────────────────────────────────────
    embeddings = load_embeddings(args.eval_dir)

    ids_path = os.path.join(args.eval_dir, 'ids.txt')
    with open(ids_path) as f:
        feat_ids = [line.strip() for line in f if line.strip()]

    if len(feat_ids) != len(embeddings):
        raise ValueError(
            f'ids.txt has {len(feat_ids)} entries but embeddings have {len(embeddings)} rows'
        )

    # ── Per-class accuracy (nearest-centroid proxy) ──────────────────────────
    print('Computing per-class nearest-centroid accuracy ...')
    per_class_acc = compute_per_class_accuracy(embeddings, feat_ids, test_rows)

    classes = sorted(per_class_acc.keys())
    per_signer = compute_per_signer_accuracy(embeddings, feat_ids, test_rows, None, classes)

    # ── Build priority table ─────────────────────────────────────────────────
    all_glosses = set(class_stats.keys()) | set(per_class_acc.keys())
    rows_out = []
    for gloss in all_glosses:
        stats = class_stats.get(gloss, {'train_count': 0, 'train_signers': 999, 'train_videos': []})
        acc = per_class_acc.get(gloss, float('nan'))
        rows_out.append({
            'gloss': gloss,
            'train_count': stats['train_count'],
            'train_signers': stats['train_signers'],
            'test_accuracy': round(acc, 4) if not np.isnan(acc) else -1.0,
            '_videos': stats['train_videos'],
        })

    # Sort: fewest signers first, then worst accuracy first
    rows_out.sort(key=lambda r: (r['train_signers'], r['test_accuracy']))

    # ── Save priority CSV ────────────────────────────────────────────────────
    priority_csv = os.path.join(args.output_dir, 'aug_priority.csv')
    with open(priority_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['gloss', 'train_count', 'train_signers', 'test_accuracy']
        )
        writer.writeheader()
        for row in rows_out:
            writer.writerow({k: v for k, v in row.items() if k != '_videos'})
    print(f'Wrote {priority_csv}')

    # ── Per-signer summary ───────────────────────────────────────────────────
    signer_csv = os.path.join(args.output_dir, 'aug_signer_accuracy.csv')
    with open(signer_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['signer_id', 'correct', 'total', 'accuracy'])
        writer.writeheader()
        for sid, (correct, total) in sorted(per_signer.items(), key=lambda x: x[1][0] / x[1][1]):
            writer.writerow({
                'signer_id': sid,
                'correct': correct,
                'total': total,
                'accuracy': round(correct / total, 4),
            })
    print(f'Wrote {signer_csv}')

    # ── Select videos up to budget ───────────────────────────────────────────
    selected = []
    classes_selected = 0
    for row in rows_out:
        if row['train_count'] == 0:
            continue
        remaining = args.budget - len(selected)
        if remaining <= 0:
            break
        videos = row['_videos'][:remaining]
        selected.extend(videos)
        classes_selected += 1

    video_list_path = os.path.join(args.output_dir, 'aug_video_list.txt')
    with open(video_list_path, 'w') as f:
        for vid in selected:
            f.write(vid + '\n')
    print(f'Selected {len(selected)} videos from {classes_selected} classes → {video_list_path}')

    # ── Summary stats ────────────────────────────────────────────────────────
    selected_set = set(selected)
    selected_single = sum(
        1 for gloss, s in class_stats.items()
        if s['train_signers'] == 1 and any(v in selected_set for v in s['train_videos'])
    )
    print(f'\n=== Summary ===')
    print(f'  Total train classes   : {len(class_stats)}')
    print(f'  1-signer classes      : {n_single}')
    print(f'  Budget                : {args.budget} videos')
    print(f'  Selected              : {len(selected)} videos from {classes_selected} classes')
    print(f'  Of which 1-signer cls : {selected_single}')
    print(f'  Overall test accuracy : {np.mean(list(per_class_acc.values())):.4f}')
    print(f'')
    print(f'Next steps:')
    print(f'  1. rsync ASL_Citizen videos for selected IDs to local Mac')
    print(f'  2. Run nano_banana augmentation (glasses + shirt_1) in parallel')
    print(f'  3. rsync augmented videos back to Snellius')
    print(f'  4. Run extract_logos_features.py on augmented videos')
    print(f'  5. Run generate_asl_citizen_aug_splits.py to build splits_aug/')
    print(f'  6. Submit training job: asl_citizen_cnn_aug_logos.job')


if __name__ == '__main__':
    main()
