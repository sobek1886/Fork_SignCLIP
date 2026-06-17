#!/usr/bin/env python3
"""Extract I3D features from AUGMENTED ASL-Citizen FRAME DIRECTORIES.

The augmentations are stored as JPG frame dirs

    {aug_root}/{video_id}/{variant}/frame_*.jpg          variant ∈ AUG_NAMES

NOT as mp4, so the video-based extract_asl_citizen_i3d_features.py cannot read them.
This is the frame-dir counterpart (analogous to extract_logos_features_asl_citizen_aug.py
on the Logos side). It REUSES the identical I3D model load, preprocessing and clip
building from extract_asl_citizen_i3d_features.py, so the augmented features are drop-in
comparable with the originals already in i3d_features/asl_citizen_{id}.npy.

Output:
    {output_dir}/{dataset_name}_{video_id}_{variant}.npy     shape (T_clips, 1024)

The {id}_{variant} key matches SignCLIPVideoCSVMetaProcessor's f"{dataset}_{datum['id']}"
for the augmented rows in splits_aug (Video file = "{id}_{variant}.mp4"), so
asl_citizen_cnn_aug_i3d_ft.yaml (vfeat_dir=i3d_features) finds them automatically.

Frame-rate note (stride bug): signer_swap + skin_mst_diffusion frames were exported at
stride 2 (HALF the frame rate); glasses + shirt_1 at stride 1. I3D builds clips from
CONSECUTIVE frames, so a stride-2 sequence plays ~2x too fast and shifts the motion
statistics vs the full-rate originals. By default each stride-2-variant frame is therefore
duplicated x2 to restore the original frame rate before clipping (--no_rate_fix disables).

Usage (match how i3d_features was made — same weights, same num_classes default):
    python extract_asl_citizen_i3d_features_aug.py \\
        --aug_root    /scratch-shared/psobecki/ASL_Citizen/augmented_frames \\
        --output_dir  /home/psobecki/ASL_Citizen/i3d_features \\
        --i3d_weights /home/psobecki/bsl5k.pth.tar \\
        --workers 4 --batch_size 128
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

# Reuse the EXACT I3D recipe (constants + model load + GPU inference) from the
# video extractor so augmented features are bit-for-bit comparable with the originals.
from extract_asl_citizen_i3d_features import (
    CLIP_LEN, CLIP_STRIDE, FRAME_SIZE, FRAME_MEAN, FRAME_STD,
    load_i3d, gpu_inference,
)

AUG_NAMES = ("glasses", "shirt_1", "signer_swap", "skin_mst_diffusion")
STRIDE2_AUGS = ("signer_swap", "skin_mst_diffusion")  # exported at half frame rate


# ── Worker: load one augmented frame dir → I3D clips (CPU only, subprocess) ───
def _preprocess_dir_worker(item):
    """Returns (out_path, clips_np | None, status). clips_np: (N,16,3,224,224) float32."""
    frame_dir, out_path, variant, overwrite, rate_fix = item

    if not overwrite and os.path.exists(out_path):
        return out_path, None, 'skip'

    # Decode JPG/PNG frames in filename order
    try:
        import cv2
        paths = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')) +
                       glob.glob(os.path.join(frame_dir, '*.jpeg')) +
                       glob.glob(os.path.join(frame_dir, '*.png')))
        if not paths:
            return out_path, None, 'error: no frame images in dir'
        frames = []
        for p in paths:
            img = cv2.imread(p)                      # BGR uint8
            if img is None:
                continue
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not frames:
            return out_path, None, 'error: no decodable frames'
        # Restore original frame rate for stride-2 variants (hold each frame x2).
        if rate_fix and variant in STRIDE2_AUGS:
            frames = [f for f in frames for _ in range(2)]
        frames_np = np.stack(frames)                 # (T,H,W,3) uint8
    except Exception as e:
        return out_path, None, f'error: {e}'

    # Preprocess + build overlapping clips — IDENTICAL to extract_asl_citizen_i3d_features.
    try:
        import cv2
        T = len(frames_np)
        processed = np.empty((T, 3, FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
        for i, frame in enumerate(frames_np):
            h, w = frame.shape[:2]
            scale = FRAME_SIZE / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            top  = (new_h - FRAME_SIZE) // 2
            left = (new_w - FRAME_SIZE) // 2
            frame = frame[top:top + FRAME_SIZE, left:left + FRAME_SIZE]
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - FRAME_MEAN) / FRAME_STD
            processed[i] = frame.transpose(2, 0, 1)

        if T < CLIP_LEN:
            reps = (CLIP_LEN // T) + 1
            processed = np.concatenate([processed] * reps, axis=0)[:CLIP_LEN]
            clips = processed[None]
        else:
            clip_list = []
            start = 0
            while start + CLIP_LEN <= T:
                clip_list.append(processed[start:start + CLIP_LEN])
                start += CLIP_STRIDE
            remaining = T - (start - CLIP_STRIDE + CLIP_LEN)
            if remaining > CLIP_LEN // 2:
                clip_list.append(processed[T - CLIP_LEN:T])
            clips = np.stack(clip_list)
        return out_path, clips, 'ok'
    except Exception as e:
        return out_path, None, f'error: {e}'


def build_work_list(aug_root, output_dir, dataset_name, variants, overwrite, rate_fix):
    """One (frame_dir, out_path, variant, overwrite, rate_fix) per existing {id}/{variant}."""
    work = []
    for vid in sorted(os.listdir(aug_root)):
        vid_dir = os.path.join(aug_root, vid)
        if not os.path.isdir(vid_dir):
            continue
        for variant in variants:
            fdir = os.path.join(vid_dir, variant)
            if not os.path.isdir(fdir):
                continue
            out_path = os.path.join(output_dir, f'{dataset_name}_{vid}_{variant}.npy')
            work.append((fdir, out_path, variant, overwrite, rate_fix))
    return work


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--aug_root', required=True,
                        help='Root of augmented frame dirs: {aug_root}/{id}/{variant}/*.jpg')
    parser.add_argument('--output_dir', required=True,
                        help='Same dir as the originals (i3d_features) — adds *_{variant}.npy')
    parser.add_argument('--i3d_weights', required=True,
                        help='SAME checkpoint used for the originals (e.g. bsl5k.pth.tar)')
    parser.add_argument('--num_classes', type=int, default=2000,
                        help='Keep equal to the original extraction (head is unused; '
                             'strict=False load → only the backbone matters).')
    parser.add_argument('--dataset_name', default='asl_citizen')
    parser.add_argument('--variants', nargs='+', default=list(AUG_NAMES),
                        help=f'Which augmentation variants to extract. Default: {AUG_NAMES}')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=None,
                        help='Frame dirs to prefetch ahead (default: workers * 8).')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no_rate_fix', action='store_true',
                        help='Do NOT duplicate stride-2 variant frames; extract as-is '
                             '(clips will run ~2x fast for signer_swap / skin_mst_diffusion).')
    args = parser.parse_args()

    rate_fix = not args.no_rate_fix
    prefetch = args.prefetch or args.workers * 8
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'CPU workers: {args.workers}  |  GPU batch size: {args.batch_size}  |  Prefetch: {prefetch} dirs')
    print(f'Variants: {args.variants}  |  rate-fix (dup x2 for {STRIDE2_AUGS}): {rate_fix}')

    print(f'Loading I3D model ({args.num_classes} classes) ...')
    model = load_i3d(args.i3d_weights, device, args.num_classes)
    print('Model loaded.')

    work = build_work_list(args.aug_root, args.output_dir, args.dataset_name,
                           args.variants, args.overwrite, rate_fix)
    print(f'Found {len(work)} augmented frame dirs to extract')
    if not work:
        print(f'Nothing to do (no {args.variants} dirs under {args.aug_root}).')
        return

    skipped = errors = done = 0
    pending_clips = []   # list of (N_i,16,3,224,224)
    pending_paths = []   # list of (out_path, N_i)

    def flush_gpu():
        nonlocal done
        if not pending_clips:
            return
        all_clips = np.concatenate(pending_clips, axis=0)
        all_feats = gpu_inference(model, all_clips, device, args.batch_size)
        offset = 0
        for out_path, n in pending_paths:
            np.save(out_path, all_feats[offset:offset + n])
            offset += n
            done += 1
        pending_clips.clear()
        pending_paths.clear()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pbar = tqdm(total=len(work), desc='Extracting(aug)')
        for chunk_start in range(0, len(work), prefetch):
            chunk = work[chunk_start:chunk_start + prefetch]
            futures = {executor.submit(_preprocess_dir_worker, item): item for item in chunk}
            for future in as_completed(futures):
                out_path, clips, status = future.result()
                if status == 'skip':
                    skipped += 1
                elif status == 'ok':
                    pending_clips.append(clips)
                    pending_paths.append((out_path, len(clips)))
                    if sum(c.shape[0] for c in pending_clips) >= args.batch_size:
                        flush_gpu()
                else:
                    tqdm.write(f'  WARN {futures[future][0]}: {status}')
                    errors += 1
                pbar.update(1)
        pbar.close()

    flush_gpu()
    print(f'\nExtracted: {done}  |  Skipped: {skipped}  |  Errors: {errors}')
    print(f'Output:  {args.output_dir}/{args.dataset_name}_<video_id>_<variant>.npy')


if __name__ == '__main__':
    main()
