"""Extract I3D BSL-5K features from ASL-Citizen video files.

Uses the CVPR'21 M+D+A 136MB model (bsl5k.pth.tar) with the self-contained
InceptionI3d implementation in mmpt/processors/models/i3d.py — no external
repositories required.

Output:
    One .npy file per video at {output_dir}/{dataset_name}_{video_id}.npy
    Shape: (T_clips, 1024) — one 1024-dim feature per 16-frame sliding-window clip.

The {dataset_name}_{video_id} key must match the ID format used by
SignCLIPVideoMetaProcessor: f"{dataset}_{datum['id']}" from tensorflow_datasets.
For ASL-Citizen the video file basename (without extension) should equal
datum['id'] from the asl_citizen tfds split.

Usage:
    python extract_asl_citizen_i3d_features.py \\
        --video_dir   /path/to/asl_citizen/videos \\
        --output_dir  /path/to/i3d_features \\
        --i3d_weights /path/to/bsl5k.pth.tar \\
        --dataset_name asl_citizen \\
        --workers 4 \\
        --batch_size 128

Install prerequisites:
    pip install opencv-python-headless  (decord optional but faster)
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ── I3D constants (must be module-level for worker processes) ─────────────────
CLIP_LEN   = 16    # frames per I3D input clip
CLIP_STRIDE = 8    # temporal stride; overlap = CLIP_LEN - CLIP_STRIDE
FRAME_SIZE  = 224  # spatial crop after resize
FRAME_MEAN  = 0.5  # BSL-5K normalisation: (pixel/255 - 0.5) / 0.5  →  [-1, 1]
FRAME_STD   = 0.5


# ── Worker function (module-level required for multiprocessing) ───────────────

def _preprocess_worker(args):
    """Decode one video and build I3D clips. Runs in a subprocess (CPU only).

    Uses OpenCV exclusively — decord is not used here because it can be
    unreliable across fork/spawn boundaries on Linux.

    Returns: (out_path, clips_np, status)
        clips_np: (N, 16, 3, 224, 224) float32, or None on skip/error
        status:   'ok' | 'skip' | 'error: <msg>'
    """
    vpath, out_path, overwrite = args

    if not overwrite and os.path.exists(out_path):
        return out_path, None, 'skip'

    # Decode
    try:
        import cv2
        cap = cv2.VideoCapture(vpath)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return out_path, None, 'error: no frames decoded'
        frames_np = np.stack(frames)   # (T, H, W, 3) uint8
    except Exception as e:
        return out_path, None, f'error: {e}'

    # Preprocess: resize-crop + normalise + build overlapping clips
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
            processed[i] = frame.transpose(2, 0, 1)   # (3, H, W)

        # Build overlapping clips
        if T < CLIP_LEN:
            reps = (CLIP_LEN // T) + 1
            processed = np.concatenate([processed] * reps, axis=0)[:CLIP_LEN]
            clips = processed[None]                    # (1, 16, 3, H, W)
        else:
            clip_list = []
            start = 0
            while start + CLIP_LEN <= T:
                clip_list.append(processed[start:start + CLIP_LEN])
                start += CLIP_STRIDE
            remaining = T - (start - CLIP_STRIDE + CLIP_LEN)
            if remaining > CLIP_LEN // 2:
                clip_list.append(processed[T - CLIP_LEN:T])
            clips = np.stack(clip_list)                # (N, 16, 3, H, W)

        return out_path, clips, 'ok'
    except Exception as e:
        return out_path, None, f'error: {e}'


# ── Model loading ─────────────────────────────────────────────────────────────

def load_i3d(weights_path, device):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mmpt.processors.models.i3d import InceptionI3d  # noqa: PLC0415

    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    state_dict = state_dict.get('state_dict', state_dict)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model = InceptionI3d(num_classes=5383, in_channels=3)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f'Weight mismatch — missing: {missing[:3]}, unexpected: {unexpected[:3]}'
        )
    return model.to(device).eval()


# ── GPU inference ─────────────────────────────────────────────────────────────

@torch.no_grad()
def gpu_inference(model, clips_np, device, batch_size):
    """clips_np: (N, 16, 3, H, W)  →  (N, 1024) numpy, processed in sub-batches."""
    # I3D expects (B, C, T, H, W)
    clips = torch.from_numpy(clips_np).permute(0, 2, 1, 3, 4).float().to(device)
    parts = []
    for i in range(0, len(clips), batch_size):
        out = model.extract_features(clips[i:i + batch_size])  # (B, 1024, T', H', W')
        out = F.adaptive_avg_pool3d(out, 1).flatten(1)          # (B, 1024)
        parts.append(out.cpu().numpy())
    return np.concatenate(parts, axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir',    required=True)
    parser.add_argument('--output_dir',   required=True)
    parser.add_argument('--i3d_weights',  required=True,
                        help='Path to bsl5k.pth.tar')
    parser.add_argument('--dataset_name', default='asl_citizen')
    parser.add_argument('--workers',      type=int, default=4,
                        help='CPU worker processes for parallel video decoding. '
                             'Set to match --cpus-per-task in your SLURM script.')
    parser.add_argument('--batch_size',   type=int, default=128,
                        help='Total clips per GPU forward pass (across multiple videos). '
                             'Higher = better GPU utilisation. Lower if OOM.')
    parser.add_argument('--prefetch',     type=int, default=None,
                        help='Videos to prefetch ahead (default: workers * 8). '
                             'Bounds CPU RAM: each video ~80 MB of clips in memory.')
    parser.add_argument('--overwrite',    action='store_true')
    args = parser.parse_args()

    prefetch = args.prefetch or args.workers * 8

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'CPU workers: {args.workers}  |  GPU batch size: {args.batch_size}  |  Prefetch: {prefetch} videos')

    print('Loading BSL-5K I3D model ...')
    model = load_i3d(args.i3d_weights, device)
    print('Model loaded (0 missing / unexpected keys)')

    exts = ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm')
    video_files = []
    for ext in exts:
        video_files.extend(glob.glob(os.path.join(args.video_dir, '**', ext), recursive=True))
    video_files = sorted(video_files)
    print(f'Found {len(video_files)} video files')

    # Build work list
    work = []
    for vpath in video_files:
        vid = os.path.splitext(os.path.basename(vpath))[0]
        out_path = os.path.join(args.output_dir, f'{args.dataset_name}_{vid}.npy')
        work.append((vpath, out_path, args.overwrite))

    skipped = errors = done = 0

    # Pending GPU buffer: accumulate clips from multiple videos before inference
    # so the GPU sees large batches rather than 5-10 clips per video.
    pending_clips  = []   # list of (N_i, 16, 3, 224, 224) arrays
    pending_paths  = []   # list of (out_path, N_i) — needed to demux results

    def flush_gpu():
        """Run inference on the accumulated clip buffer and save results."""
        nonlocal done
        if not pending_clips:
            return
        all_clips = np.concatenate(pending_clips, axis=0)   # (sum_N, 16, 3, 224, 224)
        all_feats = gpu_inference(model, all_clips, device, args.batch_size)
        offset = 0
        for out_path, n in pending_paths:
            np.save(out_path, all_feats[offset:offset + n])
            offset += n
            done += 1
        pending_clips.clear()
        pending_paths.clear()

    # Process in sliding windows of `prefetch` videos to bound RAM usage
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pbar = tqdm(total=len(work), desc='Extracting')

        for chunk_start in range(0, len(work), prefetch):
            chunk = work[chunk_start:chunk_start + prefetch]
            futures = {executor.submit(_preprocess_worker, item): item for item in chunk}

            for future in as_completed(futures):
                out_path, clips, status = future.result()

                if status == 'skip':
                    skipped += 1
                elif status == 'ok':
                    pending_clips.append(clips)
                    pending_paths.append((out_path, len(clips)))
                    # Flush to GPU when buffer is large enough
                    total_pending = sum(c.shape[0] for c in pending_clips)
                    if total_pending >= args.batch_size:
                        flush_gpu()
                else:
                    tqdm.write(f'  WARN {futures[future][0]}: {status}')
                    errors += 1

                pbar.update(1)

        pbar.close()

    # Final flush for any remaining videos
    flush_gpu()

    print(f'\nExtracted: {done}  |  Skipped: {skipped}  |  Errors: {errors}')
    print(f'Output:  {args.output_dir}/{args.dataset_name}_<video_id>.npy')
    print(f'Shape:   (T_clips, 1024)')


if __name__ == '__main__':
    main()
