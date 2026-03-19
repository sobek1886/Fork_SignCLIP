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
        --batch_size  16

Install prerequisites:
    pip install decord opencv-python-headless
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ── I3D constants ─────────────────────────────────────────────────────────────
CLIP_LEN    = 16    # frames per I3D input clip
CLIP_STRIDE = 8     # temporal stride; overlap = CLIP_LEN - CLIP_STRIDE
FRAME_SIZE  = 224   # spatial crop after resize
# BSL-5K preprocessing: normalise to [-1, 1]
FRAME_MEAN = 0.5
FRAME_STD  = 0.5


# ── Model loading ─────────────────────────────────────────────────────────────

def load_i3d(weights_path, device):
    # Import from our self-contained implementation
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mmpt.processors.models.i3d import InceptionI3d  # noqa: PLC0415

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = state_dict.get("state_dict", state_dict)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = InceptionI3d(num_classes=5383, in_channels=3)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Weight mismatch — missing: {missing[:3]}, unexpected: {unexpected[:3]}"
        )
    model = model.to(device).eval()
    return model


# ── Video loading ──────────────────────────────────────────────────────────────

def load_frames(video_path):
    """Decode all frames.  Returns (T, H, W, 3) uint8 or None on failure."""
    try:
        from decord import VideoReader, cpu  # noqa: PLC0415
        vr = VideoReader(video_path, ctx=cpu(0))
        return vr.get_batch(range(len(vr))).asnumpy()
    except Exception:
        pass
    try:
        import cv2  # noqa: PLC0415
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.stack(frames) if frames else None
    except Exception as e:
        print(f"  Could not decode {video_path}: {e}")
        return None


# ── Preprocessing ─────────────────────────────────────────────────────────────

def resize_crop(frame):
    import cv2  # noqa: PLC0415
    h, w = frame.shape[:2]
    scale = FRAME_SIZE / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top  = (new_h - FRAME_SIZE) // 2
    left = (new_w - FRAME_SIZE) // 2
    return frame[top:top + FRAME_SIZE, left:left + FRAME_SIZE]


def preprocess(frames):
    """(T,H,W,3) uint8  →  (T,3,H,W) float32 in ~[-1,1]"""
    frames = np.stack([resize_crop(f) for f in frames])   # (T,224,224,3)
    frames = frames.astype(np.float32) / 255.0
    frames = (frames - FRAME_MEAN) / FRAME_STD
    return frames.transpose(0, 3, 1, 2)                   # (T,3,224,224)


def build_clips(frames):
    """(T,3,H,W)  →  (N,16,3,H,W)  overlapping clips"""
    T = len(frames)
    if T < CLIP_LEN:
        # Loop-pad to one full clip
        reps = (CLIP_LEN // T) + 1
        frames = np.concatenate([frames] * reps, axis=0)[:CLIP_LEN]
        return frames[None]                                # (1,16,3,H,W)

    clips = []
    start = 0
    while start + CLIP_LEN <= T:
        clips.append(frames[start:start + CLIP_LEN])
        start += CLIP_STRIDE
    # Include a tail clip if the last window was cut short by more than half
    remaining = T - (start - CLIP_STRIDE + CLIP_LEN)
    if remaining > CLIP_LEN // 2:
        clips.append(frames[T - CLIP_LEN:T])

    return np.stack(clips)                                 # (N,16,3,H,W)


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def get_features(model, clips_np, device, batch_size):
    """clips_np: (N,16,3,H,W)  →  features: (N,1024)"""
    # I3D expects (B, C, T, H, W)
    clips = torch.from_numpy(clips_np).permute(0, 2, 1, 3, 4).float().to(device)
    parts = []
    for i in range(0, len(clips), batch_size):
        batch = clips[i:i + batch_size]
        out = model.extract_features(batch)                # (B,1024,T',H',W')
        out = F.adaptive_avg_pool3d(out, 1).flatten(1)     # (B,1024)
        parts.append(out.cpu().numpy())
    return np.concatenate(parts, axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir",    required=True,
                        help="Directory containing ASL-Citizen .mp4 files")
    parser.add_argument("--output_dir",   required=True,
                        help="Where to save .npy feature files")
    parser.add_argument("--i3d_weights",  required=True,
                        help="Path to bsl5k.pth.tar checkpoint")
    parser.add_argument("--dataset_name", default="asl_citizen",
                        help="Prefix for output filenames (default: asl_citizen)")
    parser.add_argument("--batch_size",   type=int, default=16,
                        help="Clips per GPU batch")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Re-extract even if .npy already exists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading BSL-5K I3D model ...")
    model = load_i3d(args.i3d_weights, device)
    print("Model loaded (0 missing / unexpected keys)")

    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
    video_files = []
    for ext in exts:
        video_files.extend(glob.glob(os.path.join(args.video_dir, "**", ext), recursive=True))
    video_files = sorted(video_files)
    print(f"Found {len(video_files)} video files")

    skipped = errors = 0
    for vpath in tqdm(video_files, desc="Extracting"):
        vid = os.path.splitext(os.path.basename(vpath))[0]
        out_path = os.path.join(args.output_dir, f"{args.dataset_name}_{vid}.npy")

        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        frames = load_frames(vpath)
        if frames is None or len(frames) == 0:
            tqdm.write(f"  WARN: could not decode {vpath}")
            errors += 1
            continue

        try:
            frames_p = preprocess(frames)              # (T,3,224,224)
            clips    = build_clips(frames_p)           # (N,16,3,224,224)
            feats    = get_features(model, clips, device, args.batch_size)  # (N,1024)
            np.save(out_path, feats)
        except Exception as e:
            tqdm.write(f"  ERROR {vpath}: {e}")
            errors += 1

    done = len(video_files) - skipped - errors
    print(f"\nExtracted: {done}  |  Skipped: {skipped}  |  Errors: {errors}")
    print(f"Output:  {args.output_dir}/{args.dataset_name}_<video_id>.npy")
    print(f"Shape:   (T_clips, 1024)")


if __name__ == "__main__":
    main()
