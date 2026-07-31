"""One-shot dump: 10 example frames per gloss, sampled from later in the
clip (skipping an initial buffer to avoid the pre-sign pause many
ASL-Citizen clips start with), for the 18 glosses in the 6
diversity-maximized ASL-LEX phonological families (fig:tsne_grid redesign
audit). Not for the thesis -- Piotr's own sanity check that the 6
families actually look visually similar within a family / distinct
across families, this time with the actual sign motion visible and one
distinct signer per gloss (was previously mostly one participant, P40,
because "first row per gloss" happened to always land on them).

Reuses the cv2.VideoCapture / BGR->RGB pattern from
extract_asl_citizen_i3d_features.py. Run on the cluster (videos are
cluster-only, no local copies): fast (18 short clips), safe to run in the
foreground, no nohup/background needed.

    python3 dump_frame_examples.py
    -> ~/ASL_Citizen/frame_examples/<gloss>_f<i>.jpg  (18 x 10 = 180 files)
"""
import os
import cv2
import numpy as np

VIDEO_DIR = "/scratch-shared/psobecki/ASL_Citizen/videos"
OUT_DIR = os.path.expanduser("~/ASL_Citizen/frame_examples")
N_FRAMES = 10
START_FRAC = 0.20   # skip the first 20% (presumed pre-sign pause)
END_FRAC = 0.90      # avoid the last 10% (presumed trailing hold)
MIN_SPAN_FRAMES = 15  # below this, fall back to spreading across the whole clip
THUMB_WIDTH = 200

# gloss -> Video file, resolved from context/ASL_citizen_context/splits/train.csv.
# Randomised per-gloss pick (seed 0, greedy anti-repeat on Participant ID) so
# the 18 examples are NOT dominated by one signer, as the first-row pick was
# (P40 for 14/18 glosses). 18/18 distinct participants achieved.
GLOSS2VIDEO = {
    "REMOTECONTROL": "8366079828801334-REMOTE CONTROL.mp4",
    "GUN1": "6247620625980512-GUN.mp4",
    "LIGHTER": "8864496316656505-LIGHTER.mp4",
    "MOP1": "5955612822201903-MOP.mp4",
    "POPCORN": "5409556868252285-POPCORN.mp4",
    "PULL": "03979078081951393-PULL.mp4",
    "SMALL": "5446958870984302-SMALL.mp4",
    "CONGRATULATIONS": "9958532597521785-CONGRATULATIONS.mp4",
    "HAMBURGER": "31702578270971826-HAMBURGER.mp4",
    "KANGAROO3": "2715140197673327-KANGAROO.mp4",
    "PARADE": "7636350993107011-PARADE.mp4",
    "BALANCE": "985894713504524-BALANCE.mp4",
    "BABY1": "8774565600191819-BABY.mp4",
    "PRAYER": "6578601859216238-seedPRAYER.mp4",
    "ROOM": "9849386468173638-ROOM.mp4",
    "CAR": "4642218872328543-CAR.mp4",
    "CANOE": "16159747075537356-CANOE.mp4",
    "SHOVEL1": "1355005799699034-SHOVEL.mp4",
}


def sample_frames(vpath):
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        cap.release()
        return None, False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    span = (END_FRAC - START_FRAC) * total
    fallback = span < MIN_SPAN_FRAMES
    if fallback:
        idx = np.linspace(0, max(total - 1, 0), N_FRAMES).astype(int).tolist()
    else:
        start, end = START_FRAC * total, END_FRAC * total
        idx = np.linspace(start, end, N_FRAMES).astype(int).tolist()
    frames, i, wanted = [], 0, set(idx)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
        if len(frames) == len(idx):
            break
    cap.release()
    return frames, fallback


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    missing, fallback_used, failed = [], [], []
    for gloss, fname in GLOSS2VIDEO.items():
        vpath = os.path.join(VIDEO_DIR, fname)
        if not os.path.isfile(vpath):
            missing.append(gloss)
            continue
        frames, fallback = sample_frames(vpath)
        if frames is None or len(frames) == 0:
            failed.append(gloss)
            continue
        if fallback:
            fallback_used.append(gloss)
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            new_h = int(h * THUMB_WIDTH / w)
            thumb = cv2.resize(frame, (THUMB_WIDTH, new_h))
            out_path = os.path.join(OUT_DIR, f"{gloss}_f{i}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
        print(f"{gloss}: wrote {len(frames)} frames"
              f"{'  (SHORT CLIP, whole-clip fallback)' if fallback else ''}")

    print(f"\n{len(GLOSS2VIDEO) - len(missing) - len(failed)}/{len(GLOSS2VIDEO)} glosses OK")
    if missing:
        print("MISSING video file:", missing)
    if failed:
        print("FAILED to open/decode:", failed)
    if fallback_used:
        print("whole-clip fallback used for:", fallback_used)
    print("wrote frames to", OUT_DIR)


if __name__ == "__main__":
    main()
