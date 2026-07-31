"""Dump per-gloss centroids + sampled video features, for the NEW
iconicity-family probe set (fig:tsne_grid redesign, Option 4: the original
58-gloss probe set turned up 0 strict 5/5-ASL-LEX-match triplets, so this
uses 6 real triplets found via a full-corpus search instead).

Mirrors dump_kq_spaces.py exactly; only GLOSSES is replaced.

Usage:
    python dump_kq_spaces_families.py \
        --space raw=/scratch-shared/psobecki/ASL_Citizen/logos_features_native \
        --space ce=/scratch-shared/psobecki/ASL_Citizen/logos_features_run2 \
        ... (9 spaces total)
"""
import os, argparse, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from eval_asl_citizen_retrieval import load_metadata, load_features

# 6 triplets, exact 5/5 ASL-LEX PHONO_COLS match, top tertile of D.Iconicity(M)
# among ASL-Citizen train glosses with >=10 clips (full-corpus search,
# 2026-07-30). Selected to MAXIMIZE cross-family diversity: MajorLocation/
# MinorLocation are Neutral for every eligible candidate (a hard property of
# this corpus, not a choice), so diversity is maximized on the remaining 3
# features -- 6 distinct handshapes (the max of 8 available, given the
# 6-family cap), all 3 Movement values represented (None/Straight/Curved),
# both SignType values represented (OneHanded/SymmetricalOrAlternating).
# See family_search_full.py for provenance.
FAMILIES = {
    "fam_a_remote_gun_lighter": ["REMOTECONTROL", "GUN1", "LIGHTER"],
    "fam_babyo_mop_pull_popcorn": ["MOP1", "POPCORN", "PULL"],
    "fam_c_small_congrats_hamburger": ["SMALL", "CONGRATULATIONS", "HAMBURGER"],
    "fam_flatb_kangaroo_parade_balance": ["KANGAROO3", "PARADE", "BALANCE"],
    "fam_openb_prayer_baby_room": ["PRAYER", "BABY1", "ROOM"],
    "fam_s_car_canoe_shovel": ["CAR", "CANOE", "SHOVEL1"],
}
GLOSSES = sorted(set(g for members in FAMILIES.values() for g in members))

ap = argparse.ArgumentParser()
ap.add_argument("--space", action="append", required=True,
                help="name=dir; repeatable")
ap.add_argument("--splits_dir", default="/home/psobecki/ASL_Citizen/splits")
ap.add_argument("--out", default="runs/king_queen_analogy/npz_families")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
recs = load_metadata(Path(args.splits_dir), "train")

for spec in args.space:
    name, d = spec.split("=", 1)
    feats, glosses = load_features(recs, Path(d), None, desc=name, workers=32,
                                   min_coverage=0, allow_partial=True)
    feats = F.normalize(torch.from_numpy(feats).float(), dim=-1).numpy()
    glosses = np.array(glosses)
    present = [g for g in GLOSSES if (glosses == g).any()]
    missing = [g for g in GLOSSES if g not in present]
    if missing:
        print(f"[{name}] WARNING missing glosses: {missing}")
    cent = np.stack([feats[glosses == g].mean(0) for g in present]).astype(np.float32)
    rng = np.random.default_rng(0)
    X, L = [], []
    for g in present:
        idx = np.where(glosses == g)[0]
        take = rng.choice(idx, size=min(14, len(idx)), replace=False)
        X.append(feats[take]); L += [g] * len(take)
    np.savez(f"{args.out}/kqfeat_fam_{name}.npz",
             glosses=np.array(present),
             centroids=cent,
             samp_feats=np.concatenate(X).astype(np.float32),
             samp_labels=np.array(L))
    print(name, "saved", len(present), "glosses,",
          np.concatenate(X).shape[0], "sampled videos")
print("DONE_DUMP_SPACES_FAMILIES")
