"""Dump per-gloss centroids + sampled video features for the king-queen /
form-neighbour / iconicity glosses, for ARBITRARY SignCLIP feature spaces given
on the command line, so those figures can be re-plotted locally.

Usage:
    python dump_kq_spaces.py \
        --space sc_on_native=/scratch-shared/psobecki/ASL_Citizen/signclip_features_sc_on_native \
        --space augft_on_ce=/scratch-shared/psobecki/ASL_Citizen/signclip_features_augft_on_ce ...

Mirrors dump_kq_features.py exactly; only the space list is parametrised.
"""
import os, sys, argparse, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from eval_asl_citizen_retrieval import load_metadata, load_features

GENDER = ["KING", "QUEEN", "MAN", "WOMAN1", "WOMAN2", "BOY", "GIRL",
          "FATHER", "MOTHER", "BROTHER", "SISTER", "SON", "DAUGHTER",
          "UNCLE", "AUNT", "GRANDFATHER", "GRANDMOTHER", "HUSBAND", "WIFE",
          "NEPHEW", "NIECE"]
FORMGRP = ["HOPE", "CRAZY", "EXPECT", "FAINT", "AGREEMENT", "BRAINSTORM",
           "CAR", "MAKE", "RESCUE", "SAFE", "FIGHT", "BREAK",
           "SHELF", "ROOF", "PARADE", "MARCH", "EQUAL", "FAIR",
           "FIX", "GET", "MORE", "OBVIOUS", "OFFICE", "CLEAR",
           "BOOK", "BOAT", "DIE", "MELT", "PEACE", "DIVIDE"]
ICON = ["VIOLIN", "CAMERA", "TREE", "CAR", "SCISSORS", "BUTTERFLY",
        "TOOTHBRUSH", "SMOKING"]
GLOSSES = sorted(set(GENDER + FORMGRP + ICON))

ap = argparse.ArgumentParser()
ap.add_argument("--space", action="append", required=True,
                help="name=dir; repeatable")
ap.add_argument("--splits_dir", default="/home/psobecki/ASL_Citizen/splits")
ap.add_argument("--out", default="runs/king_queen_analogy/npz")
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
    cent = np.stack([feats[glosses == g].mean(0) for g in present]).astype(np.float32)
    rng = np.random.default_rng(0)
    X, L = [], []
    for g in present:
        idx = np.where(glosses == g)[0]
        take = rng.choice(idx, size=min(14, len(idx)), replace=False)
        X.append(feats[take]); L += [g] * len(take)
    np.savez(f"{args.out}/kqfeat_{name}.npz",
             glosses=np.array(present),
             centroids=cent,
             samp_feats=np.concatenate(X).astype(np.float32),
             samp_labels=np.array(L))
    print(name, "saved", len(present), "glosses,",
          np.concatenate(X).shape[0], "sampled videos")
print("DONE_DUMP_SPACES")
