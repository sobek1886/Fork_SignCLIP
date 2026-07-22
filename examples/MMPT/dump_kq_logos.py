"""Same as dump_kq_features.py but for the Logos backbone-FT variant spaces,
so the king-queen / form-neighbour / iconicity figures can be re-plotted
locally for the intervention Logos features too."""
import os, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from eval_asl_citizen_retrieval import load_metadata, load_features

R = Path("/scratch-shared/psobecki/ASL_Citizen")
SPACES = {"consistency": "logos_features_run1",
          "ce": "logos_features_run2",
          "aug": "logos_features_run3",
          "sdda": "logos_features_run_sdda"}
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

OUT = "runs/king_queen_analogy/npz"
os.makedirs(OUT, exist_ok=True)
recs = load_metadata(Path("/home/psobecki/ASL_Citizen/splits"), "train")

for name, d in SPACES.items():
    feats, glosses = load_features(recs, R / d, None, desc=name, workers=32,
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
    np.savez(f"{OUT}/kqfeat_{name}.npz",
             glosses=np.array(present),
             centroids=cent,
             samp_feats=np.concatenate(X).astype(np.float32),
             samp_labels=np.array(L))
    print(name, "saved", len(present), "glosses,",
          np.concatenate(X).shape[0], "sampled videos")
print("DONE_DUMP_LOGOS")
