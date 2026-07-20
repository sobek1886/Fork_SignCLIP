#!/usr/bin/env python3
"""Local smoke tests for iconicity_analysis.py (no cluster features needed).

Run from fairseq/examples/MMPT:
    python tests/test_iconicity_analysis.py [--splits_dir DIR]

Covers:
  1. --selftest equivalent: planted iconicity effect recovered end-to-end.
  2. ASL-LEX csv parsing on tests/asllex2_sample_12rows.csv — a verbatim
     12-row, 69-column sample of ASL-LEX 2.0 signdata_updated11-18.csv
     (source: github.com/ASL-LEX/asl-lex, fetched 2026-07-19; the full
     2.7 MB file is downloaded on the cluster, see king_queen_analogy.job).
  3. Join against the real ASL Citizen split csv (gloss -> ASL-LEX Code),
     including the NA-code and variant-name (NIGHT1/DOG1/MILK1) paths.
  4. gender_location_probe on synthetic embeddings with a planted
     forehead-vs-chin location axis.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import iconicity_analysis as ia                                   # noqa: E402

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "asllex2_sample_12rows.csv"


def test_selftest():
    ia.selftest()


def test_asllex_parse():
    tab = ia.load_asllex(FIXTURE)
    assert len(tab) == 12
    tree = tab["A_01_002"]
    assert abs(tree["rating"] - 6.32) < 1e-9          # D.Iconicity(M), deaf
    assert abs(tree["rating_hearing"] - 4.232) < 1e-9  # Iconicity(M)
    assert tree["lexical_class"] == "Noun"
    assert tab["A_01_038"]["entry_id"] == "scissors"
    hearing = ia.load_asllex(FIXTURE, rating_col=ia.RATING_HEARING)
    assert abs(hearing["A_01_002"]["rating"] - 4.232) < 1e-9
    print("asllex parse OK")


def test_join(splits_dir):
    train = Path(splits_dir) / "train.csv"
    if not train.exists():
        print(f"SKIP join test: {train} not found")
        return
    g2c = ia.load_gloss_codes(splits_dir, "train")
    assert len(g2c) == 2725, len(g2c)                 # verified 2026-07-19
    tab = ia.load_asllex(FIXTURE)
    glosses = ["TREE", "NIGHT1", "HAMBURGER", "NEPHEW", "EYEGLASSES",
               "MONKEY", "READ", "BOOK", "SCISSORS", "MILK1", "COMPUTER",
               "DOG1", "KING", "QUEEN", "ABOVE"]
    joined, stats = ia.join_iconicity(glosses, g2c, tab)
    assert len(joined) == 12, (sorted(joined), stats)
    assert "KING" not in joined                        # not in 12-row fixture
    assert "ABOVE" in stats["missing_code"]            # NA code in split csv
    print("join OK: 12/12 fixture glosses joined via real Citizen codes")


def test_gender_probe():
    rng = np.random.default_rng(1)
    d = 16
    v = rng.normal(0, 1, d)
    v /= np.linalg.norm(v)
    glosses, feats, joined = [], [], {}
    for i in range(40):
        g = f"S{i:02d}"
        loc = "Forehead" if i < 8 else ("Chin" if i < 16 else "Neutral")
        joined[g] = {"MinorLocation.2.0": loc}
        c = rng.normal(0, 1, d) + (2 * v if loc == "Forehead"
                                   else (-2 * v if loc == "Chin" else 0))
        for _ in range(6):
            feats.append(c + rng.normal(0, .2, d))
            glosses.append(g)
    _, centers, uniq, g2i = ia.per_gloss_metrics(
        np.asarray(feats, dtype=np.float32), glosses, top_k=5)
    old = ia.GENDER_PAIRS
    try:
        ia.GENDER_PAIRS = [("S08", "S00")]   # chin -> forehead: aligns with u
        res = ia.gender_location_probe(centers, uniq, g2i, joined,
                                       n_null=500, seed=0)
    finally:
        ia.GENDER_PAIRS = old
    pair = res["pairs"]["S08->S00"]
    assert pair["cos_offset_locdir"] > 0.5, pair
    assert pair["null_percentile_abs"] > 0.9, pair
    print("gender_location_probe OK: planted location offset detected")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", type=Path,
                    default=Path(ia.DEFAULT_SPLITS))
    args = ap.parse_args()
    test_asllex_parse()
    test_join(args.splits_dir)
    test_gender_probe()
    test_selftest()
    print("\nALL LOCAL TESTS PASSED")
