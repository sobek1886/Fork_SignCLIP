#!/usr/bin/env python3
"""
iconicity_analysis.py - Does iconicity modulate how a sign-video embedding
                        space organises its gloss neighbourhoods?

Extension of the gendered-pair analogy battery (king_queen_analogy.py),
designed 2026-07-19. The analogy experiment showed that offset arithmetic
fails everywhere while categorical proximity tracks visual form. This module
asks the complementary question: iconicity is *within-sign* form-meaning
resemblance (DRINK, HAMMER, BOOK mimic their referents), so for highly
iconic signs a form-driven space and a meaning-driven space should agree,
and for arbitrary signs they should not. Contrasting high- vs low-iconicity
signs therefore isolates where "semantic-looking" neighbourhood structure
can be explained by form alone.

Data: ASL-LEX 2.0 (Sehyr et al., 2021), joined to ASL Citizen via the
ASL-LEX Code column that ships in the ASL Citizen split CSVs (verified
2026-07-19: 2,725/2,731 train glosses carry a valid code; 6 are NA:
SENTENCE1, MINIMUM1, INSTINCT, ABOVE, LETTER2, WELCOME1; codes map to
2,722 distinct ASL-LEX signs; WHATFOR1/2/3 and RESEARCH1/2 share a code).

Primary iconicity rating: D.Iconicity(M) - deaf-signer ratings, 1..7
(hearing non-signer Iconicity(M) available via --rating_col as robustness).

Per gloss g and space, computed from per-video features + gloss centroids:
  compactness   mean_i cos(f_i, c_g^{-i})        (leave-one-out centroid)
  margin        compactness - mean cos(c_g, top-K nearest other centroids)
  nn_acc        fraction of g's videos whose nearest centroid (own = LOO)
                is c_g
  form_overlap  fraction of g's top-K centroid neighbours sharing the
                ASL-LEX MajorLocation.2.0 / Handshape.2.0 / Movement.2.0
                value with g (is embedding proximity phonological?)

Contrast: high vs low iconicity tertile, greedy 1:1 matching of each
high-tertile gloss to an unused low-tertile gloss with the same
LexicalClass and nearest SignFrequency(M) (tie-break: nearest
Neighborhood Density 2.0). Paired sign-flip permutation test (default
10,000 resamples) on the mean paired difference of each metric.
Secondary: Spearman rho of rating vs metric over all rated glosses.

Gender-location probe (--gender_probe): ASL marks gender partly by
location (male ~ forehead, female ~ chin/jaw). Builds the location
direction u = normalize(mean c_g[MinorLocation=Forehead] -
mean c_g[MinorLocation in {Chin, UnderChin}]) and reports
cos(u, normalize(c_f - c_m)) for each gendered pair against a null of
random same-space gloss-pair offsets. If the "gender offset" is mostly a
location-in-form offset, the analogy null gets a concrete mechanistic
reading and H-visual vs H-semantic cannot be separated by gendered pairs
at all.

The ASL-LEX csv is NOT bundled (2.7 MB). Get it on the cluster with:
  wget -O $HOME/ASL_Citizen/asllex2_signdata.csv \
    "https://raw.githack.com/ASL-LEX/asl-lex/master/data-analysis/scripts/data/signdata_updated11-18.csv"
(URL + schema verified 2026-07-19; canonical source: https://osf.io/zpha4/,
license CC BY-NC 4.0, cite Sehyr et al. 2021 + Caselli et al. 2017.)

Usage (single space):
  python iconicity_analysis.py \
      --feature_dir /scratch-shared/psobecki/ASL_Citizen/logos_features_native \
      --splits_dir  $HOME/ASL_Citizen/splits --split train \
      --asllex_csv  $HOME/ASL_Citizen/asllex2_signdata.csv \
      --out_json    runs/king_queen_analogy/iconicity_raw.json

Multi-space + gender probe (mirrors king_queen_analogy.py --space):
  python iconicity_analysis.py \
      --space raw=/scratch-shared/psobecki/ASL_Citizen/logos_features_native \
      --space signclip_ft=/scratch-shared/psobecki/ASL_Citizen/signclip_features_signclip_ft \
      --splits_dir $HOME/ASL_Citizen/splits --split train \
      --asllex_csv $HOME/ASL_Citizen/asllex2_signdata.csv \
      --gender_probe --out_json runs/king_queen_analogy/iconicity_multi.json

Smoke test without cluster features (synthetic embeddings):
  python iconicity_analysis.py --selftest
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Optional heavy deps only needed for real feature loading, not --selftest.
try:
    from eval_asl_citizen_retrieval import (load_metadata, load_features,
                                            DEFAULT_SPLITS, _SPLIT_FILES)
except Exception:                                     # pragma: no cover
    load_metadata = load_features = None
    DEFAULT_SPLITS = "/home/psobecki/ASL_Citizen/splits"
    _SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}

RATING_DEAF = "D.Iconicity(M)"
RATING_HEARING = "Iconicity(M)"
PHONO_COLS = ("MajorLocation.2.0", "MinorLocation.2.0", "Handshape.2.0",
              "Movement.2.0", "SignType.2.0")
MATCH_NUMERIC = ("SignFrequency(M)", "Neighborhood Density 2.0")
GENDER_PAIRS = [("KING", "QUEEN"), ("BOY", "GIRL"),
                ("MAN", "WOMAN1"), ("MAN", "WOMAN2"),
                ("PRINCE", "PRINCESS")]  # PRINCESS absent from ASL Citizen;
                                         # kept so the skip is explicit.


# ---------------------------------------------------------------------------
# data joins
# ---------------------------------------------------------------------------

def load_gloss_codes(splits_dir, split):
    """Gloss -> ASL-LEX Code from the ASL Citizen split csv ('NA' dropped)."""
    path = Path(splits_dir) / _SPLIT_FILES[split]
    g2c = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            code = row["ASL-LEX Code"].strip()
            if code and code != "NA":
                g2c[row["Gloss"].strip()] = code
    return g2c


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_asllex(asllex_csv, rating_col=RATING_DEAF):
    """Code -> {rating, lexical_class, phono..., match covariates}."""
    table = {}
    # The published signdata CSV is an Excel export and is NOT valid UTF-8
    # (byte 0xf4 crashed the 2026-07-19 cluster run) NOR valid cp1252 (it also
    # contains 0x8f, undefined there). latin-1 maps all 256 bytes, so it can
    # never fail; only free-text columns are affected, never Code/ratings/
    # phono fields.
    raw = open(asllex_csv, "rb").read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    import io
    with io.StringIO(text, newline="") as f:
        reader = csv.DictReader(f)
        if "Code" not in reader.fieldnames:
            raise ValueError(f"'Code' column missing in {asllex_csv}; "
                             f"got {reader.fieldnames[:8]}...")
        for row in reader:
            code = (row.get("Code") or "").strip()
            if not code:
                continue
            rec = {"rating": _num(row.get(rating_col)),
                   "rating_hearing": _num(row.get(RATING_HEARING)),
                   "lexical_class": (row.get("LexicalClass") or "NA").strip(),
                   "entry_id": (row.get("EntryID") or "").strip()}
            for c in PHONO_COLS:
                rec[c] = (row.get(c) or "NA").strip()
            for c in MATCH_NUMERIC:
                rec[c] = _num(row.get(c))
            table[code] = rec
    return table


def join_iconicity(glosses, g2c, asllex):
    """-> {gloss: asllex record} for glosses with a code, a row and a rating."""
    joined, missing_code, missing_row, missing_rating = {}, [], [], []
    for g in glosses:
        code = g2c.get(g)
        if code is None:
            missing_code.append(g)
            continue
        rec = asllex.get(code)
        if rec is None:
            missing_row.append(g)
            continue
        if rec["rating"] is None:
            missing_rating.append(g)
            continue
        joined[g] = rec
    stats = {"n_glosses": len(glosses), "n_joined": len(joined),
             "n_missing_code": len(missing_code),
             "n_code_not_in_asllex": len(missing_row),
             "n_missing_rating": len(missing_rating),
             "missing_code": sorted(missing_code),
             "code_not_in_asllex": sorted(missing_row)[:20],
             "missing_rating": sorted(missing_rating)[:20]}
    return joined, stats


# ---------------------------------------------------------------------------
# per-gloss geometry
# ---------------------------------------------------------------------------

def _l2(x, axis=-1, eps=1e-12):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def per_gloss_metrics(feats, glosses, top_k=10):
    """feats (N,D) raw; -> dict gloss -> metrics + centroid matrix info.

    Centroids are means of l2-normalised video vectors, then re-normalised
    (matches king_queen_analogy.build_centers up to the pre-normalisation,
    which is applied there before centroid construction as well).
    """
    feats = _l2(np.asarray(feats, dtype=np.float32))
    glosses = np.asarray(glosses)
    uniq = sorted(set(glosses.tolist()))
    g2i = {g: i for i, g in enumerate(uniq)}
    idx_by_g = {g: np.where(glosses == g)[0] for g in uniq}

    sums = np.zeros((len(uniq), feats.shape[1]), dtype=np.float64)
    counts = np.zeros(len(uniq), dtype=np.int64)
    for g in uniq:
        ii = idx_by_g[g]
        sums[g2i[g]] = feats[ii].sum(axis=0)
        counts[g2i[g]] = len(ii)
    centers = _l2((sums / counts[:, None]).astype(np.float32))

    # centroid-centroid sims for margin + neighbour lists
    cc = centers @ centers.T
    np.fill_diagonal(cc, -np.inf)
    nn_idx = np.argsort(-cc, axis=1)[:, :top_k]
    nn_sim = np.take_along_axis(cc, nn_idx, axis=1)

    out = {}
    for g in uniq:
        gi = g2i[g]
        ii = idx_by_g[g]
        n = len(ii)
        F = feats[ii]                                     # (n, D)
        # leave-one-out own centroid per video
        loo = _l2((sums[gi][None, :] - F).astype(np.float32) / max(n - 1, 1))
        comp = float(np.mean(np.sum(F * loo, axis=1))) if n > 1 else float("nan")
        # nn accuracy: own-LOO sim vs best other centroid
        sims = F @ centers.T                              # (n, G)
        sims[:, gi] = -np.inf
        best_other = sims.max(axis=1)
        own = np.sum(F * loo, axis=1) if n > 1 else np.full(n, np.nan)
        nn_acc = float(np.mean(own > best_other)) if n > 1 else float("nan")
        margin = comp - float(nn_sim[gi].mean())
        out[g] = {"n_videos": int(n), "compactness": comp,
                  "margin": margin, "nn_acc": nn_acc,
                  "neighbors": [uniq[j] for j in nn_idx[gi]]}
    return out, centers, uniq, g2i


def add_form_overlap(metrics, joined, top_k=10):
    """Fraction of top-K embedding neighbours sharing each phono feature."""
    for g, m in metrics.items():
        rec = joined.get(g)
        if rec is None:
            continue
        for col in ("MajorLocation.2.0", "Handshape.2.0", "Movement.2.0"):
            vals = [joined[nb][col] for nb in m["neighbors"][:top_k]
                    if nb in joined and joined[nb][col] != "NA"]
            if vals and rec[col] != "NA":
                m[f"overlap_{col}"] = float(
                    np.mean([v == rec[col] for v in vals]))
            else:
                m[f"overlap_{col}"] = float("nan")


# ---------------------------------------------------------------------------
# high/low contrast with matched controls
# ---------------------------------------------------------------------------

def tertile_split(joined):
    rated = sorted(joined.items(), key=lambda kv: kv[1]["rating"])
    n = len(rated)
    lo = dict(rated[: n // 3])
    hi = dict(rated[-(n // 3):])
    return hi, lo


def greedy_match(hi, lo):
    """Each high gloss -> unused low gloss, same LexicalClass, nearest
    SignFrequency(M) (tie-break Neighborhood Density 2.0). Returns pairs."""
    used, pairs = set(), []
    by_class = defaultdict(list)
    for g, r in lo.items():
        by_class[r["lexical_class"]].append(g)
    for g, r in sorted(hi.items()):
        cands = [c for c in by_class.get(r["lexical_class"], []) if c not in used]
        if not cands:
            continue

        def key(c):
            rc = lo[c]
            df = abs((rc["SignFrequency(M)"] or 0) - (r["SignFrequency(M)"] or 0))
            dn = abs((rc["Neighborhood Density 2.0"] or 0)
                     - (r["Neighborhood Density 2.0"] or 0))
            return (df, dn)
        best = min(cands, key=key)
        used.add(best)
        pairs.append((g, best))
    return pairs


def paired_permutation(diffs, n_perm=10000, seed=0):
    """Two-sided sign-flip test on mean of paired differences."""
    diffs = np.asarray([d for d in diffs if math.isfinite(d)])
    if len(diffs) == 0:
        return float("nan"), float("nan"), 0
    obs = float(diffs.mean())
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    null = (flips * diffs[None, :]).mean(axis=1)
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return obs, p, int(len(diffs))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom else float("nan")


METRIC_KEYS = ("compactness", "margin", "nn_acc",
               "overlap_MajorLocation.2.0", "overlap_Handshape.2.0",
               "overlap_Movement.2.0")


def iconicity_contrast(metrics, joined, n_perm=10000, seed=0):
    hi, lo = tertile_split(joined)
    pairs = greedy_match(hi, lo)
    res = {"n_high": len(hi), "n_low": len(lo), "n_matched_pairs": len(pairs),
           "rating_high_mean": float(np.mean([hi[g]["rating"] for g in hi])),
           "rating_low_mean": float(np.mean([lo[g]["rating"] for g in lo])),
           "metrics": {}}
    for key in METRIC_KEYS:
        diffs = [metrics[a].get(key, float("nan"))
                 - metrics[b].get(key, float("nan"))
                 for a, b in pairs if a in metrics and b in metrics]
        obs, p, n = paired_permutation(diffs, n_perm=n_perm, seed=seed)
        rho = spearman([joined[g]["rating"] for g in metrics if g in joined],
                       [metrics[g].get(key, float("nan"))
                        for g in metrics if g in joined])
        res["metrics"][key] = {"mean_paired_diff_high_minus_low": obs,
                               "perm_p_two_sided": p, "n_pairs_used": n,
                               "spearman_rho_all_rated": rho}
    res["matched_pairs_sample"] = pairs[:15]
    return res


# ---------------------------------------------------------------------------
# gender-offset vs location-direction probe
# ---------------------------------------------------------------------------

def gender_location_probe(centers, uniq, g2i, joined, n_null=1000, seed=0):
    fore = [g for g in uniq if joined.get(g, {}).get("MinorLocation.2.0")
            == "Forehead"]
    chin = [g for g in uniq if joined.get(g, {}).get("MinorLocation.2.0")
            in ("Chin", "UnderChin")]
    if len(fore) < 5 or len(chin) < 5:
        return {"error": f"too few location-tagged glosses "
                         f"(forehead={len(fore)}, chin={len(chin)})"}
    u = _l2(centers[[g2i[g] for g in fore]].mean(axis=0)
            - centers[[g2i[g] for g in chin]].mean(axis=0))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_null):
        a, b = rng.choice(len(uniq), size=2, replace=False)
        null.append(float(_l2(centers[b] - centers[a]) @ u))
    null = np.asarray(null)
    out = {"n_forehead": len(fore), "n_chin": len(chin),
           "null_abs_cos_95pct": float(np.quantile(np.abs(null), 0.95)),
           "pairs": {}}
    for m, f in GENDER_PAIRS:
        if m in g2i and f in g2i:
            c = float(_l2(centers[g2i[f]] - centers[g2i[m]]) @ u)
            pct = float(np.mean(np.abs(null) <= abs(c)))
            out["pairs"][f"{m}->{f}"] = {"cos_offset_locdir": c,
                                         "null_percentile_abs": pct}
        else:
            out["pairs"][f"{m}->{f}"] = {"skipped": "gloss missing"}
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def analyze_space_iconicity(space_name, feats, glosses, g2c, asllex,
                            top_k=10, n_perm=10000, gender_probe=False,
                            seed=0):
    metrics, centers, uniq, g2i = per_gloss_metrics(feats, glosses, top_k=top_k)
    joined, join_stats = join_iconicity(uniq, g2c, asllex)
    add_form_overlap(metrics, joined, top_k=top_k)
    result = {"join_stats": join_stats,
              "contrast": iconicity_contrast(metrics, joined,
                                             n_perm=n_perm, seed=seed)}
    if gender_probe:
        result["gender_location_probe"] = gender_location_probe(
            centers, uniq, g2i, joined, seed=seed)
    print(f"\n  [{space_name}] joined {join_stats['n_joined']}/"
          f"{join_stats['n_glosses']} glosses; "
          f"{result['contrast']['n_matched_pairs']} matched pairs")
    for k, v in result["contrast"]["metrics"].items():
        print(f"    {k:>28}: dHL={v['mean_paired_diff_high_minus_low']:+.4f} "
              f"p={v['perm_p_two_sided']:.4f} "
              f"rho={v['spearman_rho_all_rated']:+.3f}")
    return result


def parse_space(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--space must be NAME=DIR, got {s!r}")
    name, d = s.split("=", 1)
    return name, Path(d)


def selftest():
    """Synthetic end-to-end run: plant tighter clusters for 'iconic' glosses
    and verify the contrast recovers the planted effect."""
    rng = np.random.default_rng(0)
    n_gloss, d, n_vid = 60, 32, 12
    glosses, feats, g2c, asllex = [], [], {}, {}
    classes = ["Noun", "Verb", "Adjective"]
    for i in range(n_gloss):
        g = f"G{i:03d}"
        code = f"T_00_{i:03d}"
        g2c[g] = code
        iconic = i < n_gloss // 2
        rating = 6.0 + rng.normal(0, .3) if iconic else 1.5 + rng.normal(0, .3)
        asllex[code] = {"rating": float(rating), "rating_hearing": float(rating),
                        "lexical_class": classes[i % 3], "entry_id": g.lower(),
                        "MajorLocation.2.0": "Head" if iconic else "Neutral",
                        "MinorLocation.2.0":
                            "Forehead" if i % 5 == 0 else
                            ("Chin" if i % 5 == 1 else "Neutral"),
                        "Handshape.2.0": f"h{i % 7}",
                        "Movement.2.0": f"m{i % 4}",
                        "SignType.2.0": "OneHanded",
                        "SignFrequency(M)": float(3 + (i % 5)),
                        "Neighborhood Density 2.0": float(i % 9)}
        center = rng.normal(0, 1, d)
        spread = 0.15 if iconic else 0.9          # planted effect
        for _ in range(n_vid):
            feats.append(center + rng.normal(0, spread, d))
            glosses.append(g)
    feats = np.asarray(feats, dtype=np.float32)
    res = analyze_space_iconicity("selftest", feats, glosses, g2c, asllex,
                                  top_k=5, n_perm=2000, gender_probe=False)
    comp = res["contrast"]["metrics"]["compactness"]
    assert comp["mean_paired_diff_high_minus_low"] > 0.05, comp
    assert comp["perm_p_two_sided"] < 0.01, comp
    assert res["contrast"]["n_matched_pairs"] >= 10, res["contrast"]
    rho = comp["spearman_rho_all_rated"]
    assert rho > 0.5, rho
    print("\nselftest OK: planted iconicity effect recovered "
          f"(dHL={comp['mean_paired_diff_high_minus_low']:.3f}, "
          f"p={comp['perm_p_two_sided']:.4f}, rho={rho:.3f})")
    return res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature_dir", type=Path, default=None)
    ap.add_argument("--space", action="append", type=parse_space, default=None)
    ap.add_argument("--splits_dir", type=Path, default=Path(DEFAULT_SPLITS))
    ap.add_argument("--split", type=str, default="train",
                    choices=["train", "val", "test"])
    ap.add_argument("--asllex_csv", type=Path, default=None,
                    help="ASL-LEX 2.0 signdata csv (see module docstring "
                         "for the download command)")
    ap.add_argument("--rating_col", type=str, default=RATING_DEAF,
                    help=f"{RATING_DEAF} (deaf, default) or {RATING_HEARING}")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gender_probe", action="store_true")
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--load_workers", type=int, default=32)
    ap.add_argument("--selftest", action="store_true",
                    help="run on synthetic data, no features/csv needed")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.asllex_csv is None:
        sys.exit("--asllex_csv is required (or use --selftest)")
    if load_features is None:
        sys.exit("eval_asl_citizen_retrieval import failed; run from "
                 "fairseq/examples/MMPT")

    spaces = dict(args.space) if args.space else None
    if spaces is None:
        if args.feature_dir is None:
            sys.exit("give --feature_dir or at least one --space NAME=DIR")
        spaces = {"single": args.feature_dir}

    g2c = load_gloss_codes(args.splits_dir, args.split)
    asllex = load_asllex(args.asllex_csv, rating_col=args.rating_col)
    print(f"ASL-LEX rows: {len(asllex)}; ASL Citizen glosses with code: "
          f"{len(g2c)}")

    recs = load_metadata(args.splits_dir, args.split)
    results = {"config": {"rating_col": args.rating_col, "top_k": args.top_k,
                          "n_perm": args.n_perm, "seed": args.seed,
                          "split": args.split}}
    for name, feat_dir in spaces.items():
        print(f"\n================ Space: {name} ({feat_dir}) ================")
        feats, glosses = load_features(
            recs, feat_dir, None, desc=f"Loading {name}",
            workers=args.load_workers, min_coverage=0, allow_partial=True)
        results[name] = analyze_space_iconicity(
            name, feats, glosses, g2c, asllex, top_k=args.top_k,
            n_perm=args.n_perm, gender_probe=args.gender_probe,
            seed=args.seed)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
