#!/usr/bin/env python3
"""
analyze_augmentation_geometry.py - Augmented-vs-original proximity & the sim-to-real
                                   gap in the feature space (thesis experiment E4).

The supervisor's question has two halves that CAN BOTH BE TRUE at once, so we test
both explicitly:

  H-local  : each augment lands near its OWN original — a small, sign-preserving
             appearance perturbation.
  H-gap    : augments AS A GROUP occupy a systematically separable sub-region — a
             "synthetic" direction distinguishes reals from synthetics regardless
             of sign/signer (a sim-to-real domain shift).

The interesting finding is the conjunction: augments individually close to their
originals (cos ~0.9) yet collectively linearly separable from reals (AUC ~0.9) via
a small systematic shift. The four analysis blocks:

BLOCK 1 — Calibrated proximity ("do augments land near their originals?").
  cos(orig, aug) per variant, CALIBRATED against per-anchor control distributions
  drawn from the same features:
    - same-gloss / same-signer / other take     -> repeatability ceiling
    - same-gloss / DIFFERENT signer             -> the real signer-change magnitude
    - DIFFERENT gloss / same signer             -> sign-identity floor at fixed person
    - different gloss / different signer         -> global floor
  Headline metrics: normalized displacement (1-cos_aug)/(1-cos_same-gloss-diff-signer)
  = "the edit moves you X% as far as a real signer change"; and the margin over the
  sign floor (aug closer to its own sign than to any other sign by the same signer?).

BLOCK 2 — Sim-to-real gap (the core novel part).
  (a) real-vs-synthetic linear separability (ROC-AUC, GroupKFold by video id so a
      video's orig+aug never split across folds) per variant and pooled — reported
      on RAW features and on L2-NORMALIZED features (the raw-vs-normalized AUC gap
      says how much of the synthetic signal is magnitude vs direction).
  (b) shared displacement axis: direction consistency, anisotropy ||mean d||/mean||d||,
      PCA EVR of the displacement vectors.
  (c) de-shift test, leak-free: within each CV fold the mean displacement is
      estimated on the TRAIN folds only, subtracted from train and test augments,
      and a fresh classifier is scored on the test fold. The drop vs (a) = how much
      of the gap is a simple correctable translation vs. something structural.

BLOCK 3 — Sign preservation (kept from the old S3).
  1-NN gloss retention @1/@5 over the real gallery (self excluded) + signer-prototype
  flip rate: confirms the edit moved APPEARANCE, not MEANING.

BLOCK 4 — Mechanism: is the synthetic axis the SIGNER axis?
  Estimate from real data a signer subspace (within-gloss between-signer prototype
  differences -> PCA) and a sign subspace (between-gloss prototype differences ->
  PCA, orthogonalised against the signer subspace). Decompose each displacement:
  fraction of its energy in the signer subspace vs the sign subspace vs residual.
  A faithful appearance edit moves along the real signer axis; a pure synthetic
  artifact moves in a direction orthogonal to both.

Everything runs per --space, so pointing it at the intervention re-extractions
(run1 / run3 / glosscon / sdda, ...) answers "does the intervention close the gap
and pull augments toward their originals?" — tying E4 to E2/E3. Spaces WITHOUT
variant .npy files are skipped (they contain originals only).
NOTE on splits: the original 6000-video augment batch is train-only
(aug_video_list ⊂ train.csv), so with --split test those pairs simply drop out.
Intervention-space numbers on --split train are OPTIMISTIC (the model trained on
those exact augments). The held-out batch `augmented_frames_512_test` (500
test-split videos x 4 variants) exists for exactly this: extract it into the same
feature dir and run with --split test for the honest intervention comparison.

Feature pairing is by filename in ONE dir: {dataset}_{id}.npy (original) and
{dataset}_{id}_{variant}.npy (augment) — the layout the extraction jobs produce.

Usage (on Snellius; CPU is fine)
  python analyze_augmentation_geometry.py \
      --space frozen_logos=/scratch-shared/psobecki/ASL_Citizen/logos_features_native \
      --space run1=/scratch-shared/psobecki/ASL_Citizen/logos_features_run1 \
      --space glosscon=/scratch-shared/psobecki/ASL_Citizen/logos_features_run_full_glosscon \
      --splits_dir /home/psobecki/ASL_Citizen/splits \
      --split train \
      --out_json runs/aug_geometry/aug_geometry.json \
      --out_fig_dir runs/aug_geometry/figs
"""
import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_asl_citizen_per_signer import read_records, load_with_records

DEFAULT_VARIANTS = ["glasses", "shirt_1", "signer_swap", "skin_mst_diffusion"]

# The two control distributions worth plotting against cos(orig,aug).
_KEY_CONTROLS = ["same_gloss_diff_signer", "diff_gloss_same_signer"]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def parse_space(s):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--space must be NAME=DIR, got {s!r}")
    name, d = s.split("=", 1)
    return name, Path(d)


def l2n(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def pooled(path):
    f = np.load(path).astype(np.float32)
    if f.ndim == 2:
        if f.shape[0] == 0:
            raise ValueError("empty feature")
        f = f.mean(axis=0)
    return f


def boot_ci(vals, fn=np.median, n_boot=1000, rng=None, alpha=0.05):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [None, None]
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    stats = fn(vals[idx], axis=1)
    return [float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))]


def dist_summary(vals):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "q25": float(np.quantile(vals, 0.25)),
        "q75": float(np.quantile(vals, 0.75)),
        "std": float(vals.std()),
    }


def discover_pairs(feat_dir: Path, variants):
    """variant -> list of (aug_path, feat_id). Pure filename logic."""
    pairs = defaultdict(list)
    suffixes = {v: f"_{v}" for v in variants}
    for p in feat_dir.glob("*.npy"):
        stem = p.stem
        for v, suf in suffixes.items():
            if stem.endswith(suf):
                pairs[v].append((p, stem[: -len(suf)]))
                break
    return pairs


def load_aug(pair_list, workers=16, desc="aug"):
    """[(aug_path, feat_id)] -> (feats (n,D), feat_ids) for those that load."""
    def _load(item):
        path, fid = item
        try:
            return pooled(path), fid
        except Exception:
            return None
    feats, ids = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_load, pair_list):
            if res is not None:
                feats.append(res[0])
                ids.append(res[1])
    if not feats:
        return np.zeros((0, 0), np.float32), []
    return np.stack(feats), ids


# --------------------------------------------------------------------------- #
# BLOCK 1 — calibrated proximity
# --------------------------------------------------------------------------- #
def build_group_index(g_gloss, g_signer):
    by_gloss, by_gs, by_signer = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, (g, s) in enumerate(zip(g_gloss, g_signer)):
        by_gloss[g].append(i)
        by_gs[(g, s)].append(i)
        by_signer[s].append(i)
    return ({g: np.asarray(v) for g, v in by_gloss.items()},
            {k: np.asarray(v) for k, v in by_gs.items()},
            {s: np.asarray(v) for s, v in by_signer.items()})


def _mean_cos(anchor_vec, Gn, pool, K, rng):
    if pool is None or len(pool) == 0:
        return np.nan
    take = pool if len(pool) <= K else rng.choice(pool, K, replace=False)
    return float((Gn[take] @ anchor_vec).mean())


def compute_controls(anchors, Gn, g_gloss, g_signer, by_gloss, by_gs, by_signer, K, rng):
    """anchor gallery-idx -> {control_name: mean cosine}. Computed once, reused across variants."""
    out = {}
    all_idx = np.arange(Gn.shape[0])
    for a in anchors:
        g, s = g_gloss[a], g_signer[a]
        av = Gn[a]
        gs = by_gs[(g, s)]
        take_pool = gs[gs != a]
        gloss_pool = by_gloss[g]
        signer_diff_pool = gloss_pool[g_signer[gloss_pool] != s]           # same gloss, diff signer
        spool = by_signer[s]
        diffsign_pool = spool[g_gloss[spool] != g]                          # same signer, diff gloss
        # global: sample random, drop same gloss or same signer
        cand = rng.choice(all_idx, size=min(len(all_idx), K * 4), replace=False)
        cand = cand[(g_gloss[cand] != g) & (g_signer[cand] != s)]
        out[a] = {
            "same_gloss_same_signer": _mean_cos(av, Gn, take_pool, K, rng),
            "same_gloss_diff_signer": _mean_cos(av, Gn, signer_diff_pool, K, rng),
            "diff_gloss_same_signer": _mean_cos(av, Gn, diffsign_pool, K, rng),
            "diff_gloss_diff_signer": _mean_cos(av, Gn, cand[:K], K, rng),
        }
    return out


def block1_proximity(variant_data, Gn, controls, rng, n_boot):
    """variant_data[v] = dict(orig_idx, aug_norm). Returns per-variant proximity stats + raw arrays for plotting."""
    result, plot = {}, {}
    for v, d in variant_data.items():
        oi, An = d["orig_idx"], d["aug_norm"]
        A = (Gn[oi] * An).sum(axis=1)                                       # cos(orig, aug), rowwise
        c_signer = np.array([controls[a]["same_gloss_diff_signer"] for a in oi])
        c_signfloor = np.array([controls[a]["diff_gloss_same_signer"] for a in oi])
        # normalized displacement: edit distance as a fraction of a real signer change
        denom = 1.0 - c_signer
        ok = np.isfinite(denom) & (denom > 1e-6) & np.isfinite(A)
        norm_disp = (1.0 - A[ok]) / denom[ok]
        margin = A - c_signfloor
        result[v] = {
            "n": int(A.size),
            "cos_orig_aug": dist_summary(A),
            "cos_orig_aug_median_ci": boot_ci(A, rng=rng, n_boot=n_boot),
            "control_same_gloss_diff_signer": dist_summary(c_signer),
            "control_diff_gloss_same_signer": dist_summary(c_signfloor),
            "normalized_displacement": dist_summary(norm_disp),
            "normalized_displacement_median_ci": boot_ci(norm_disp, rng=rng, n_boot=n_boot),
            "margin_over_sign_floor": dist_summary(margin),
            "frac_aug_closer_than_diff_signer":
                float(np.mean((A > c_signer)[np.isfinite(c_signer)])) if np.isfinite(c_signer).any() else None,
            "frac_aug_above_sign_floor":
                float(np.mean((margin > 0)[np.isfinite(margin)])) if np.isfinite(margin).any() else None,
        }
        plot[v] = {"cos": A, "c_signer": c_signer, "c_signfloor": c_signfloor}
    return result, plot


# --------------------------------------------------------------------------- #
# BLOCK 2 — sim-to-real gap
# --------------------------------------------------------------------------- #
def separability(X, y, groups, seed=0):
    """GroupKFold ROC-AUC for real(0) vs synthetic(1). Returns None if degenerate."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import GroupKFold, cross_val_score
    except ImportError:
        return None
    y = np.asarray(y)
    n_groups = len(set(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2 or len(set(y.tolist())) < 2:
        return None
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    try:
        scores = cross_val_score(pipe, X, y, groups=np.asarray(groups),
                                 cv=GroupKFold(n_splits=n_splits), scoring="roc_auc")
    except Exception as e:
        return {"error": str(e)}
    return {"auc_mean": float(scores.mean()), "auc_std": float(scores.std()),
            "n_splits": int(n_splits), "n": int(len(y))}


def separability_deshift(orig, aug, groups, seed=0):
    """Leak-free de-shift AUC over row-aligned (orig, aug) pairs.

    GroupKFold by video id; within each fold the mean displacement is estimated
    on the TRAIN-fold pairs ONLY, subtracted from train AND test augments, and a
    fresh classifier is fit on the train fold and scored on the test fold. This
    avoids the optimism of estimating the shift on the evaluation samples."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    groups = np.asarray(groups)
    n_splits = min(5, len(set(groups.tolist())))
    if n_splits < 2:
        return None
    aucs = []
    try:
        rng = np.random.RandomState(seed)
        for tr, te in GroupKFold(n_splits).split(orig, groups=groups):
            md = (aug[tr] - orig[tr]).mean(axis=0)
            # Unpaired fit: each train pair contributes EITHER its original OR
            # its de-shifted augment, never both. With both, the mean-centred
            # displacements sum to zero and w=0 is an exact stationary point
            # of any class-balanced linear loss — the optimiser then returns
            # the zero classifier and AUC is pinned at exactly 0.5 (observed
            # 2026-07-19). Random assignment breaks the symmetry; the test
            # fold keeps both members.
            half = rng.rand(len(tr)) < 0.5
            Xtr = np.concatenate([orig[tr][~half], aug[tr][half] - md], axis=0)
            ytr = np.array([0] * int((~half).sum()) + [1] * int(half.sum()))
            Xte = np.concatenate([orig[te], aug[te] - md], axis=0)
            yte = np.array([0] * len(te) + [1] * len(te))
            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            )
            pipe.fit(Xtr, ytr)
            aucs.append(roc_auc_score(yte, pipe.decision_function(Xte)))
    except Exception as e:
        return {"error": str(e)}
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "n_splits": int(n_splits), "n": int(2 * len(groups))}


def direction_stats(deltas):
    mags = np.linalg.norm(deltas, axis=1)
    dhat = l2n(deltas)
    n = dhat.shape[0]
    s = np.linalg.norm(dhat.sum(axis=0)) ** 2 - n
    mean_pairwise_cos = float(s / (n * (n - 1))) if n > 1 else float("nan")
    mean_delta = deltas.mean(axis=0)
    anisotropy = float(np.linalg.norm(mean_delta) / (mags.mean() + 1e-12))   # ||mean d|| / mean||d||
    evr1 = evr5 = float("nan")
    if n > 5:
        try:
            from sklearn.decomposition import PCA
            k = min(5, n - 1, deltas.shape[1])
            pca = PCA(n_components=k).fit(deltas)
            evr1 = float(pca.explained_variance_ratio_[0])
            evr5 = float(pca.explained_variance_ratio_[:k].sum())
        except ImportError:
            pass
    return {
        "n": int(n),
        "mean_delta_norm": float(mags.mean()),
        "direction_consistency_mean_pairwise_cos": mean_pairwise_cos,
        "anisotropy_meannorm_over_meannorm": anisotropy,
        "pca_evr_top1": evr1,
        "pca_evr_top5": evr5,
    }


def block2_gap(variant_data, G, unique_orig_idx, seed):
    """Separability (full + de-shifted), displacement direction, cross-variant direction cos."""
    result = {"variants": {}}
    mean_dirs = {}
    pooled_X, pooled_y, pooled_groups = [], [], []
    # reals appear once (each unique aug'd original)
    reals = G[unique_orig_idx]
    for v, d in variant_data.items():
        oi, aug = d["orig_idx"], d["aug_raw"]
        deltas = aug - G[oi]
        dstats = direction_stats(deltas)
        mean_delta = deltas.mean(axis=0)
        mean_dirs[v] = l2n(mean_delta)

        # 2a separability real vs this variant (groups = video id so orig+aug share a fold),
        # on raw AND l2-normalized features (raw-vs-norm gap = magnitude share of the signal)
        Xv = np.concatenate([G[oi], aug], axis=0)
        yv = np.array([0] * len(oi) + [1] * len(aug))
        gv = np.concatenate([oi, oi])                                       # same group id for a video's orig & aug
        sep_full = separability(Xv, yv, gv, seed)
        sep_full_l2 = separability(l2n(Xv), yv, gv, seed)

        # 2c de-shift, leak-free (per-fold shift estimation; see separability_deshift)
        sep_deshift = separability_deshift(G[oi], aug, oi, seed)
        drop = None
        if sep_full and sep_deshift and "auc_mean" in sep_full and "auc_mean" in sep_deshift:
            drop = float(sep_full["auc_mean"] - sep_deshift["auc_mean"])

        result["variants"][v] = {
            "displacement": dstats,
            "separability_full": sep_full,
            "separability_full_l2norm": sep_full_l2,
            "separability_deshift": sep_deshift,
            "auc_drop_after_deshift": drop,
        }
        pooled_X.append(aug); pooled_y += [1] * len(aug); pooled_groups.append(oi)

    # pooled: reals (each once) vs ALL augments across variants
    if pooled_X:
        Xall = np.concatenate([reals] + pooled_X, axis=0)
        yall = np.array([0] * len(reals) + pooled_y)
        gall = np.concatenate([unique_orig_idx] + pooled_groups)
        result["separability_all_variants"] = separability(Xall, yall, gall, seed)
        result["separability_all_variants_l2norm"] = separability(l2n(Xall), yall, gall, seed)

    # cross-variant direction cosine matrix
    if len(mean_dirs) > 1:
        names = sorted(mean_dirs)
        M = np.stack([mean_dirs[n] for n in names])
        result["cross_variant_direction_cos"] = {"order": names, "matrix": (M @ M.T).round(4).tolist()}
    return result, mean_dirs


# --------------------------------------------------------------------------- #
# BLOCK 3 — sign preservation (motion retained / appearance moved)
# --------------------------------------------------------------------------- #
def block3_preservation(variant_data, Gn, g_gloss, g_signer, id_to_gidx, chunk=2048):
    signers = sorted(set(g_signer.tolist()))
    proto = l2n(np.stack([Gn[g_signer == s].mean(axis=0) for s in signers]))
    signers = np.array(signers)
    result = {}
    for v, d in variant_data.items():
        oi, An, ids = d["orig_idx"], d["aug_norm"], d["feat_ids"]
        true_gloss = g_gloss[oi]
        ret1 = np.zeros(len(oi), bool)
        ret5 = np.zeros(len(oi), bool)
        for c in range(0, len(oi), chunk):
            sl = slice(c, c + chunk)
            sims = An[sl] @ Gn.T                                            # (b, N)
            b = sims.shape[0]
            sims[np.arange(b), oi[sl]] = -np.inf                            # exclude self
            nn1 = sims.argmax(axis=1)
            ret1[sl] = g_gloss[nn1] == true_gloss[sl]
            top5 = np.argpartition(-sims, 5, axis=1)[:, :5]
            ret5[sl] = [true_gloss[sl][i] in g_gloss[top5[i]].tolist() for i in range(b)]
        near_aug = signers[(An @ proto.T).argmax(axis=1)]
        near_orig = signers[(Gn[oi] @ proto.T).argmax(axis=1)]
        result[v] = {
            "gloss_retention_at1": float(ret1.mean()),
            "gloss_retention_at5": float(ret5.mean()),
            "signer_flip_rate": float((near_aug != near_orig).mean()),
        }
    return result


# --------------------------------------------------------------------------- #
# BLOCK 4 — signer-axis vs sign-axis decomposition
# --------------------------------------------------------------------------- #
def _pca_dirs(M, k):
    from sklearn.decomposition import PCA
    k = max(1, min(k, M.shape[0] - 1, M.shape[1]))
    return PCA(n_components=k).fit(M).components_                            # (k, D) orthonormal


def _orthonormal_complement(U_perp_of, U):
    """Rows of U with the U_perp_of components removed, re-orthonormalised.
    Uses the R diagonal to drop directions fully explained by U_perp_of (QR
    always unit-normalises Q, so filter on |diag(R)|, not on ||Q columns||)."""
    B = U - (U @ U_perp_of.T) @ U_perp_of
    Q, R = np.linalg.qr(B.T)                                                # (D, k)
    keep = np.abs(np.diag(R)) > 1e-6
    return Q.T[keep]


def _energy_fraction(deltas, U):
    if U is None or U.shape[0] == 0:
        return {"n": 0}
    proj = deltas @ U.T
    num = (proj ** 2).sum(axis=1)
    den = (deltas ** 2).sum(axis=1)
    frac = num / np.maximum(den, 1e-12)
    return dist_summary(frac)


def block4_subspaces(variant_data, G, g_gloss, g_signer, k):
    try:
        import sklearn.decomposition  # noqa: F401
    except ImportError:
        return {"skipped": "sklearn unavailable"}
    # signer subspace: within-gloss between-signer prototype deviations
    signer_vecs = []
    for g in set(g_gloss.tolist()):
        gi = np.where(g_gloss == g)[0]
        sgs = set(g_signer[gi].tolist())
        if len(sgs) < 2:
            continue
        gmean = G[gi].mean(axis=0)
        for s in sgs:
            gsi = gi[g_signer[gi] == s]
            signer_vecs.append(G[gsi].mean(axis=0) - gmean)
    # sign subspace: between-gloss prototype deviations
    glosses = sorted(set(g_gloss.tolist()))
    gproto = np.stack([G[g_gloss == g].mean(axis=0) for g in glosses])
    sign_vecs = gproto - gproto.mean(axis=0, keepdims=True)

    if len(signer_vecs) < 3 or sign_vecs.shape[0] < 3:
        return {"skipped": "insufficient prototypes for subspaces"}

    U_signer = _pca_dirs(np.stack(signer_vecs), k)
    U_sign = _pca_dirs(sign_vecs, k)
    U_sign_perp = _orthonormal_complement(U_signer, U_sign)

    out = {"signer_subspace_dim": int(U_signer.shape[0]),
           "sign_subspace_dim_orthogonalised": int(U_sign_perp.shape[0]),
           "variants": {}}
    for v, d in variant_data.items():
        deltas = d["aug_raw"] - G[d["orig_idx"]]
        e_signer = _energy_fraction(deltas, U_signer)
        e_sign = _energy_fraction(deltas, U_sign_perp)
        mean_dir = l2n(deltas.mean(axis=0))
        out["variants"][v] = {
            "energy_in_signer_subspace": e_signer,
            "energy_in_sign_subspace_perp": e_sign,
            "shared_axis_energy_in_signer_subspace":
                float((mean_dir @ U_signer.T) @ (U_signer @ mean_dir)),
        }
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_proximity(space, plot, out_png):
    try:
        variants = list(plot.keys())
        data, labels = [], []
        for v in variants:
            arr = plot[v]["cos"][np.isfinite(plot[v]["cos"])]
            if len(arr):
                data.append(arr); labels.append(v)
        # one pooled copy of each control (from the first variant's anchors)
        if variants:
            cs = plot[variants[0]]["c_signer"]; cs = cs[np.isfinite(cs)]
            cf = plot[variants[0]]["c_signfloor"]; cf = cf[np.isfinite(cf)]
            if len(cs):
                data.append(cs); labels.append("ctrl: same-gloss\ndiff-signer")
            if len(cf):
                data.append(cf); labels.append("ctrl: diff-gloss\nsame-signer")
        if not data:
            return
        fig, ax = plt.subplots(figsize=(1.6 * len(data) + 2, 5))
        ax.violinplot(data, showmedians=True)
        ax.set_xticks(range(1, len(data) + 1)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("cosine similarity to source video")
        ax.set_title(f"{space}: augment proximity vs. controls")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    except Exception as e:
        print(f"  [fig_proximity failed: {e}]")


def fig_separability(space, block2, out_png):
    try:
        variants, full, desh = [], [], []
        for v, r in block2["variants"].items():
            sf, sd = r.get("separability_full"), r.get("separability_deshift")
            if not sf or "auc_mean" not in sf:
                continue
            variants.append(v); full.append(sf["auc_mean"])
            desh.append(sd["auc_mean"] if sd and "auc_mean" in sd else np.nan)
        if not variants:
            return
        x = np.arange(len(variants)); w = 0.38
        fig, ax = plt.subplots(figsize=(1.5 * len(variants) + 2, 5))
        ax.bar(x - w / 2, full, w, label="full")
        ax.bar(x + w / 2, desh, w, label="after de-shift")
        ax.axhline(0.5, ls="--", c="k", lw=1, label="chance")
        ax.set_xticks(x); ax.set_xticklabels(variants, fontsize=8)
        ax.set_ylabel("real-vs-synthetic ROC-AUC"); ax.set_ylim(0.4, 1.0)
        ax.set_title(f"{space}: sim-to-real separability"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    except Exception as e:
        print(f"  [fig_separability failed: {e}]")


def fig_subspace(space, block4, out_png):
    try:
        if "variants" not in block4:
            return
        variants = list(block4["variants"].keys())
        sgn = [block4["variants"][v]["energy_in_signer_subspace"].get("median", 0) for v in variants]
        sgnp = [block4["variants"][v]["energy_in_sign_subspace_perp"].get("median", 0) for v in variants]
        res = [max(0.0, 1 - a - b) for a, b in zip(sgn, sgnp)]
        x = np.arange(len(variants))
        fig, ax = plt.subplots(figsize=(1.5 * len(variants) + 2, 5))
        ax.bar(x, sgn, label="signer subspace")
        ax.bar(x, sgnp, bottom=sgn, label="sign subspace (⊥signer)")
        ax.bar(x, res, bottom=[a + b for a, b in zip(sgn, sgnp)], label="residual")
        ax.set_xticks(x); ax.set_xticklabels(variants, fontsize=8)
        ax.set_ylabel("fraction of displacement energy (median)")
        ax.set_title(f"{space}: where the edit moves the embedding"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    except Exception as e:
        print(f"  [fig_subspace failed: {e}]")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", action="append", type=parse_space, required=True,
                    help="NAME=FEATURE_DIR; repeat to compare spaces")
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--dataset_name", default="asl_citizen")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--n_controls", type=int, default=64,
                    help="control samples per pool per anchor (block 1)")
    ap.add_argument("--subspace_k", type=int, default=10,
                    help="dims for the signer/sign subspaces (block 4)")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pairs", type=int, default=None, help="cap pairs per variant (debug)")
    ap.add_argument("--load_workers", type=int, default=32)
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--out_fig_dir", type=Path, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    recs = read_records(args.splits_dir, args.split, args.dataset_name, with_signer=True)

    results = {"_meta": {"split": args.split, "variants": args.variants,
                         "n_controls": args.n_controls, "subspace_k": args.subspace_k}}

    for space_name, feat_dir in dict(args.space).items():
        print(f"\n================ Space: {space_name} ({feat_dir}) ================")
        feat_dir = Path(feat_dir)
        if not feat_dir.exists():
            print("  dir not found — skipping"); results[space_name] = {"skipped": "dir not found"}; continue

        pairs = discover_pairs(feat_dir, args.variants)
        if not any(pairs.values()):
            print("  no variant files — skipping (originals-only space)")
            results[space_name] = {"skipped": "no variant files"}; continue

        # gallery = full-split originals present in this dir
        G, gallery_recs = load_with_records(recs, feat_dir, workers=args.load_workers,
                                            desc=f"{space_name} gallery")
        g_gloss = np.array([r["gloss"] for r in gallery_recs])
        g_signer = np.array([r["signer"] for r in gallery_recs])
        id_to_gidx = {r["feat_id"]: i for i, r in enumerate(gallery_recs)}
        Gn = l2n(G)
        by_gloss, by_gs, by_signer = build_group_index(g_gloss, g_signer)
        print(f"  gallery: {G.shape[0]} originals, {G.shape[1]}-dim, "
              f"{len(by_gloss)} glosses, {len(by_signer)} signers")

        # load augments, align to gallery (need gloss/signer of the source)
        variant_data = {}
        for v in args.variants:
            pl = pairs.get(v, [])
            if args.max_pairs:
                pl = pl[: args.max_pairs]
            if len(pl) < 5:
                if pl:
                    print(f"  {v}: only {len(pl)} pairs — skipped"); continue
                continue
            aug_raw, ids = load_aug(pl, workers=args.load_workers, desc=f"{space_name}:{v}")
            keep = [i for i, fid in enumerate(ids) if fid in id_to_gidx]
            if len(keep) < 5:
                print(f"  {v}: {len(keep)} pairs with a source in the split — skipped"); continue
            aug_raw = aug_raw[keep]
            oi = np.array([id_to_gidx[ids[i]] for i in keep])
            variant_data[v] = {"orig_idx": oi, "aug_raw": aug_raw,
                               "aug_norm": l2n(aug_raw), "feat_ids": [ids[i] for i in keep]}
            print(f"  {v}: {len(oi)} paired augments")

        if not variant_data:
            results[space_name] = {"skipped": "no usable variant pairs in split"}; continue

        unique_orig_idx = np.array(sorted(set(int(i) for d in variant_data.values()
                                              for i in d["orig_idx"])))
        anchors = unique_orig_idx.tolist()

        # controls once, then all blocks
        print(f"  computing controls for {len(anchors)} anchors ...")
        controls = compute_controls(anchors, Gn, g_gloss, g_signer,
                                    by_gloss, by_gs, by_signer, args.n_controls, rng)

        b1, plot = block1_proximity(variant_data, Gn, controls, rng, args.n_boot)
        b2, _ = block2_gap(variant_data, G, unique_orig_idx, args.seed)
        b3 = block3_preservation(variant_data, Gn, g_gloss, g_signer, id_to_gidx)
        b4 = block4_subspaces(variant_data, G, g_gloss, g_signer, args.subspace_k)

        results[space_name] = {
            "n_gallery": int(G.shape[0]),
            "block1_proximity": b1,
            "block2_gap": b2,
            "block3_preservation": b3,
            "block4_subspaces": b4,
        }

        # console headline
        for v in variant_data:
            p, g2 = b1[v], b2["variants"][v]
            auc = (g2["separability_full"] or {}).get("auc_mean", float("nan"))
            print(f"    {v:>20}: cos={p['cos_orig_aug']['median']:.3f}  "
                  f"norm_disp={p['normalized_displacement'].get('median', float('nan')):.2f}  "
                  f"AUC={auc:.3f}  gloss@1={b3[v]['gloss_retention_at1']:.3f}  "
                  f"signer-flip={b3[v]['signer_flip_rate']:.3f}")

        if args.out_fig_dir:
            args.out_fig_dir.mkdir(parents=True, exist_ok=True)
            fig_proximity(space_name, plot, args.out_fig_dir / f"proximity_{space_name}.png")
            fig_separability(space_name, b2, args.out_fig_dir / f"separability_{space_name}.png")
            fig_subspace(space_name, b4, args.out_fig_dir / f"subspace_{space_name}.png")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
