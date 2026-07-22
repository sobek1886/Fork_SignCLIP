"""Polished king-queen gender-pair figures: t-SNE with per-pair arrows at
several densities, plus a PCA offset-quiver that directly visualises whether a
shared gender direction exists. Raw Logos space."""
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from eval_asl_citizen_retrieval import load_metadata, load_features

RAW = Path("/scratch-shared/psobecki/ASL_Citizen/logos_features_native")
ALL_PAIRS = [("KING", "QUEEN"), ("MAN", "WOMAN1"), ("MAN", "WOMAN2"),
             ("BOY", "GIRL"), ("FATHER", "MOTHER"), ("BROTHER", "SISTER"),
             ("SON", "DAUGHTER"), ("UNCLE", "AUNT"),
             ("GRANDFATHER", "GRANDMOTHER"), ("HUSBAND", "WIFE"),
             ("NEPHEW", "NIECE")]
CORE6 = [("KING", "QUEEN"), ("MAN", "WOMAN1"), ("BOY", "GIRL"),
         ("FATHER", "MOTHER"), ("BROTHER", "SISTER"), ("SON", "DAUGHTER")]
CORE4 = [("KING", "QUEEN"), ("MAN", "WOMAN1"), ("BOY", "GIRL"),
         ("FATHER", "MOTHER")]
D = "runs/king_queen_analogy"

recs = load_metadata(Path("/home/psobecki/ASL_Citizen/splits"), "train")
feats, glosses = load_features(recs, RAW, None, desc="raw", workers=32,
                               min_coverage=0, allow_partial=True)
feats = F.normalize(torch.from_numpy(feats).float(), dim=-1).numpy()
glosses = np.array(glosses)


def centroid(g):
    return feats[glosses == g].mean(0)


def tsne_arrows(pairs, out, title):
    gl = [g for p in pairs for g in p if (glosses == g).any()]
    rng = np.random.default_rng(0)
    X, lab = [], []
    for g in gl:
        idx = np.where(glosses == g)[0]
        take = rng.choice(idx, size=min(14, len(idx)), replace=False)
        X.append(feats[take])
        lab += [g] * len(take)
    X = np.concatenate(X)
    lab = np.array(lab)
    perp = min(15, max(5, len(X) // 6))
    proj = TSNE(n_components=2, perplexity=perp, init="pca",
                random_state=0).fit_transform(X)
    cen = {g: proj[lab == g].mean(0) for g in gl}
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cmap = plt.get_cmap("tab20")
    for i, (m, f) in enumerate(pairs):
        if m not in cen or f not in cen:
            continue
        c = cmap((i * 2) % 20)
        ax.scatter(proj[lab == m, 0], proj[lab == m, 1], color=c, marker="o",
                   s=16, alpha=0.30)
        ax.scatter(proj[lab == f, 0], proj[lab == f, 1], color=c, marker="^",
                   s=16, alpha=0.30)
        mx, my = cen[m]
        fx, fy = cen[f]
        ax.annotate("", xy=(fx, fy), xytext=(mx, my),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2, alpha=0.95))
        ax.text(mx, my, m, fontsize=7, color=c, ha="center", va="center",
                fontweight="bold")
        ax.text(fx, fy, f, fontsize=7, color=c, ha="center", va="center",
                fontweight="bold")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def offset_quiver(pairs, out):
    offs, labs = [], []
    for m, f in pairs:
        if (glosses == m).any() and (glosses == f).any():
            o = centroid(f) - centroid(m)
            offs.append(o / (np.linalg.norm(o) + 1e-9))
            labs.append(f"{m}->{f}")
    O = np.stack(offs)
    cons = float(np.mean([O[i] @ O[j] for i, j in combinations(range(len(O)), 2)]))
    P = PCA(n_components=2).fit_transform(O)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cmap = plt.get_cmap("tab20")
    lim = np.abs(P).max() * 1.25
    for i, (v, l) in enumerate(zip(P, labs)):
        c = cmap((i * 2) % 20)
        ax.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2))
        ax.text(v[0] * 1.05, v[1] * 1.05, l, fontsize=7, color=c)
    ax.scatter([0], [0], color="k", s=25, zorder=5)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(f"Gender offsets (female-male), PCA-2D, {len(labs)} pairs; "
                 f"mean pairwise cos = {cons:.3f}\n(arrows clustered in one "
                 f"direction = shared gender vector)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out, "consistency", round(cons, 3))


tsne_arrows(ALL_PAIRS, f"{D}/kq_arrows_11.png",
            "raw Logos: 11 gendered pairs (t-SNE; arrow = female-male). "
            "NB t-SNE distorts directions.")
tsne_arrows(CORE6, f"{D}/kq_arrows_6.png",
            "raw Logos: 6 gendered pairs (t-SNE; arrow = female-male).")
tsne_arrows(CORE4, f"{D}/kq_arrows_4.png",
            "raw Logos: 4 gendered pairs (t-SNE; arrow = female-male).")
offset_quiver(ALL_PAIRS, f"{D}/kq_offset_quiver.png")
print("DONE_KQ_ARROWS")
