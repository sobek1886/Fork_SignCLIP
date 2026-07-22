"""Form-structure probes on raw Logos features.
Panel 1: t-SNE of phonological form-neighbor GROUPS (glosses sharing all 5
         ASL-LEX form features but different meanings) -> do form-twins cluster?
Panel 2: violin of embedding cosine between gloss centroids, bucketed by the
         number of shared ASL-LEX form features (0..5) -> dose-response of
         form-overlap vs embedding similarity."""
import numpy as np, torch, torch.nn.functional as F, csv, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from eval_asl_citizen_retrieval import load_metadata, load_features

RAW = Path("/scratch-shared/psobecki/ASL_Citizen/logos_features_native")
D = "runs/king_queen_analogy"
FEATS = ["Movement.2.0", "MajorLocation.2.0", "MinorLocation.2.0",
         "Handshape.2.0", "SignType.2.0"]

# ---- vocab + ASL-LEX form signatures ----
vocab = set()
for f in glob.glob("/home/psobecki/ASL_Citizen/splits/*.csv"):
    with open(f, newline="") as fh:
        for row in csv.reader(fh):
            for c in row:
                c = c.strip()
                if c and not c.lower().endswith(".mp4"):
                    vocab.add(c.upper())
form = {}
with open("/home/psobecki/ASL_Citizen/asllex2_signdata.csv", encoding="latin-1") as fh:
    for d in csv.DictReader(fh):
        g = (d.get("EntryID") or "").strip().upper()
        if g not in vocab:
            continue
        sig = tuple((d.get(c) or "").strip() for c in FEATS)
        if all(sig):
            form[g] = sig

# ---- raw Logos features -> per-video, normalized ----
recs = load_metadata(Path("/home/psobecki/ASL_Citizen/splits"), "train")
feats, glosses = load_features(recs, RAW, None, desc="raw", workers=32,
                               min_coverage=0, allow_partial=True)
feats = F.normalize(torch.from_numpy(feats).float(), dim=-1).numpy()
glosses = np.array(glosses)

# ===================== Panel 1: form-neighbor groups =====================
GROUPS = {
    "forehead / index-1 / still": ["HOPE", "CRAZY", "EXPECT", "FAINT", "AGREEMENT", "BRAINSTORM"],
    "neutral / two-S / curved":   ["CAR", "MAKE", "RESCUE", "SAFE", "FIGHT", "BREAK"],
    "neutral / flat-B / straight": ["SHELF", "ROOF", "PARADE", "MARCH", "EQUAL", "FAIR"],
    "neutral / O-hand / straight": ["FIX", "GET", "MORE", "OBVIOUS", "OFFICE", "CLEAR"],
    "neutral / open-B / curved":  ["BOOK", "BOAT", "DIE", "MELT", "PEACE", "DIVIDE"],
}
sel = [g for gs in GROUPS.values() for g in gs if (glosses == g).any()]
rng = np.random.default_rng(0)
X, lab = [], []
for g in sel:
    idx = np.where(glosses == g)[0]
    take = rng.choice(idx, size=min(14, len(idx)), replace=False)
    X.append(feats[take]); lab += [g] * len(take)
X = np.concatenate(X); lab = np.array(lab)
proj = TSNE(n_components=2, perplexity=20, init="pca", random_state=0).fit_transform(X)
cen = {g: proj[lab == g].mean(0) for g in sel}
fig, ax = plt.subplots(figsize=(8.5, 7.5))
cmap = plt.get_cmap("tab10")
for i, (gname, gs) in enumerate(GROUPS.items()):
    c = cmap(i % 10)
    for g in gs:
        if g not in cen:
            continue
        m = lab == g
        ax.scatter(proj[m, 0], proj[m, 1], color=c, s=13, alpha=0.22)
        ax.text(cen[g][0], cen[g][1], g, fontsize=7, color=c, ha="center",
                va="center", fontweight="bold")
    ax.scatter([], [], color=c, label=gname)
ax.legend(fontsize=8, title="shared form signature", loc="best")
ax.set_title("Raw Logos: phonological form-neighbour groups\n"
             "(each colour = glosses sharing Movement+Location+Handshape+SignType, "
             "different meanings)", fontsize=9)
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(f"{D}/form_neighbor_groups.png", dpi=150)
plt.close(fig); print("saved form_neighbor_groups.png")

# ================= Panel 2: violins by #shared features ==================
gl = [g for g in form if (glosses == g).any()]
C = np.stack([feats[glosses == g].mean(0) for g in gl])
C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
S = C @ C.T
sig = np.array([form[g] for g in gl])
N = len(gl)
shared = np.zeros((N, N), dtype=int)
for k in range(5):
    col = sig[:, k]
    shared += (col[:, None] == col[None, :]).astype(int)
iu = np.triu_indices(N, k=1)
sh, co = shared[iu], S[iu]
data = [co[sh == k] for k in range(6)]
means = [float(np.mean(b)) if len(b) else float("nan") for b in data]
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.violinplot(data, positions=range(6), showmeans=True, showextrema=False, widths=0.85)
ax.plot(range(6), means, "k--o", lw=1, ms=4, label="mean")
ax.set_xticks(range(6))
ax.set_xticklabels([f"{k}\n(n={len(data[k]):,})" for k in range(6)])
ax.set_xlabel("# shared ASL-LEX form features (Movement, MajLoc, MinLoc, Handshape, SignType)")
ax.set_ylabel("embedding cosine (gloss centroids, raw Logos)")
ax.set_title("Form overlap predicts embedding similarity\n"
             "mean cosine by shared-feature count: "
             + "  ".join(f"{k}:{means[k]:.3f}" for k in range(6)), fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{D}/form_overlap_violins.png", dpi=150)
plt.close(fig)
print("saved form_overlap_violins.png  means", [round(m, 3) for m in means])
print("DONE_KQ_FORM")
