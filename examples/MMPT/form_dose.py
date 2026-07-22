"""Form-overlap dose-response across feature spaces.
For each --space NAME=DIR: compute embedding cosine between gloss centroids,
bucketed by # shared ASL-LEX form features (0..5); cache buckets to npz; render
4 plot styles. Reruns are free from the npz (see form_dose_plot.py)."""
import argparse, os, csv, glob
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eval_asl_citizen_retrieval import load_metadata, load_features

FEATS = ["Movement.2.0", "MajorLocation.2.0", "MinorLocation.2.0",
         "Handshape.2.0", "SignType.2.0"]
OUT = "runs/king_queen_analogy/form_dose"
os.makedirs(OUT, exist_ok=True)


def parse_space(s):
    n, d = s.split("=", 1)
    return n, Path(d)


ap = argparse.ArgumentParser()
ap.add_argument("--space", action="append", type=parse_space, required=True)
ap.add_argument("--styles", action="store_true", help="render all 4 style variants")
args = ap.parse_args()

# vocab + form signatures
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
        if g in vocab:
            sig = tuple((d.get(c) or "").strip() for c in FEATS)
            if all(sig):
                form[g] = sig
recs = load_metadata(Path("/home/psobecki/ASL_Citizen/splits"), "train")


def compute(name, fdir):
    feats, glosses = load_features(recs, fdir, None, desc=name, workers=32,
                                   min_coverage=0, allow_partial=True)
    feats = F.normalize(torch.from_numpy(feats).float(), dim=-1).numpy()
    glosses = np.array(glosses)
    gl = [g for g in form if (glosses == g).any()]
    C = np.stack([feats[glosses == g].mean(0) for g in gl])
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    S = C @ C.T
    sig = np.array([form[g] for g in gl])
    N = len(gl)
    shared = np.zeros((N, N), int)
    for k in range(5):
        shared += (sig[:, k][:, None] == sig[:, k][None, :]).astype(int)
    iu = np.triu_indices(N, 1)
    sh, co = shared[iu], S[iu]
    data = [co[sh == k].astype(np.float32) for k in range(6)]
    np.savez(f"{OUT}/buckets_{name}.npz", **{f"b{k}": data[k] for k in range(6)})
    print(name, "n_gloss", N, "means", [round(float(np.mean(b)), 3) for b in data])
    return data


def render_styles(name, data):
    means = [float(np.mean(b)) for b in data]
    xt = [f"{k}\n(n={len(data[k]):,})" for k in range(6)]

    def base(ax):
        ax.axhline(0, color="grey", lw=0.6, ls=":")
        ax.set_xticks(range(6)); ax.set_xticklabels(xt)
        ax.set_ylabel("embedding cosine (gloss centroids)")
        ax.set_xlabel("# shared ASL-LEX form features")

    # v1 violin + extrema + medians
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.violinplot(data, positions=range(6), showextrema=True, showmedians=True, widths=0.85)
    ax.plot(range(6), means, "k--o", ms=4, lw=1, label="mean")
    base(ax); ax.legend(fontsize=8)
    ax.set_title(f"{name}: violin + extrema/median", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}_v1_violin_extrema.png", dpi=150); plt.close(fig)

    # v2 boxplot (no outliers)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.boxplot(data, positions=range(6), widths=0.6, showfliers=False,
               medianprops=dict(color="C1", lw=1.5))
    ax.plot(range(6), means, "k--o", ms=4, lw=1, label="mean")
    base(ax); ax.legend(fontsize=8)
    ax.set_title(f"{name}: boxplot (whiskers, no fliers)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}_v2_box.png", dpi=150); plt.close(fig)

    # v3 box inside violin
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    vp = ax.violinplot(data, positions=range(6), showextrema=False, widths=0.9)
    for b in vp["bodies"]:
        b.set_alpha(0.30)
    ax.boxplot(data, positions=range(6), widths=0.14, showfliers=False,
               medianprops=dict(color="C1", lw=1.5))
    ax.plot(range(6), means, "k--o", ms=3, lw=1)
    base(ax)
    ax.set_title(f"{name}: violin + box", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}_v3_boxviolin.png", dpi=150); plt.close(fig)

    # v4 bucket vs 0-shared null overlay
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for k in range(6):
        n0 = ax.violinplot([data[0]], positions=[k], showextrema=False, widths=0.85)
        for b in n0["bodies"]:
            b.set_facecolor("grey"); b.set_alpha(0.15)
        vp = ax.violinplot([data[k]], positions=[k], showextrema=False, widths=0.6)
        for b in vp["bodies"]:
            b.set_alpha(0.75)
    ax.plot(range(6), means, "k--o", ms=4, lw=1)
    base(ax)
    ax.set_title(f"{name}: bucket (colour) vs 0-shared null (grey)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}_v4_nullref.png", dpi=150); plt.close(fig)
    print("rendered 4 styles for", name)


allmeans = {}
for name, fdir in args.space:
    data = compute(name, fdir)
    allmeans[name] = [float(np.mean(b)) for b in data]
    if args.styles:
        render_styles(name, data)

if len(allmeans) > 1:
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, ms in allmeans.items():
        ax.plot(range(6), ms, "-o", label=name)
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("# shared ASL-LEX form features")
    ax.set_ylabel("mean embedding cosine")
    ax.legend(fontsize=8)
    ax.set_title("Form-overlap dose-response across spaces\n"
                 "(flatter slope = intervention reduced form-structure)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{OUT}/COMPARE_means.png", dpi=150); plt.close(fig)
    print("saved COMPARE_means.png")
print("DONE_FORM_DOSE")
