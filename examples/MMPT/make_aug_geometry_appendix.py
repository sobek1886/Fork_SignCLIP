#!/usr/bin/env python3
"""Generate the thesis-appendix LaTeX tables for the augmentation-geometry
analysis: the FULL grid (all feature spaces x edit types x both samples)
with uncertainty, from the JSONs written by analyze_augmentation_geometry.py.

The output is a self-contained appendix block (\\appendix + section + two
tables) meant to be pasted verbatim into thesis/ai-draft.tex. Never edit the
pasted tables by hand — rerun this script after any geometry rerun:

  python make_aug_geometry_appendix.py \
      --train_json .../aug_geometry_train.json \
      --test_json  .../aug_geometry_test.json \
      --out_tex    aug_geometry_appendix.tex
"""
import argparse
import json

# (json key, display name) — display order fixed to match the main-text tables
SPACES = [("frozen", "frozen"), ("run2", "CE"), ("run3", "aug-as-data"),
          ("run1", "consistency"), ("sdda", "SDDA")]
VARIANTS = [("glasses", "glasses"), ("shirt_1", "shirt"),
            ("skin_mst_diffusion", "skin tone"), ("signer_swap", "signer swap")]


def row(space, vkey, vname):
    b1 = space["block1_proximity"][vkey]
    b2 = space["block2_gap"]["variants"][vkey]
    b3 = space["block3_preservation"][vkey]
    b4 = space["block4_subspaces"]["variants"][vkey]
    nd = b1["normalized_displacement"]["median"]
    lo, hi = b1["normalized_displacement_median_ci"]
    sf, sd = b2["separability_full"], b2["separability_deshift"]
    return (f"\\;\\;{vname} & {nd:.2f} [{lo:.2f}, {hi:.2f}]"
            f" & {sf['auc_mean']:.3f}\\,$\\pm$\\,{sf['auc_std']:.3f}"
            f" & {sd['auc_mean']:.2f}\\,$\\pm$\\,{sd['auc_std']:.2f}"
            f" & {b3['gloss_retention_at1']:.2f}"
            f" & {b3['signer_flip_rate']:.2f}"
            f" & {b4['energy_in_signer_subspace']['median']:.2f} \\\\")


def table(data, sample_line, label):
    lines = [
        "\\begin{table}[H]", "\\centering", "\\footnotesize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lcccccc}", "\\toprule",
        "\\textbf{Edit type} & \\textbf{n.disp [95\\% CI]} & \\textbf{AUC} &"
        " \\textbf{AUC$_{\\text{de-shift}}$} & \\textbf{gloss@1} &"
        " \\textbf{flip} & \\textbf{$E_{\\text{signer}}$} \\\\",
    ]
    for skey, sname in SPACES:
        sp = data[skey]
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{sname}}}}} \\\\")
        for vkey, vname in VARIANTS:
            lines.append(row(sp, vkey, vname))
        pooled = sp["block2_gap"]["separability_all_variants"]
        lines.append(f"\\;\\;all edits (pooled) & ---"
                     f" & {pooled['auc_mean']:.3f}\\,$\\pm$\\,{pooled['auc_std']:.3f}"
                     f" & --- & --- & --- & --- \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        f"\\caption{{Full augmentation-geometry grid, {sample_line} n.disp ="
        " median normalised displacement with the $95\\%$ bootstrap"
        " confidence interval of the median; AUC and AUC$_{\\text{de-shift}}$"
        " = mean $\\pm$ sd over the five grouped cross-validation folds;"
        " gloss@1, flip and $E_{\\text{signer}}$ (medians) as defined in"
        " \\S\\ref{sec:results-aug-geometry}. The pooled row is the"
        " all-edit-types AUC quoted in Table~\\ref{tab:aug_geometry_spaces}.}",
        f"\\label{{{label}}}", "\\end{table}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--out_tex", required=True)
    args = ap.parse_args()

    train = json.load(open(args.train_json))
    test = json.load(open(args.test_json))

    parts = [
        "\\appendix",
        "",
        "\\section{Full augmentation-geometry grid}",
        "\\label{app:aug_geometry_grid}",
        "Tables~\\ref{tab:aug_geometry_grid_train}"
        " and~\\ref{tab:aug_geometry_grid_test} report every cell of the"
        " \\S\\ref{sec:results-aug-geometry} analysis --- all five feature"
        " spaces $\\times$ four edit types $\\times$ both evaluation samples"
        " --- with uncertainty estimates for the quantities the main-text"
        " tables report as point values.",
        "",
        table(train,
              "training pairs ($6{,}000$ per edit type, selection-biased).",
              "tab:aug_geometry_grid_train"),
        "",
        table(test,
              "held-out test pairs ($500$ per edit type, signer-disjoint).",
              "tab:aug_geometry_grid_test"),
        "",
    ]
    with open(args.out_tex, "w") as f:
        f.write("\n".join(parts))
    print(f"Wrote {args.out_tex}")


if __name__ == "__main__":
    main()
