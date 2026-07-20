#!/usr/bin/env python3
"""One-command analysis of the NGT Flux-vs-Unreal ladder + signer-count ablation.

Run AFTER the ngt_k_runs.job arms finish. Consumes the per-run
eval_results.json files ("style": "front", produced by eval_ngt_retrieval.py
on take-list manifests) and emits everything the thesis sections
sec:ngt-single-view / sec:ngt-signer-count need:

  1. Per-arm summary  — mean±sd over seeds for R@1/R@5/R@10/MRR, both
     front-view directions (avg-view rows too where present), plus the
     length-only shortcut floor.
  2. Sanity gates     — identical sentence counts within each tier; length
     baseline far below every model arm (margin reported, violations flagged).
  3. Paired contrasts — exact McNemar + bootstrap CI per seed pair for the
     pre-registered comparisons (E3|E4, E5|E6, E1|E5, E1|E6, E5s|E6s, and
     real_R|unreal_full_R at each R), reusing compare_ngt_arms.py.
  4. Headline         — fraction of the parametric (Unreal) gain retained by
     the diffusion (Flux) route: (E6-E1)/(E5-E1), per seed and per direction.
  5. Dose-response    — Δ(R) = unreal_full_R − real_R for R∈{1..4} with seed
     spread, plus the sentence-coverage covariate from the manifest sidecars.
  6. Outputs          — <out_dir>/ngt_summary.{json,md}, ngt_tables.tex
     (booktabs rows for the two thesis tables), and optional --figs PNGs
     (ladder bars, dose-response curve).

Arms are looked up by the canonical names from the 2026-07-16 handoff;
missing arms are reported and skipped, so the script is safe to run while
only part of the grid has landed.

Usage (login node, CPU, instant):
    python summarize_ngt_experiments.py \
        --base_dir /scratch-shared/psobecki/runs/retri_ngt \
        --coverage_dir /home/psobecki \
        --out_dir runs/ngt_summary --figs
"""

import argparse
import json
from pathlib import Path

import numpy as np

from compare_ngt_arms import compare_direction, load_runs

METRICS = ["R@1", "R@5", "R@10", "MRR"]
DIRECTIONS = [("b2s_front", "B→S front"), ("s2b_front", "S→B front")]

# Canonical arm registry: name -> (tier, short label used in tables/figures)
ARMS = {
    "ngt_sv_real":            ("ladder", "E1 real (K=1)"),
    "ngt_sv_real_multiview":  ("ladder", "E2 real multiview"),
    "ngt_sv_unreal":          ("ladder", "E3 +Unreal (K=3)"),
    "ngt_sv_flux":            ("ladder", "E4 +Flux (K=3)"),
    "ngt_sv_unreal_full":     ("ladder", "E5 +Unreal full (K=7)"),
    "ngt_sv_flux_full":       ("ladder", "E6 +Flux full"),
    "ngt_sv_flux_glasses":    ("ladder", "E7 +Flux glasses"),
    "ngt_sv_combined":        ("ladder", "E8 combined"),
    "ngt_sv_unreal_only":     ("ladder", "E9 Unreal only"),
    "ngt_sv_flux_only":       ("ladder", "E10 Flux only"),
    "ngt_subset_real":        ("subset", "E1s real"),
    "ngt_subset_unreal_full": ("subset", "E5s +Unreal full"),
    "ngt_subset_flux_full":   ("subset", "E6s +Flux full"),
}
for _r in range(1, 5):
    ARMS[f"ngt_sv_real_R{_r}"] = ("signer_count", f"R{_r} real")
    ARMS[f"ngt_sv_unreal_full_R{_r}"] = ("signer_count", f"R{_r} +Unreal full")

# Pre-registered paired contrasts (arm_a, arm_b, label).
CONTRASTS = [
    ("ngt_sv_unreal", "ngt_sv_flux", "E3 vs E4 (matched K=3)"),
    ("ngt_sv_unreal_full", "ngt_sv_flux_full", "E5 vs E6 (full strength)"),
    ("ngt_sv_real", "ngt_sv_unreal_full", "E1 vs E5 (Unreal gain)"),
    ("ngt_sv_real", "ngt_sv_flux_full", "E1 vs E6 (Flux gain)"),
    ("ngt_subset_unreal_full", "ngt_subset_flux_full", "E5s vs E6s (subset)"),
] + [(f"ngt_sv_real_R{r}", f"ngt_sv_unreal_full_R{r}", f"Δ(R={r})")
     for r in range(1, 5)]


def arm_summary(base_dir, name, k):
    """Mean±sd over seeds for every metric group present, or None if absent."""
    rows, n_sentences, length_baseline = [], None, None
    for i in range(1, k + 1):
        p = Path(base_dir) / f"{name}_run{i}" / "eval_results.json"
        if not p.exists():
            continue
        with open(p) as f:
            r = json.load(f)
        if r.get("style") != "front":
            print(f"  NOTE {p} is legacy-style; skipped (front-style only).")
            continue
        rows.append(r)
        n_sentences = r["n_sentences"]
        length_baseline = r.get("length_baseline")
    if not rows:
        return None
    out = {"n_seeds": len(rows), "n_sentences": n_sentences, "groups": {}}
    for key in ("b2s_front", "s2b_front", "b2s_avg", "s2b_avg"):
        if key not in rows[0]:
            continue
        out["groups"][key] = {
            m: {"mean": float(np.mean([r[key][m] for r in rows])),
                "std": float(np.std([r[key][m] for r in rows], ddof=1))
                if len(rows) > 1 else 0.0}
            for m in METRICS
        }
    if length_baseline:
        out["length_baseline"] = {
            k2: {m: length_baseline[k2][m] for m in METRICS}
            for k2 in ("len_b2s", "len_s2b") if k2 in length_baseline
        }
    return out


def gate_checks(summaries):
    """Sentence-count identity per tier + length-floor margins."""
    gates = {"sentence_counts": {}, "length_floor": {}, "violations": []}
    by_tier = {}
    for name, s in summaries.items():
        by_tier.setdefault(ARMS[name][0], {})[name] = s["n_sentences"]
    for tier, counts in by_tier.items():
        gates["sentence_counts"][tier] = counts
        if tier != "signer_count" and len(set(counts.values())) > 1:
            gates["violations"].append(
                f"sentence counts differ within tier '{tier}': {counts}")
    for name, s in summaries.items():
        lb = s.get("length_baseline")
        if not lb:
            continue
        for dkey, lkey in (("b2s_front", "len_b2s"), ("s2b_front", "len_s2b")):
            model_r1 = s["groups"][dkey]["R@1"]["mean"]
            floor_r1 = lb[lkey]["R@1"]
            margin = model_r1 - floor_r1
            gates["length_floor"][f"{name}:{dkey}"] = {
                "model_R@1": model_r1, "floor_R@1": floor_r1, "margin": margin}
            if margin < 10.0:
                gates["violations"].append(
                    f"{name} {dkey}: R@1 {model_r1:.1f} within 10pts of the "
                    f"length-only floor {floor_r1:.1f} — the length shortcut "
                    "may be doing the work; protocol caveat required.")
    return gates


def paired_contrasts(base_dir, present, k, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for a, b, label in CONTRASTS:
        if a not in present or b not in present:
            out[label] = {"status": "SKIPPED (arm missing)", "a": a, "b": b}
            continue
        runs_a, runs_b = load_runs(base_dir, a, k), load_runs(base_dir, b, k)
        seeds = sorted(set(runs_a) & set(runs_b))
        entry = {"a": a, "b": b, "directions": {}}
        for dkey, _ in DIRECTIONS:
            per_seed = [compare_direction(runs_a[i], runs_b[i],
                                          f"{dkey}_rank", n_bootstrap, rng)
                        for i in seeds]
            deltas = [r["delta_r1_b_minus_a"] for r in per_seed]
            entry["directions"][dkey] = {
                "mean_delta_r1": float(np.mean(deltas)),
                "std_delta_r1": float(np.std(deltas, ddof=1))
                if len(deltas) > 1 else 0.0,
                "seed_mcnemar_ps": [r["mcnemar_p_two_sided"] for r in per_seed],
                "n_seeds_significant_005": int(sum(
                    r["mcnemar_p_two_sided"] < 0.05 for r in per_seed)),
            }
        out[label] = entry
    return out


def headline_fraction(summaries, base_dir, k):
    """(E6-E1)/(E5-E1) per seed-matched triple and per direction."""
    names = ("ngt_sv_real", "ngt_sv_unreal_full", "ngt_sv_flux_full")
    if not all(n in summaries for n in names):
        return {"status": "SKIPPED (needs E1, E5, E6)"}
    runs = {n: load_runs(base_dir, n, k) for n in names}
    seeds = sorted(set.intersection(*[set(r) for r in runs.values()]))
    out = {}
    for dkey, _ in DIRECTIONS:
        fracs = []
        for i in seeds:
            r1 = {}
            for n in names:
                ranks = [v[f"{dkey}_rank"] for v in runs[n][i].values()]
                r1[n] = float(np.mean([rk == 0 for rk in ranks]))
            unreal_gain = r1[names[1]] - r1[names[0]]
            flux_gain = r1[names[2]] - r1[names[0]]
            fracs.append(flux_gain / unreal_gain if abs(unreal_gain) > 1e-9
                         else float("nan"))
        fracs = [f for f in fracs if not np.isnan(f)]
        out[dkey] = {
            "per_seed": fracs,
            "mean": float(np.mean(fracs)) if fracs else None,
            "std": float(np.std(fracs, ddof=1)) if len(fracs) > 1 else 0.0,
        }
    return out


def dose_response(summaries, coverage_dir):
    """Δ(R) with seed spread + coverage covariate per condition."""
    out = {}
    for r in range(1, 5):
        a, b = f"ngt_sv_real_R{r}", f"ngt_sv_unreal_full_R{r}"
        if a not in summaries or b not in summaries:
            out[f"R{r}"] = {"status": "SKIPPED (arm missing)"}
            continue
        entry = {}
        for dkey, _ in DIRECTIONS:
            ra, rb = summaries[a]["groups"][dkey], summaries[b]["groups"][dkey]
            entry[dkey] = {
                "real_R@1": ra["R@1"], "unreal_R@1": rb["R@1"],
                "delta_R@1_mean": rb["R@1"]["mean"] - ra["R@1"]["mean"],
            }
        if coverage_dir:
            cov = Path(coverage_dir) / f"ngt_sv_R{r}_coverage.json"
            if cov.exists():
                with open(cov) as f:
                    entry["coverage"] = json.load(f)
        out[f"R{r}"] = entry
    return out


def latex_rows(summaries):
    """Booktabs rows matching the thesis table style (both front directions)."""
    lines = ["% Auto-generated by summarize_ngt_experiments.py — paste-ready",
             "% columns: Arm & R@1 & R@5 & R@10 & MRR (mean$\\pm$sd over seeds)"]
    for dkey, dlabel in DIRECTIONS:
        lines.append(f"% --- direction: {dlabel} ---")
        for name, s in summaries.items():
            g = s["groups"].get(dkey)
            if not g:
                continue
            cells = " & ".join(
                f"{g[m]['mean']:.1f}$\\pm${g[m]['std']:.1f}" for m in METRICS)
            label = ARMS[name][1].replace("+", "$+$")
            lines.append(f"{label} & {cells} \\\\  % {name}")
        lb = next((s["length_baseline"] for s in summaries.values()
                   if s.get("length_baseline")), None)
        if lb:
            lkey = "len_b2s" if dkey == "b2s_front" else "len_s2b"
            cells = " & ".join(f"{lb[lkey][m]:.1f}" for m in METRICS)
            lines.append(f"\\emph{{length-only floor}} & {cells} \\\\")
    return "\n".join(lines) + "\n"


def markdown_report(summaries, gates, contrasts, headline, dose):
    md = ["# NGT experiment summary\n"]
    for tier in ("ladder", "subset", "signer_count"):
        rows = {n: s for n, s in summaries.items() if ARMS[n][0] == tier}
        if not rows:
            continue
        md.append(f"## {tier}\n")
        md.append("| Arm | dir | R@1 | R@5 | R@10 | MRR | seeds |")
        md.append("|---|---|---|---|---|---|---|")
        for name, s in rows.items():
            for dkey, dlabel in DIRECTIONS:
                g = s["groups"].get(dkey)
                if not g:
                    continue
                md.append(
                    f"| {ARMS[name][1]} | {dlabel} | "
                    + " | ".join(f"{g[m]['mean']:.1f}±{g[m]['std']:.1f}"
                                 for m in METRICS)
                    + f" | {s['n_seeds']} |")
        md.append("")
    md.append("## Gates\n")
    md.append("VIOLATIONS:\n" + ("\n".join(f"- {v}" for v in gates["violations"])
                                 if gates["violations"] else "- none") + "\n")
    md.append("## Paired contrasts (McNemar per seed pair)\n")
    for label, c in contrasts.items():
        if "status" in c:
            md.append(f"- **{label}**: {c['status']}")
            continue
        for dkey, dlabel in DIRECTIONS:
            d = c["directions"][dkey]
            md.append(f"- **{label}** ({dlabel}): ΔR@1 = "
                      f"{d['mean_delta_r1']:+.3f} ± {d['std_delta_r1']:.3f}, "
                      f"{d['n_seeds_significant_005']}/"
                      f"{len(d['seed_mcnemar_ps'])} seeds p<0.05")
    md.append("\n## Headline: fraction of Unreal gain retained by Flux\n")
    if "status" in headline:
        md.append(headline["status"])
    else:
        for dkey, dlabel in DIRECTIONS:
            h = headline[dkey]
            md.append(f"- {dlabel}: (E6−E1)/(E5−E1) = "
                      f"{h['mean']:.2f} ± {h['std']:.2f} "
                      f"(per seed: {['%.2f' % f for f in h['per_seed']]})")
    md.append("\n## Dose-response Δ(R)\n")
    for rlabel, entry in dose.items():
        if "status" in entry:
            md.append(f"- {rlabel}: {entry['status']}")
            continue
        parts = []
        for dkey, dlabel in DIRECTIONS:
            e = entry[dkey]
            parts.append(f"{dlabel}: Δ={e['delta_R@1_mean']:+.1f} "
                         f"(real {e['real_R@1']['mean']:.1f} → "
                         f"unreal {e['unreal_R@1']['mean']:.1f})")
        cov = entry.get("coverage", {})
        covs = (f" | coverage: {cov.get('total_bushuis_matched', '?')} matched"
                if cov else "")
        md.append(f"- {rlabel}: " + "; ".join(parts) + covs)
    return "\n".join(md) + "\n"


def make_figs(summaries, dose, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ladder = {n: s for n, s in summaries.items() if ARMS[n][0] == "ladder"}
    if ladder:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, (dkey, dlabel) in zip(axes, DIRECTIONS):
            names = [n for n in ladder if dkey in ladder[n]["groups"]]
            means = [ladder[n]["groups"][dkey]["R@1"]["mean"] for n in names]
            stds = [ladder[n]["groups"][dkey]["R@1"]["std"] for n in names]
            ax.bar(range(len(names)), means, yerr=stds, capsize=3)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([ARMS[n][1] for n in names],
                               rotation=45, ha="right", fontsize=8)
            ax.set_title(dlabel)
            ax.set_ylabel("R@1 (%)")
        fig.suptitle("NGT single-view ladder")
        fig.tight_layout()
        fig.savefig(out_dir / "ladder_r1.png", dpi=200)
        plt.close(fig)

    rs = [r for r in range(1, 5) if "status" not in dose.get(f"R{r}", {"status": 1})]
    if rs:
        fig, ax = plt.subplots(figsize=(6, 4))
        for dkey, dlabel in DIRECTIONS:
            ax.plot(rs, [dose[f"R{r}"][dkey]["delta_R@1_mean"] for r in rs],
                    marker="o", label=dlabel)
        ax.set_xlabel("real signers in training (R)")
        ax.set_ylabel("Δ R@1 (Unreal-full − real), pts")
        ax.set_xticks(rs)
        ax.axhline(0, color="grey", lw=0.5)
        ax.legend()
        ax.set_title("Synthetic gain vs real signer diversity")
        fig.tight_layout()
        fig.savefig(out_dir / "dose_response.png", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--coverage_dir", default=None,
                    help="dir holding ngt_sv_R*_coverage.json sidecars")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, default=Path("runs/ngt_summary"))
    ap.add_argument("--figs", action="store_true")
    args = ap.parse_args()

    summaries = {}
    for name in ARMS:
        s = arm_summary(args.base_dir, name, args.k)
        if s is None:
            print(f"MISSING arm: {name}")
        else:
            summaries[name] = s
            print(f"loaded {name}: {s['n_seeds']} seeds, "
                  f"{s['n_sentences']} sentences")
    if not summaries:
        raise SystemExit("No arms found — nothing to summarize.")

    gates = gate_checks(summaries)
    contrasts = paired_contrasts(args.base_dir, summaries, args.k,
                                 args.n_bootstrap, args.seed)
    headline = headline_fraction(summaries, args.base_dir, args.k)
    dose = dose_response(summaries, args.coverage_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "ngt_summary.json", "w") as f:
        json.dump({"summaries": summaries, "gates": gates,
                   "contrasts": contrasts, "headline": headline,
                   "dose_response": dose}, f, indent=2)
    md = markdown_report(summaries, gates, contrasts, headline, dose)
    (args.out_dir / "ngt_summary.md").write_text(md)
    (args.out_dir / "ngt_tables.tex").write_text(latex_rows(summaries))
    if args.figs:
        make_figs(summaries, dose, args.out_dir)
    print(md)
    print(f"Wrote {args.out_dir}/ngt_summary.{{json,md}} + ngt_tables.tex"
          + (" + figs" if args.figs else ""))


if __name__ == "__main__":
    main()
