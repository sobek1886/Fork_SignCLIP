# Thesis end-game plan (as of 2026-06-29)

Deadlines (confirmed): **company-extension decision 2026-06-30 (TOMORROW)** · draft to supervisor **2026-07-08** (9 days) · final/university **2026-07-17** (18 days), university extension possible based on the draft. Writing not started.

## Verdict: deliverable — *if you switch to writing mode now*

You are **not** short on results. You have more than enough for a master's thesis. The
risk to your thesis is **unwritten pages and a missing narrative**, not missing
experiments. Every remaining experiment on your list is optional polish; several you
already crossed out yourself. **The single highest-leverage decision is: freeze the
experiments and start assembling the document today.** Continuing to chase experiments
(especially diffusion 3-view, which you noted triples generation time) is the one thing
that can actually sink this.

---

## 1. The gap between the proposal and what you did

This is the source of "I don't know my story anymore." The proposal and the results are
about *different problems*:

| | Proposal (months ago) | What you actually have |
|---|---|---|
| Task | **Continuous** SLR (CSLR) | **Isolated** sign recognition (ISLR) |
| Metric | WER | Retrieval: R@1, DCG, MRR |
| Dataset | **Isharah** | ASL-Citizen, WLASL(100), NGT |
| Synthetic source | Unreal/SMPL-X only | Unreal (NGT) **+** Flux diffusion (WLASL100) **+** 2D appearance edits (ASL-Citizen) |
| Model | SignCLIP | SignCLIP-on-Logos **and** direct Logos backbone FT |

Confirmed: **zero CSLR / Isharah / WER experiments exist** anywhere in the repo or on
disk. So the proposal's RQs (all phrased around WER on Isharah) are **unanswerable as
written**. This is a normal thesis pivot — but it must be (a) sanctioned by your
supervisor and (b) reframed honestly in the introduction. Do not try to add Isharah/CSLR
now; it is not feasible and would torpedo the thesis.

---

## 2. The story (this is the thesis)

**One sentence:** *Synthetic, appearance-varied training data makes isolated-sign
recognition signer-invariant precisely when training and test differ in appearance —
Unreal-Engine view augmentation delivers large cross-signer gains (NGT R@1 0.24 → 0.61),
while the same idea is neutral-to-harmful in-domain (ASL-Citizen), because in-domain
models are already appearance-robust.*

That is a genuinely good thesis: a clear hypothesis grounded in the appearance-bias
literature, a **positive headline result** (Unreal/NGT), a **rigorous negative result with
a mechanistic explanation** (in-domain ASL-Citizen), and a **unifying insight**
(in-domain vs cross-domain is the deciding factor).

### The arc (each stage maps to results you already have)
1. **Diagnosis — sign representations carry signer/appearance identity.**
   Exp 1–3 (counterfactual + cross-gloss signer probe) + L2 signer-bias probe (Table 3).
2. **In-domain, there's little to fix.** ASL-Citizen is signer-disjoint but same webcam
   domain. Appearance augmentation + invariance objectives buy ~nothing; several hurt.
   Plain CE fine-tuning at moderate depth is the only lever. Shown twice, independently:
   SignCLIP-on-Logos (Tables 1–2) and direct backbone FT (Tables 6–7). *Rigorous null.*
3. **The fair test of appearance invariance is cross-domain / cross-signer.** (The bridge.)
4. **Cross-domain, it works — Unreal is the win.** NGT: 3 real + 6 Unreal-rendered
   appearance-varied views (K=9) lifts retrieval to a held-out real signer (Bushuis)
   from R@1 0.24 → 0.61, MRR 0.36 → 0.71 (Table 5). *Positive headline.*
5. **(Secondary) Diffusion (Flux) as an alternative synthetic source** — WLASL100,
   modest/flat (Table 4). Shows the idea generalizes beyond Unreal; keep it short.

### Reframed research questions (3 of your 4 survive)
- **Central:** To what extent can synthetic appearance-varied training data (Unreal
  renders / diffusion edits) reduce appearance/signer bias and improve **signer-independent
  isolated** sign recognition?
- **RQ1:** Does synthetic appearance augmentation improve signer-independent ISLR — and
  does the answer depend on whether train/test share an appearance domain (ASL-Citizen
  in-domain vs NGT cross-domain)? *(old RQ1, retargeted from WER→retrieval)*
- **RQ2:** Does an explicit invariance / contrastive objective on counterfactual pairs add
  value beyond using the synthetic data as plain augmentation? *(old RQ2)*
- **RQ3:** How does the intervention reshape the embedding space w.r.t. separating motion
  from appearance? *(old RQ3/3a/3b — survives almost verbatim; answered by the L2 probes,
  cross-gloss signer retrieval, and Approach A/B diagnostics)*
- **RQ4 (optional / future work):** RGB-vs-pose gap — partially covered by the L2 probe
  (MediaPipe vs Logos vs I3D) and the I3D experiments if run. Downgrade or defer.

---

## 3. What you already have (the reassurance)

**Results — done:**
- Diagnostics: Exp 1 (WLASL counterfactual), Exp 2/3 (cross-gloss signer probe,
  SignCLIP + MediaPipe), L2 signer-bias probe (Table 3).
- ASL-Citizen SignCLIP appearance-aug (Tables 1–2).
- ASL-Citizen backbone FT — full ablation: CE / consistency / clscontrast / SDDA /
  glosscon × unfreeze depth (Tables 6–7).
- NGT Unreal cross-signer (Table 5) — the headline.
- WLASL100 Flux diffusion (Table 4).

**Writing already in fragments (~60% of the raw material exists):**
- Proposal Ch.1 Introduction + Ch.2 Related Work + 36-ref bibliography (reusable).
- `backbone_ft_methods.tex` (193 lines) — a polished methods section, paste-ready.
- `LOGOS_ARCHITECTURE_AND_LOSSES.md` (379 ln) — methods source for backbone/losses.
- `EVALUATION_SETUPS.md` + `RETRIEVAL_TASKS_EXPLAINED.md` — basically your Evaluation
  Protocols section already written, including the NGT-eval critique (use in Limitations).
- `EXPERIMENTS.md`, `CONFIG.md`, `DATASET.md` — reproducibility appendix material.

**Missing (real tasks):** a LaTeX thesis skeleton (`main.tex` + chapters) and **all
figures/tables** — results live only in MLflow, nothing is exported yet.

---

## 4. Remaining-experiment triage

| Item | Call | Why |
|---|---|---|
| ASL backbone runs (clscontrast, full_ce, full_glosscon, sdda) | **DONE** | already in Tables 6–7 |
| NGT: SupCon-only vs CE+SupCon "does it make sense" | **WRITE, don't re-run** | NGT has no classifier head; SupCon-over-views is the right design — justify in Methods/Discussion (see EVALUATION_SETUPS.md) |
| NGT: 3-view framing | **WRITE** | results already "avg-pool over available views"; a framing paragraph, not an experiment |
| NGT Unreal: signer-count ablation | **OPTIONAL (high)** | only genuinely valuable add-on — dose-response of synthetic signer diversity. Run *only* if draft is on track |
| Diffusion 3-view single-signer (shirt/glasses/swap/skin/all) | **DROP** | huge generate+train cost; Table 4 already shows the diffusion alternative exists |
| Diffusion 1-view multi-signer (all variants) | **DROP** | same |
| ASL interpretability: per-signer + mechanistic probe | **OPTIONAL (high)** | strengthens RQ3; scripts exist (`eval_asl_citizen_per_signer.py`, `diagnose_appearance_invariance.py`). Run if push-button for ONE figure; else L2 probe + cross-gloss probe already cover RQ3 |
| SignCLIP control (`eval_signclip_features.job`) | **OPTIONAL (med)** | clean control; or just cite RETRIEVAL_TASKS_EXPLAINED.md |
| More SignCLIP aug cells (signer-swap only, skin only, …) | **DROP** | in-domain story is already "aug barely helps"; more cells won't change it |
| I3D augmentation comparison | **OPTIONAL (low)** | L2 table already shows I3D is most appearance-biased (Inter-L2 13.09); only if everything else done |

**Default recommendation: freeze all experiments, write.** Spend any spare compute on the
signer-count ablation or the RQ3 probe *in the background*, never blocking writing.

---

## 5. Thesis outline mapped to existing material

- **Ch.1 Introduction** — adapt proposal Ch.1; **rewrite the RQs** (§2 above); add an
  honest "scope" paragraph (ISLR/retrieval, not CSLR/WER).
- **Ch.2 Related Work** — proposal Ch.2 skeleton; lean into appearance bias (StillMix,
  Appearance-Free, Byvshev) + synthetic data (Unreal, diffusion); trim CSLR-only framing.
- **Ch.3 Methods** — 3.1 backbones/features (Logos MViTv2-S, I3D, MediaPipe; from
  LOGOS_ARCHITECTURE_AND_LOSSES.md) · 3.2 synthetic augmentation (Unreal / Flux / 2D edits)
  · 3.3 objectives (`backbone_ft_methods.tex`, paste-ready) · 3.4 eval protocols
  (EVALUATION_SETUPS.md + RETRIEVAL_TASKS_EXPLAINED.md).
- **Ch.4 Results** — 4.1 diagnostics · 4.2 in-domain ASL-Citizen (null) · 4.3 cross-domain
  NGT/Unreal (headline) · 4.4 diffusion (short).
- **Ch.5 Discussion** — in-domain vs cross-domain insight; why CE > invariance in-domain;
  why Unreal works; limitations (small NGT gallery, single held-out signer, no
  min_coverage guard — all in EVALUATION_SETUPS.md); the honest CSLR pivot.
- **Ch.6 Conclusion + Future Work** — CSLR/Isharah/WER, diffusion 3-view, signer-count.

---

## 6. The real timeline (pivot already sanctioned by supervisor)

- **2026-06-30 (TOMORROW):** last day to request the *company* extension if you want to
  keep working there in August. **Recommendation: request it.** It is pure optionality —
  writing hasn't started and you want 4 more experiments; an unused extension costs ~nothing,
  a needed-but-unavailable one is catastrophic. A university extension into August is only
  useful if you can also work at the company then, so this preserves the entire "finish in
  August" path. (Only skip it if there's a real cost/signal I can't see.)
- **2026-07-08 (draft, 9 days):** complete skeleton + your existing 7 result tables +
  placeholders for pending experiments. **Built on results you ALREADY have — does not
  depend on the 4 new experiments.**
- **2026-07-17 (final, 18 days):** university extension decided from the draft.

Key reframe: **the draft is due before the new experiments need to finish.** Write the
draft around Tables 1–7 (they already tell the whole story); list the 4 new experiments as
in-progress. Supervisor feedback then tells you which are worth finishing.

## 7. Experiment plan — writing foreground, cluster jobs background

Priority order (run async on Snellius; never block writing). Realistic capacity ≈ 2–3 of
these over 18 days; the 4th is a stretch.
1. **NGT signer-count ablation (Unreal)** — readiest (data on Snellius), strengthens the
   headline with a dose-response curve. Fire it off first, today/tomorrow.
2. **Mechanistic appearance-invariance probes (RQ3)** — analysis-only, scripts exist
   (`diagnose_appearance_invariance.py`, `eval_asl_citizen_per_signer.py`), cheap. One figure.
3. **I3D + same augmentations** — rhetorical payoff: "Logos is already appearance-robust,
   but the method *does* help the more appearance-biased I3D (Inter-L2 13.09) — the backbone
   our motivation came from." Infra mostly exists. Rescues the in-domain null.
4. **Flux-on-NGT bridging (single-view)** — most novel, least ready. See §8.

## 8. The Flux-vs-Unreal bridging experiment (single-view)

Question: *can diffusion appearance-augmentation substitute for the Unreal pipeline?* The
asymmetry you identified is the crux: **Unreal synthesizes novel appearance + novel views
from one recording; diffusion (2D edit) gives appearance only, no novel views.** So isolate
the **appearance axis on a single fixed view**:
- **B0:** single real view, no augmentation.
- **B-unreal:** single real view + Unreal appearance variants (same view + motion).
- **B-flux:** single real view + Flux appearance variants (same view).
- Eval: cross-signer retrieval to held-out signer(s).
- Claim: if **B-unreal ≈ B-flux > B0** → diffusion substitutes **for appearance**; Unreal's
  *extra* value (novel views) is exactly what the existing K=9 multi-view result buys and
  diffusion cannot match from single-view input.

Prefer the **single-view / multiple-signers** data if moving it to Snellius is cheap (gives
a variance estimate, fixes the "single held-out signer" limitation in EVALUATION_SETUPS.md).
Single-signer is acceptable for a first cut. Note: "lack of other views" is *not* a problem
here — single-view is the point. This experiment is the connective tissue that fuses the
Unreal (NGT) and diffusion (WLASL) halves into one coherent methods comparison.

## 9. Next 48 hours
1. **Tomorrow:** request the company extension (preserve August).
2. **Today/tomorrow:** fire the signer-count ablation on Snellius (background).
3. **Today:** LaTeX skeleton (reuse proposal template), reframe RQs, paste in proposal
   Ch.1–2 + `backbone_ft_methods.tex`.
4. **This week → July 8:** export Tables 1–7 from MLflow; write Methods + Results around
   existing results; draft to supervisor.
