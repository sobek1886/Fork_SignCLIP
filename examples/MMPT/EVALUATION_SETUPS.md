# Evaluation Setups in MMPT — A Complete Map

This document explains every evaluation protocol used in this repo for the
ASL-Citizen / NGT sign-recognition work, how they relate to the source papers
(Logos, SignRep), why each was chosen, and a critical assessment of whether the
NGT evaluation is principled.

It covers three distinct modelling phases that accumulated over the project:

| Phase | What is trained | Representation evaluated | Eval protocol | Status |
|-------|-----------------|--------------------------|---------------|--------|
| **A. SignCLIP-on-Logos** | A SignCLIP transformer (`MMBertForEncoder`, 12 layers) on top of **frozen** Logos features, video↔text contrastive (`MMContraLoss`) | pooled video/text embeddings out of the transformer | native CLIP t2v/v2t retrieval + downstream LogReg/KNN on exported embeddings + L2 signer probe | superseded |
| **B. Logos backbone fine-tune** | The MViTv2-S **Logos backbone itself** (no SignCLIP), CE / appearance-consistency | re-extracted **frozen** `.npy` features | SignRep-style frozen NN retrieval (Eval A) + linear probe (Eval B) | current ASL-Citizen pipeline |
| **C. NGT unreal augmentation** | A SignCLIP transformer (same arch as A) on multi-view NGT pairs, `SupConLoss` | pooled video embeddings out of the transformer | bespoke paired cross-signer cosine retrieval | the eval under scrutiny |

The crucial lineage fact: **the NGT evaluation (C) was built during the SignCLIP
phase (A) and shares its paradigm** — it scores embeddings coming *out of the
MMBert transformer* with cosine retrieval. When the ASL-Citizen work later moved
to the SignRep frozen-feature protocol (B), the NGT eval was left on the older
SignCLIP-style paradigm. That mismatch is the root of most of the concerns below.

---

## 1. What the two source papers actually evaluate

Both papers were read in full. This matters because some labels in our code
(`"SignRep Table 3"`, `"SignRep Table 4"`) and memory notes need qualifying.

### Logos (Ovodov et al., 2025) — *native protocol*
- **Task:** isolated sign recognition (ISLR) as **classification only**. No retrieval, no Recall@k, no DCG/MRR.
- **Heads:** a trained softmax classification head, in three transfer variants — (a) end-to-end fine-tune, (b) frozen encoder + trained FC head, (c) multi-dataset co-training with language-specific heads.
- **Metric:** **Top-1 accuracy only**.
- **Datasets:** WLASL, AUTSL, Slovo. **Logos never evaluates on ASL-Citizen** (it appears once, in their dataset-survey table).
- **Loss:** cross-entropy + label smoothing + auxiliary boundary regression. **No contrastive loss.**
- **Backbone:** MViTv2-S, Kinetics-400 init, 32×224×224 clips, step 2.

> Consequence: there is **no "native Logos ASL-Citizen protocol"** to copy. Any DCG/Recall number we report on ASL-Citizen is *our* protocol, not Logos's. The "Logos DCG 91.69" figure in the memory notes is from our own frozen-retrieval eval (Setup B/A), not from the Logos paper.

### SignRep (Wong et al., ICCV 2025) — *frozen-feature protocol*
- **Motivation (quoted):** *"We choose a retrieval approach over a linear evaluation protocol, as dictionary retrieval is a common and practical task... This method also offers a better evaluation of the generalization of the features to unseen datasets."* Retrieval *"directly evaluates the quality of the learned representations without any fine-tuning."*
- **Table 4** = frozen-feature **dictionary retrieval, no downstream training**: gallery = train dictionary, query = test, assignment by cosine similarity of L2-normalized features. Reports DCG / Rec@1 / Rec@5. Two SignRep rows: `avg` (mean-pool) vs `weighted` (activity-weighted pooling, eq. 11–12). **This is the protocol our Eval A mirrors.**
- **Table 3** = ASL-Citizen *recognition*. Note: for **ASL-Citizen specifically this is also retrieval-based** (cosine-NN label assignment), **not** a trained linear classifier. SignRep only trains a linear classifier for WLASL2000 / NMFs-CSL.
- **Backbone:** Hiera-B, Kinetics-MAE init, pretrained on YouTube-SL-25.

> Consequence: our **Eval B linear probe is our own addition**, not a reproduction of SignRep Table 3. The printed reference numbers (`SignRep DCG 90.84…`) come from SignRep's NN recognition, not a linear probe. The numbers are a fair reference point, but the `"SignRep Table 3 protocol"` label in `eval_asl_citizen_linear_probe.py` is slightly inaccurate and should be softened.

The **activity-weighted pooling** (SignRep eq. 11–12): each sliding-window segment `n` gets a hand-activity weight `γ_n = max(P_LH, P_RH)` (a hand is inactive if it stays below mid-stomach and does not move). The query representation is the normalized weighted mean `z = Σ γ_n z_n / Σ γ_n`, suppressing resting frames. Our `--weighted` path in `eval_asl_citizen_retrieval.py` reconstructs `γ_n` from MediaPipe pose instead of a learned head.

---

## 2. Setup A — SignCLIP-on-Logos (the earlier phase)

**Model.** `MMFusionSeparate` with a 12-layer `MMBertForEncoder` video encoder and a
`BertModel` text encoder, trained with `MMContraLoss` (video↔text InfoNCE) on top of
**frozen** pre-extracted Logos features (`vfeat_dim: 768`).
Config: `projects/retri/signclip_asl/asl_citizen_cnn_scratch_logos_nowd.yaml`.

This phase has **three** evaluation modes:

### A1. Native SignCLIP video↔text retrieval
- Driver: `mmpt_cli/predict.py <test_*.yaml>` with `predictor: RetrievalPredictor`, `metric: RWTHFSMetric` (see `test_asl_citizen_cnn_scratch_logos_nowd.yaml`).
- `RetrievalPredictor` (`mmpt/evaluators/predictor.py:113`) runs the model over the split, collects `pooled_video` and `pooled_text`, and builds a video×text similarity matrix.
- `RWTHFSMetric` (`mmpt/evaluators/metric.py:78`) reports **both directions**:
  - **t2v** and **v2t**: R@1/5/10, P@1/5/10, MedianR, MeanR.
  - Texts are **deduplicated** (`text not in texts[:idx]`), so the gallery is the set of *unique glosses* — i.e. it is effectively gloss-level retrieval against the unique-gloss text embeddings.
- This is SignCLIP's **own** protocol (inherited from the upstream RWTH/fingerspelling code), not anything from Logos or SignRep.

### A2. Downstream recognition on exported embeddings
- `export_signclip_features.py` dumps `video_embeddings.npy` + `texts.txt` per split; then:
  - `test_recognition_supervised.py` — **linear probe**: `sklearn LogisticRegression` over gloss labels, filtered to labels seen in train; reports Top-1/5/10 + MedianR.
  - `test_recognition_few_shot_knn.py` — **few-shot KNN**: 1/5/10/100-shot `KNeighborsClassifier`; Top-1/5/10.
- This is conceptually the same family as SignRep's linear-probe / few-shot recognition, implemented on the SignCLIP output embeddings.

### A3. L2 signer-bias probe (diagnostic, not a recognition metric)
- `asl_citizen_signclip_l2_probe.py` measures, per gloss with ≥2 signers, the ratio of **inter-signer** to **intra-signer** L2 distance among pooled embeddings. High ratio ⇒ embeddings still carry signer identity; ratio ≈ 1 ⇒ approaching signer-invariance.
- This is an **invariance diagnostic**, the direct conceptual ancestor of the NGT appearance-invariance question.

**Why this phase was abandoned.** The memory notes record that **raw frozen Logos
features already beat the SignCLIP-on-frozen embeddings** on ASL-Citizen retrieval.
There was no point training a contrastive head on top of features that were already
more discriminative than the head's output — so the project pivoted to fine-tuning
the backbone itself (Setup B).

---

## 3. Setup B — Logos backbone fine-tune (current ASL-Citizen pipeline)

**Model.** The MViTv2-S Logos backbone is fine-tuned directly
(`Fork_Logos/train_logos_asl_citizen.py`, standalone PyTorch, **no SignCLIP/fairseq**),
then features are re-extracted to `.npy`. Evaluation never touches the backbone — it
operates on the frozen re-extracted features, so it is fully comparable across runs.

Driver: `jobs/eval_asl_citizen_features.job` → `eval_asl_citizen_features.py`
(loads features once, runs both evals, logs into one MLflow run).

### B1. Eval A — frozen video-to-video NN retrieval (`eval_asl_citizen_retrieval.py`)
- **Bit-exact port of the official `microsoft/ASL-citizen-code` reference** (`I3D Features/test_features.py`) = the ASL-Citizen dataset's *own* benchmark, also used by SignRep Table 4.
- Gallery = TRAIN (40,154 videos, 2,731 glosses); queries = TEST (32,941), signer-disjoint.
- Per-clip features `(T×768)` mean-pooled → `(768,)`; per-query cosine similarity to every gallery video; **per-gloss max similarity** (per-gloss NN); rank all 2,731 glosses.
- With `r` = 0-indexed rank of the true gloss: `DCG=1/log2(r+2)`, `MRR=1/(r+1)`, `Rec@k=1[r<k]`, k∈{1,5,10,20}, mean ×100.
- `--min_coverage 0.999` **forces every run onto the identical gallery** — guards against a partial feature dir silently giving an easier eval on a smaller gallery.

### B2. Eval B — linear probe (`eval_asl_citizen_linear_probe.py`)
- Train `Linear(768→2731)` softmax on frozen TRAIN features (AdamW, 50 epochs), evaluate on frozen TEST, rank classes by logit, **identical DCG/MRR/Rec@k formulas** as Eval A.
- (As noted in §1, this is our own protocol, loosely benchmarked against SignRep's NN-recognition numbers.)

**Why the SignRep frozen protocol for this phase — and why it is the right call:**
1. **It is the native ASL-Citizen metric.** DCG/MRR/Recall retrieval is what the dataset defines; SignRep adopted it. Logos's Top-1-on-WLASL setup simply does not apply to ASL-Citizen.
2. **It isolates the representation, not a head.** The thesis question is whether backbone fine-tuning / appearance augmentation changes the *features*. Frozen NN retrieval measures exactly that; a trained classification head would conflate head capacity and training noise with representation quality and require per-run training.
3. **Comparability across a large sweep.** ~15 feature sets (baselines, run1/2/3, mid-unfreeze, SDDA, glosscon…) on one fixed gallery; NN retrieval is deterministic (no seed variance).
4. **External anchors.** SignRep Tables 3/4 give published I3D / ST-GCN / SignRep numbers on the *same* protocol.

---

## 4. Setup C — NGT unreal-augmentation cross-signer retrieval

**Goal.** Test whether adding synthetic Unreal-Engine signer renders (palmer/digits)
to real mocap views (piotrAnims) makes the embedding **appearance/signer-invariant**,
measured by transfer to a genuinely held-out real signer (**Bushuis**). This is the
*cross-domain* counterpart to the in-domain ASL-Citizen invariance question (which the
memory notes found shows ~nothing in-domain because ASL-Citizen test is signer-disjoint
but same webcam domain).

**Model.** Same SignCLIP transformer as Setup A (`MMFusionSeparate` + 12-layer
`MMBertForEncoder` on frozen Logos features), trained with `SupConLoss` to pull
together the `K` views of each sign (`NGTPairTask`, `mmpt/tasks/ngt_pair_task.py`):
- `ngt_baseline.yaml` — K=3 real views.
- `ngt_appearance_invariant.yaml` — K=9 (3 real + 3 palmer + 3 digits).
- `ngt_cls_token.yaml`, `ngt_mean_pool.yaml` — pooling ablations.
- `ngt_unreal_baseline.yaml` — K=3 synthetic only (no real signal).
- Batch sizes are chosen so encoder-pass budgets match (K×batch ≈ 192).

**Eval.** `jobs/eval_ngt.job` → `eval_ngt_retrieval.py`:
- Eval signs = `manifest ∩ {Bushuis features present}`.
- Each sign embeds its real/synthetic views + one Bushuis reference through the trained encoder.
- **Paired 1:1 retrieval**: query `i` must retrieve exactly gallery `i` among the `N` eval signs. Reports R@1/5/10/MRR, both directions (`bushuis→signer`, `signer→bushuis`) and two galleries (MIDDLE-only, avg(L+M+R)).

---

## 5. Is the NGT evaluation principled? — assessment

**Verdict: as a *controlled internal ablation* it is reasonable and well-motivated;
as a *retrieval benchmark comparable to the ASL-Citizen numbers* it has real flaws.**
The worries are partially valid. Breakdown:

### A note on "held-out signs" — *not* a valid concern
It is tempting to fault the NGT eval for training on every gallery sign (`ngt_pair_task.py`: *"There is no val/test split for NGT pretraining — all signs are used for training."*). But this matches the native ASL-Citizen protocol, which is **signer-disjoint, gloss-shared, closed-vocabulary**: train and test contain the *same* 2,731 glosses, and any test query whose gloss is absent from the train gallery is skipped as "closed-vocab" (`eval_asl_citizen_retrieval.py`, the `skipped` warning). "Generalization" in ISLR dictionary retrieval means **unseen signers / instances of the same signs**, never unseen glosses. NGT matches this axis exactly: gallery = seen training views, query = held-out **Bushuis** signer, shared sign vocabulary — structurally identical to ASL-Citizen's "gallery = train, query = new-signer test." So holding out *signs* would be the wrong fix; the NGT design is aligned with the literature on this axis.

### Valid concerns
1. **Tiny, variable gallery ⇒ inflated, non-comparable metrics.** `N` = number of matched signs (likely a few hundred). Chance R@1 = 1/N, so absolute numbers are **not** comparable to ASL-Citizen's 2,731-way retrieval. **Worse: `eval_ngt_retrieval.py` has no `min_coverage` guard** — it silently takes the intersection, so a config missing some Bushuis/manifest features gets a *smaller, easier* gallery and becomes non-comparable to the others. **This is the exact bug class already fixed on the ASL side** (`--min_coverage`).
2. **Instance-level 1:1 retrieval, not gloss-level.** ASL-Citizen has *many* gallery instances per gloss (per-gloss max-similarity); NGT has *one* gallery item per sign and scores query `i` only against exactly gallery `i`. No gloss-class notion, no distractor pool beyond the `N` eval signs; two distinct sign IDs that are the same lexical sign would score a correct retrieval as wrong.
3. **Narrow signer diversity; single seed.** ASL-Citizen has many disjoint test signers; NGT has a single held-out signer (Bushuis) and one run → no variance estimate and a generalization claim resting on one target signer.

### What is actually sound
- **Matched-budget ablation** (K=3 vs K=9 vs unreal-only, equal encoder passes) is good experimental hygiene.
- **Bushuis is a genuinely held-out real signer / different capture domain** — exactly the cross-domain test the in-domain ASL-Citizen eval *cannot* perform, and the one the memory notes flagged as necessary.
- Reporting both directions and both pooling galleries is fine.

So the **design** answers a legitimate question; the **reporting and comparability** are not yet literature-grade.

---

## 6. Can the NGT eval reuse the Setup-B (SignRep) methodology?

**Not as-is, but it should be aligned at the protocol level.**

The SignRep Table 4 protocol needs (a) a labeled ISLR test set with many glosses and
(b) a train-gallery / test-query split with **per-gloss** min-distance retrieval over a
large fixed vocabulary. The current NGT data is a small paired multi-view set with one
gallery instance per sign and a single held-out signer — so the 2,731-gloss protocol
cannot be applied directly without a comparable NGT benchmark (e.g. a Corpus NGT /
Signbank dictionary split). Note the gloss-sharing is *not* the obstacle (ASL-Citizen
shares glosses too); the obstacles are gallery size, gloss-vs-instance granularity, and
signer count.

Recommended fixes (in order of impact):
1. **Add a coverage / fixed-`N` guard** so every config is evaluated on the *identical* sign set (close the same hole fixed for ASL).
2. **Report `N` and chance (1/N)** next to every metric; never present these as comparable to ASL-Citizen absolutes.
3. **Reuse the exact `evaluate()` / DCG-MRR-Rec@k code** from `eval_asl_citizen_retrieval.py`, enlarge the distractor gallery (pool all Bushuis signs + extra distractors, or retrieve against the full NGT dictionary), and — if multiple instances per sign become available — make it gloss-level rather than 1:1.
4. **Broaden the held-out set / report variance**: more than one target signer and/or multiple seeds, given the small `N`. (The generalization axis is unseen *signers*, not unseen signs — keep it that way.)
5. Frame it in the thesis as a **controlled cross-domain ablation of the appearance-invariance intervention**, complementary to — not competing with — the ASL-Citizen benchmark.

---

## Appendix — file map

| Concern | Files |
|---------|-------|
| Setup A train | `projects/retri/signclip_asl/asl_citizen_cnn_scratch_logos_nowd.yaml` |
| Setup A native retrieval | `projects/retri/signclip_asl/test_*.yaml`, `mmpt_cli/predict.py`, `mmpt/evaluators/predictor.py` (`RetrievalPredictor`), `mmpt/evaluators/metric.py` (`RWTHFSMetric`) |
| Setup A downstream recognition | `export_signclip_features.py`, `test_recognition_supervised.py`, `test_recognition_few_shot_knn.py` |
| Setup A invariance diagnostic | `asl_citizen_signclip_l2_probe.py`, `asl_citizen_feature_l2_probe.py` |
| Setup B driver | `jobs/eval_asl_citizen_features.job`, `eval_asl_citizen_features.py` |
| Setup B Eval A (NN retrieval) | `eval_asl_citizen_retrieval.py` |
| Setup B Eval B (linear probe) | `eval_asl_citizen_linear_probe.py` |
| Setup B trainer | `Fork_Logos/train_logos_asl_citizen.py`, `extract_logos_features.py` |
| Setup C train | `projects/retri/signclip_ngt/ngt_*.yaml`, `mmpt/tasks/ngt_pair_task.py` |
| Setup C eval | `jobs/eval_ngt.job`, `eval_ngt_retrieval.py`, `make_ngt_pair_manifest.py` |
| Papers | `papers/Logos/Logos.pdf`, `papers/Logos/Logos_reference.pdf`, `papers/Wong_SignRep_*.pdf` |
</content>
</invoke>
