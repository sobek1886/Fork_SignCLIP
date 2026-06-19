# Logos backbone fine-tuning: architecture, what we change, and the losses

This document explains the **Logos MViTv2-S** sign-representation backbone, exactly what the
fine-tuning experiments change, and — in detail — **the different training losses we try and
what each is applied to**. Every claim is pinned to a `file:line` in the actual code.

The two repos involved:

| Repo | Role | Key files |
|---|---|---|
| `Fork_Logos/` | **training** (standalone PyTorch, no SignCLIP) | `train_logos_asl_citizen.py`, `extract_logos_features.py` |
| `fairseq/examples/MMPT/` | **evaluation** of the re-extracted features | `eval_asl_citizen_retrieval.py`, `eval_asl_citizen_features.py` |

> All line numbers below refer to `Fork_Logos/train_logos_asl_citizen.py` unless another file is named.

---

## 1. TL;DR

- The backbone is **MViTv2-S** (Multiscale Vision Transformer v2, "small"): a 16-block video
  transformer that turns a 32-frame clip into a single **768-d CLS-token** vector `z`.
- It was pretrained by the *Logos* project on Logos + AUTSL + WLASL (thousands of signers).
- We **fine-tune the backbone itself** (not a head on frozen features) by attaching a fresh
  `Linear(768 → 2731)` gloss classifier and unfreezing the top *k* of 16 blocks.
- **Everything — the classifier, every contrastive/consistency loss, and the features we
  extract for evaluation — operates on the same 768-d CLS token `z`.**
- We compare **three families of loss**, all built on cross-entropy:
  1. **CE** (cross-entropy on glosses) — every run.
  2. **Appearance-consistency** (cosine pull of an original toward its appearance-augmented copies) — the primary intervention.
  3. **CLS-token SupCon** (instance-level supervised-contrastive, with negatives + temperature) — a stronger variant of (2).
  - plus an optional **gloss-SupCon** auxiliary term.

---

## 2. The Logos backbone: MViTv2-S architecture

**Input.** One clip `x` of shape `(B, 3, T=32, 224, 224)` — 32 RGB frames at 224². The 32 frames
are sampled every `FRAME_INTERVAL=2` original frames, so a clip spans 64 consecutive frames
(`train_logos_asl_citizen.py:43-44`, `_sample_clip:138-153`).

**Tokenization.** A 3D ("cube") patch-embedding conv tokenizes the clip into a grid of
spatio-temporal tokens, and a learned **CLS token** is prepended.

**16 blocks in 4 multiscale stages.** MViTv2-S stacks `NUM_BLOCKS = 16` transformer blocks
(`:50`) grouped into 4 stages (depths `[1, 2, 11, 2]` per the MViTv2-S spec). "Multiscale" means
that, via **pooling attention** (the Q/K/V are spatially-temporally pooled with strided convs),
each stage transition **halves the spatio-temporal resolution and doubles the channel width**:

```
channels:   96  →  192  →  384  →  768
stage:       1      2       3       4
            (coarser space-time, richer channels as depth increases)
```

So **early blocks** carry high-resolution, low-level/appearance-ish features (texture, colour,
local shape); **late blocks** carry coarse, semantic/motion features. The final block emits a
**768-d** representation.

**Output = the CLS token.** The backbone is built with `output_cls_token=True` (the mmaction2
default), so the global clip descriptor is the **CLS token** after the final norm — a single
`768`-d vector. This is `z`.

Where it is built (`load_backbone_trainable:352-372`):

```python
# Fork_Logos/train_logos_asl_citizen.py:358-360
backbone = MODELS.build(dict(
    type="MViT", arch="small", drop_path_rate=0.1, dim_mul_in_attention=False,
))
```

Logos weights are loaded by stripping the `backbone.` prefix, `strict=False`, and — crucially —
**kept trainable** (no `.eval()`/`requires_grad=False` here, unlike the extractor) (`:361-372`).

---

## 3. The representation `z` (read this before the losses)

**The single most important implementation fact:** the CLS token is the representation used
*everywhere*. The model's `_pool` unwraps the backbone output to the CLS token, and `forward`
returns `(z, logits)`:

```python
# Fork_Logos/train_logos_asl_citizen.py:386-397
def _pool(self, feat):
    while isinstance(feat, (list, tuple)):
        feat = feat[-1]          # ← lands on the CLS token (output_cls_token=True)
    if feat.ndim == 5:           # (N,C,T,H,W) — only if cls token were OFF
        feat = feat.mean(dim=[2, 3, 4])
    elif feat.ndim == 3:         # (N,seq,C)
        feat = feat.mean(dim=1)
    return feat                  # (N, 768)

def forward(self, x):
    feat = self._pool(self.backbone(x))   # feat == z, the 768-d CLS token
    return feat, self.head(feat)
```

Because `output_cls_token=True`, `feat[-1]` is already the `(N, 768)` CLS token, so the
mean-pool branches are dead code. **`z` is the CLS token**, and:

- the classifier sees `head(z)`,
- the consistency loss compares `z_orig` vs `z_aug`,
- the SupCon losses shape the geometry of the `z`-space,
- `extract_logos_features.py` extracts this same `z` for evaluation.

This is why all our objectives are "on the CLS token" — there is no other representation.

---

## 4. What we change vs. the pretrained Logos model

| Change | Where | Notes |
|---|---|---|
| Add a gloss classifier head | `LogosClassifier.__init__:381-384` | `nn.Linear(768, 2731)`, truncated-normal init |
| Keep the backbone trainable | `load_backbone_trainable:352-372` | no freeze inside the loader |
| Unfreeze only the **top k** of 16 blocks | `set_trainable:416-440` | `--unfreeze_blocks k`; head always trainable |
| Layer-wise LR decay (full FT) | `build_param_groups:455-478` | `lr(depth)=base·llrd^(max−depth)` |
| New training objective(s) | `main` loop `:800-845` | CE ± consistency ± SupCon (Section 5) |
| Re-extract features, evaluate frozen | MMPT repo | extractor unchanged → checkpoint round-trips |

We do **not** change the architecture, the preprocessing, or the feature definition — so a
checkpoint saved here (`save_ckpt:780-787`, keys prefixed `backbone.` / `head.`) is read cleanly
by the unmodified `extract_logos_features.py`.

---

## 5. The losses (the core of the experiments)

All runs minimise cross-entropy; the experiments differ in **which extra term (if any)** is
added and **what it is applied to**. The total loss assembled per batch is:

```
L = L_CE  [ + λ_consist · L_consistency ]  [ + λ_supcon · L_glossSupCon ]
```

Summary — **what each loss is applied to**, and the flags/runs that enable it:

| Loss | Applied to | Positives / target | Enabled by | Runs | Code |
|---|---|---|---|---|---|
| **Cross-entropy** `L_CE` | `head(z)` logits vs gloss label | the gloss `y` | always | all | `:816` (paired), `:840` (plain) |
| **Appearance-consistency** `L_cons` | `z_orig` vs `z_aug` (cosine) | a video's own augmentations | `--paired` (default mode) | run1, run1_*, twostage | `:826-828` |
| **CLS-token SupCon** `L_con^cls` | the `z`-space (orig+augs = positives, rest = negatives) | same source video, instance label | `--paired --cls_contrastive` | run1_clscontrast | `:823-824, 828` |
| **Gloss-SupCon** (aux) | the `z`-space (same-gloss = positives) | the gloss `y` | `--lambda_supcon > 0` | optional ("Run 4") | `:832-834, 843-845` |

The weights are CLI args: `--lambda_consist` (default `0.5`, `:627`) and `--lambda_supcon`
(default `0.0`, `:632`).

### 5.1 Cross-entropy `L_CE` — every run

Standard softmax CE of the head logits against the gloss label. Applied to **`head(z)`**.

```python
# plain (CE-only) path — Fork_Logos/train_logos_asl_citizen.py:836-841
clips, labels = batch
feat, logits = model(clips)        # feat = z
loss_ce = ce(logits, labels)       # CrossEntropyLoss(), defined :750
loss = loss_ce
```

In the **paired** path, the augmentations are *also* classified (aug-as-data), so CE is computed
over `[originals ; augmentations]` with the augmentations carrying their source video's gloss:

```python
# paired path — :810-816
feat, logits = model(x)            # x = cat([orig, augs]); feat = z for all
labels_all = torch.cat([labels, batch["aug_labels"]...], 0)
loss_ce = ce(logits, labels_all)
```

### 5.2 Appearance-consistency `L_cons` — the primary intervention (Run 1)

Goal: make `z` **invariant to appearance** by pulling a video's original representation toward
the representations of its appearance-augmented copies (glasses / shirt / signer-swap / skin).
It is a **positive-only cosine pull** — no negatives.

Math (matches `backbone_ft_methods.tex`, Eq. base):

```
L_cons = mean over (i,a) pairs in batch of  ( 1 − cos(z_i, z_i^a) )
```

Code — note it indexes `feat` (which is `z`) into the original block `[:B]` and the augmentation
block `[B:]`, using the `owner` map to align each aug with its source video:

```python
# Fork_Logos/train_logos_asl_citizen.py:826-828
cos = F.cosine_similarity(feat[B:], feat[:B][owner], dim=-1)   # aug z  vs  its source z
loss_consist = (1.0 - cos).mean()
loss = loss + args.lambda_consist * loss_consist
```

- `feat[:B]` = the `z` of the `B` originals; `feat[B:]` = the `z` of the `M` augmentations;
  `feat[:B][owner]` gathers, for each augmentation, its **source original's** `z`.
- Enabled by `--paired` (`:623`) without `--cls_contrastive`.
- Only fires for batch items that actually have augmentations on disk (≈15% of train videos);
  `n_pair_batches` is logged (`:829, 904-905`).

### 5.3 CLS-token SupCon `L_con^cls` — instance contrastive (run1_clscontrast)

Supervisor's suggestion: replace the positive-only pull with a **proper contrastive** objective
that *also repels different videos*, on the same CLS token. Treat **an original + all its
augmentations as one positive set** (same "instance"), every other clip in the batch as a
negative, with ℓ2-normalised tokens and temperature τ.

Math (Eq. supcon in the .tex):

```
L_con^cls = Σ_u  (−1/|P(u)|) Σ_{p∈P(u)}  log [ exp(z_u·z_p/τ) / Σ_{v≠u} exp(z_u·z_v/τ) ]
```

How the **instance label** is built — originals get ids `0..B-1`, and each augmentation gets its
**owner's** id, so an original and its augs share a label (= positives):

```python
# Fork_Logos/train_logos_asl_citizen.py:823-824
inst = torch.cat([torch.arange(B, device=device), owner], 0)   # instance id per clip in z
loss_consist = supcon_loss(feat, inst, args.supcon_temperature) # feat = z
```

The SupCon kernel (`supcon_loss:485-499`) — note the **fp16-safe** masked-fill (it used to be
`-1e9`, which overflows half precision under AMP and crashed the run):

```python
# Fork_Logos/train_logos_asl_citizen.py:485-499
def supcon_loss(feats, labels, temperature=0.07):
    feats = F.normalize(feats, dim=-1)
    sim = feats @ feats.T / temperature
    self_mask = torch.eye(N, dtype=torch.bool, device=feats.device)
    sim = sim.masked_fill(self_mask, torch.finfo(sim.dtype).min)  # :491 fp16-safe
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    ...
    return -mean_log_prob_pos.mean()
```

It is still gated by `--paired`, plugged in at the same `loss + λ_consist·loss_consist` site
(`:820-828`), so `--lambda_consist` weights it.

### 5.4 Gloss-SupCon — optional auxiliary ("Run 4")

The *same* `supcon_loss` kernel, but with **gloss labels** instead of instance labels — pulls
together all clips of the **same gloss** in the batch (and pushes apart different glosses). It is
additive and independent of the paired machinery:

```python
# Fork_Logos/train_logos_asl_citizen.py:832-834 (paired)  /  :843-845 (plain)
if args.lambda_supcon > 0:
    loss = loss + args.lambda_supcon * supcon_loss(feat, labels_all, args.supcon_temperature)
```

Default `--lambda_supcon 0.0` (off). The difference from 5.3 is purely **what `labels` means**:
*instance* (same source video) in 5.3 vs *gloss* (same class) here.

---

## 6. How a paired batch flows through the model

The consistency/SupCon losses need an original and its augmentations **in the same forward
pass**. The collate function batches originals, flattens all present augmentations, and records
an `owner` index mapping each augmentation back to its source original:

```python
# Fork_Logos/train_logos_asl_citizen.py:326-345  (paired_collate)
origs  = torch.stack([b[0] for b in batch])      # (B, 3, T, H, W)
# for each original `owner`, append its aug clips and remember owner/label
return {"orig": origs, "labels": labels,
        "augs": augs,            # (M, 3, T, H, W) — M = total augs in the batch
        "aug_owner": aug_owner,  # (M,) source-original index for each aug
        "aug_labels": aug_labels}
```

Then one concatenated forward, split into the `B` originals and `M` augmentations
(`:801-828`):

```
x = cat([orig (B) ; augs (M)])  →  feat = z for all (B+M)
                                   feat[:B] = originals' z
                                   feat[B:] = augmentations' z
CE  on head(z) for all B+M  (augs classified too = aug-as-data)
consistency / SupCon  uses feat[:B], feat[B:], owner
```

`PairedDataset:258-323` iterates **original** rows (so coverage is all videos) and attaches
whichever of `augmented_frames/{video_id}/{aug}/` exist (0, 1, or more).

---

## 7. Fine-tuning depth (what "unfreeze top k" means) + LLRD

```python
# Fork_Logos/train_logos_asl_citizen.py:416-440  (set_trainable)
first_trainable = NUM_BLOCKS - unfreeze_blocks   # k = unfreeze_blocks
# blocks with index >= first_trainable are trainable; earlier blocks frozen;
# final norm + head always trainable; >=NUM_BLOCKS ⇒ full fine-tune
```

- `--unfreeze_blocks 2` (**partial**): only the last stage (blocks 14–15) adapts.
- `--unfreeze_blocks 6` (**middle-ground**, "6-layer"): last stage + part of stage 3.
- `--unfreeze_blocks 16` (**full**): whole backbone, usually with `--llrd 0.75`.

Frozen blocks are also put in `eval()` to disable stochastic depth (`set_train_mode:443-452`).

Layer-wise LR decay gives deeper blocks a higher LR (`build_param_groups:455-478`, depth via
`_block_index:404-413`): `lr(depth) = base_lr · llrd^(max_depth − depth)`.

> **Empirical note (see `backbone_ft_methods.tex` §Findings):** depth is the lever — CE with
> `k=6` is the best run; adding the consistency term *hurts* at `k=6`. The consistency objective
> only ever had gradient access to the **top** blocks, i.e. **not** the early/appearance-encoding
> layers — a known limitation to revisit if testing appearance invariance cross-domain.

---

## 8. Two-stage training (`twostage`)

Resume an already-converged CE baseline (backbone **and** head), then continue at 100× lower LR
with the consistency term added:

```python
# Fork_Logos/train_logos_asl_citizen.py:731-739  (--load_head)
_head = {k[len("head."):]: v for k, v in _st.items() if k.startswith("head.")}
model.head.load_state_dict(_head)        # resume the head, not just the backbone
```

The job sets `--checkpoint <run2_direct224 checkpoint> --load_head --paired --lr 1e-5`.

---

## 9. Run matrix (loss × data × depth)

| Run (job) | `--paired` | extra flag | data (`--train_csv`) | loss | depth |
|---|---|---|---|---|---|
| `run2` baseline | – | – | `splits` | CE | 2 |
| `run3` aug-as-data | – | – | `splits_aug` | CE (augs as extra data) | 2 |
| `run1` consistency | ✓ | – | `splits` + augs | CE + cosine consistency | 2 |
| `run1_{glasses,shirt1,signerswap,skin}` | ✓ | `--aug_names <one>` | `splits` + that aug | CE + consistency (isolated) | 2 |
| `run2_mid` | – | – | `splits` | CE | 6 + LLRD |
| `run1_mid` | ✓ | – | `splits` + augs | CE + consistency | 6 + LLRD |
| `run1_clscontrast` | ✓ | `--cls_contrastive` | `splits` + augs | CE + CLS-SupCon | 2 |
| `twostage` | ✓ | `--load_head` (resume run2) | `splits` + augs | CE + consistency @ low LR | 2 |

Jobs live in `Fork_Logos/ft_asl_citizen_*.job`.

---

## 10. Preprocessing & data gotchas (affect *what* `z` sees)

- **Preprocessing modes** (`_preprocess_frame:92-135`): `logos_native` (aspect-preserving
  resize-300 + pad + center-crop 224, what Logos trained on), `direct224` (legacy stretch),
  `letterbox224` (resize-long-224 + pad, best *frozen* but washes out after FT), `resize300crop`.
  The training transform **must equal** the re-extraction `--preproc` or the features are invalid
  (`--orig_preproc`/`--aug_preproc`, `:612-618`).
- **Stride fix** (`STRIDE2_AUGS:55`): `signer_swap`/`skin_mst_diffusion` frames were exported at
  stride 2 (half the frames), so their clips use `frame_interval = 1` to span the same duration
  as the stride-1 augs (`ClipDataset:224`, `PairedDataset:315`).

---

## 11. Evaluation (separate, on re-extracted frozen features)

After training, `extract_logos_features.py` re-extracts the **same CLS-token `z`** for every
ASL-Citizen video (with `--preproc` matching training), then in the MMPT repo:

- **Eval A — nearest-neighbour dictionary retrieval** (`eval_asl_citizen_retrieval.py`): per-gloss
  max cosine similarity of a test `z` to the train gallery, rank the GT gloss.
- **Eval B — linear probe** (`eval_asl_citizen_features.py` runs A + B together).
- Both report DCG / MRR / Rec@{1,5,10,20} on the signer-disjoint test split. Jobs:
  `MMPT/jobs/eval_asl_citizen_features.job`, `eval_asl_citizen_ablations.job`.

---

## 12. File map

| File | What it documents above |
|---|---|
| `Fork_Logos/train_logos_asl_citizen.py` | the whole trainer — losses §5, batch flow §6, depth §7, two-stage §8 |
| `Fork_Logos/extract_logos_features.py` | re-extraction of `z` (same `_pool`/preprocessing) |
| `MMPT/backbone_ft_methods.tex` | the thesis methods + findings prose |
| `MMPT/eval_asl_citizen_retrieval.py` / `eval_asl_citizen_features.py` | Eval A / Eval B |
