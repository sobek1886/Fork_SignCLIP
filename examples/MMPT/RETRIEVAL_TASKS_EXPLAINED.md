# Video-to-Text vs Video-to-Video retrieval on ASL-Citizen

Why "raw Logos Rec@1 ≈ 0.82" and "SignCLIP Rec@1 ≈ 0.80" are **not** the same number.
Both ultimately rank glosses for a query video, but they differ in (a) what the query is
compared against, (b) whether a learned cross-modal alignment is needed, and (c) how many
reference points exist per gloss. That makes one task structurally harder than the other.

---

## 1. Video-to-Text (SignCLIP) — what produced your 0.80

The trained SignCLIP model embeds the query **video** and each gloss **text** into a shared
space, then scores video·text. Gallery = **text** vectors (one semantic vector per gloss,
produced by BERT from the gloss string — all videos of "APPLE" map to the *same* text).

`mmpt/evaluators/predictor.py` — build the score matrix from the two modalities:
```python
def _get_pooled_outputs(self, outputs):
    return outputs["pooled_video"], outputs["pooled_text"]   # video & text embeddings
...
scores = np.matmul(text_hidden, video_hidden.T)              # cross-modal: text × video
```

`mmpt/evaluators/metric.py` (`RetrievalMetric`) — rank, the matching pair is the diagonal:
```python
x  = outputs                       # score matrix
sx = np.sort(-x, axis=1)           # rank candidates for each query
d  = np.diag(-x)[:, None]          # score of the CORRECT (paired) gloss text
ind = np.where((sx - d) == 0)[1]   # rank position of the correct gloss text
metrics["R1"] = float(np.sum(ind == 0)) / len(ind)   # correct text ranked #1?
```

Key properties:
- **Cross-modal**: the video must be mapped into the *text/language* space. This only works
  because contrastive training **learned the alignment** — raw features can't do it.
- **Gallery = ~2731 distinct gloss-text vectors** (1 per gloss). For a query video there is
  effectively **one reference per gloss** (its word).
- This is what you actually deploy for a dictionary: "sign in → get the word", and it
  generalizes to glosses with no training videos (encode the word, no retraining).

---

## 2. Video-to-Video NN (Eval A / SignRep Table 4) — what produced the 0.82

No text, no training. Embed the query **video** and compare it to **training videos**; for
each gloss keep the best-matching training video; rank glosses by that similarity.

`eval_asl_citizen_retrieval.py` (`evaluate`):
```python
G = F.normalize(gallery_feats)          # ~40k TRAIN VIDEO features (~15 per gloss)
Qb = F.normalize(query_feats[rows])     # test video features
sim = Qb @ G.T                          # uni-modal: video × video

# per-gloss MAX similarity = best of that gloss's ~15 gallery videos
gloss_sim = scatter_reduce(sim, gallery_gloss_idx, reduce="amax")
gt_sim = gloss_sim.gather(1, labels)            # similarity to the correct gloss
ranks  = (gloss_sim > gt_sim).sum(dim=1)        # how many glosses beat it
# Rec@1 = (ranks < 1)
```

Key properties:
- **Uni-modal**: pure visual similarity between videos. No language, no learned mapping —
  works straight out of the frozen backbone.
- **Gallery = ~40k training videos (~15 per gloss)**. A query has **~15 reference points
  per gloss** and is scored by its *single best* match (`amax`).
- Requires keeping the whole training gallery at inference; cannot handle a gloss with no
  stored videos.

---

## 2b. Linear probe (Eval B / SignRep Table 3) — the *right* thing to compare V2T against

`eval_asl_citizen_linear_probe.py` trains a single `Linear(768 -> n_gloss)` classifier on
the **frozen** train features, then ranks the ~2731 gloss classes by logit for each test
video:
```python
probe = nn.Linear(D, n_gloss)            # one weight ROW per gloss
...
logits = probe(xb)                       # (b, n_gloss): score of each gloss
gt     = logits.gather(1, yb)            # logit of the CORRECT gloss
ranks  = (logits > gt).sum(dim=1)        # how many glosses outscore it
```

Structurally this is **closer to V2T than Eval A is**: the gallery is **one learned vector
per gloss** (the classifier weight row), reached through a **trained mapping** — exactly
V2T's shape. Eval A, by contrast, keeps ~15 video references per gloss and matches the best
one (`amax`).

But it is **not** cross-modal and **not** open-vocabulary:
- The per-gloss anchor is a free **visual prototype** learned by cross-entropy. It never sees
  the gloss *string*, never touches BERT, carries **no language semantics** — the gloss is
  just an integer **label**.
- Closed-vocab: a gloss with no training videos has no weight row and cannot be classified.

So Eval B is the **ceiling for any single-vector-per-gloss model** built on these features —
the per-gloss vector is unconstrained. V2T solves the same-shaped task but **forces** that
vector to live in BERT text space (cross-modal + open-vocab). The gap from Eval B down to
V2T is the *price of that capability*, not lost discriminability.

| | **Eval A** (V2V NN) | **Eval B** (linear probe) | **SignCLIP** (V2T) |
|---|---|---|---|
| Gallery | ~40k train **videos** | n_gloss learned **weight rows** | ~2731 **BERT text** vectors |
| Refs per gloss | **~15** | **1** | **1** |
| Match rule | max over gloss's videos | single logit | single dot product |
| Learned mapping? | no (frozen) | **yes** (head on frozen feats) | **yes** (full contrastive) |
| Uses language/text? | no | **no** (label index only) | **yes** (BERT of gloss string) |
| Open-vocab? | no | no | **yes** |

Ladder of constraints on "rank glosses for a query video":
- **Eval A (0.82)** — easiest: ~15 refs/gloss, no learning. Raw visual similarity.
- **Eval B (0.86)** — 1 ref/gloss, learned, *unconstrained* prototype. Ceiling for a
  single-vector-per-gloss classifier on these features.
- **V2T (0.80)** — 1 ref/gloss, learned, *constrained* to BERT text space. The ~6-pt gap
  below Eval B buys cross-modal, open-vocabulary querying.

## 3. Why 0.80 (V2T) vs 0.82 (V2V) is not "SignCLIP lost"

| | **Video→Text (SignCLIP, 0.80)** | **Video→Video NN (raw Logos, 0.82)** |
|---|---|---|
| Compared against | gloss **text** vectors | training **videos** |
| References per gloss | **1** (the word) | **~15** (videos) |
| Modality | **cross-modal** (needs learned alignment) | uni-modal (raw visual similarity) |
| Needs training? | yes (contrastive) | no (frozen features) |
| Best-case match rule | single text per gloss | **max** over ~15 videos |
| Inference cost | 2731 text vectors | ~40k-video gallery |
| Open vocabulary? | **yes** (encode any word) | no (need stored videos) |

The V2V baseline gets **~15× more reference points per gloss** and only has to match *one of
them* (`amax`), while V2T must hit a **single** text vector **and** bridge the video→language
gap. So a harder task scoring 0.80 vs an easier task scoring 0.82 says the features are
strong — **not** that the contrastive model is worse. They measure different things.

**The honest takeaway:** raw frozen Logos features are already highly discriminative
(V2V 0.82, linear probe 0.86). SignCLIP's value is the *capability* it adds — a compact,
**open-vocabulary, text-queryable** model — at a tiny cost in raw discriminability on this
in-domain set. That is exactly why the next step is to fine-tune the **encoder** itself
(where the discriminability lives) rather than train heads on frozen features.

**One-line version for the meeting:** "They're different retrieval tasks — V2V matches a
video against ~15 stored videos per gloss (uni-modal, no training), V2T matches it against a
single text vector per gloss (cross-modal, learned). The near-tie means the supervised
backbone is already strong, which motivates fine-tuning it directly."
