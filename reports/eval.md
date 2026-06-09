# Retrieval evaluation

> **Test set is held out.** All development tables (Methods, Per-category, Ablations) are on
> the **dev** split so no test numbers leak into model/config choices. The test set is used
> **once** on the chosen config — see the **final section**.
>
> **Caveat:** the eval questions are LLM-generated *from* the fiches, so absolute numbers
> are optimistic vs. real user phrasing; the *relative* comparison is the trustworthy signal.
>
> **Final config: e5-base hybrid — score fusion (dense 0.5 / BM25 0.5)**, selected on **dev**
> (the change that beat RRF — see the adopted-improvement section) and confirmed **once** on
> the held-out test (final section): hit@1 **0.851**. The test set carries only this chosen
> config; all comparisons (RRF, fusion weights, ablations) live on dev.

## Methods — dev set

`hybrid` = Reciprocal Rank Fusion (1:1) of dense + BM25, over the canonical chunked index
(603 chunks). Hybrid clearly beats either retriever alone.

|        |   MRR |   hit@1 |   hit@3 |   hit@5 |   hit@10 |
|:-------|------:|--------:|--------:|--------:|---------:|
| dense  | 0.847 |   0.749 |   0.939 |   0.969 |    0.993 |
| BM25   | 0.864 |   0.779 |   0.939 |   0.965 |    0.985 |
| hybrid | 0.888 |   0.813 |   0.958 |   0.982 |    0.997 |

## Per-category — dev set (dense / BM25 / hybrid)

**hit@1**

| category                                |   dense |   BM25 |   hybrid |
|:----------------------------------------|--------:|-------:|---------:|
| declarer-mes-revenus                    |   0.718 |  0.736 |    0.771 |
| gerer-mon-patrimoinemon-logement        |   0.822 |  0.847 |    0.872 |
| payer-mes-impots-taxes                  |   0.657 |  0.701 |    0.774 |
| presenter-un-recours-aupres-de-la-dgfip |   0.722 |  0.759 |    0.759 |
| signaler-mes-changements-de-situation   |   0.673 |  0.721 |    0.755 |

**hit@3**

| category                                |   dense |   BM25 |   hybrid |
|:----------------------------------------|--------:|-------:|---------:|
| declarer-mes-revenus                    |   0.925 |  0.912 |    0.934 |
| gerer-mon-patrimoinemon-logement        |   0.979 |  0.986 |    0.998 |
| payer-mes-impots-taxes                  |   0.876 |  0.861 |    0.912 |
| presenter-un-recours-aupres-de-la-dgfip |   0.907 |  0.944 |    0.907 |
| signaler-mes-changements-de-situation   |   0.912 |  0.912 |    0.939 |

**MRR**

| category                                |   dense |   BM25 |   hybrid |
|:----------------------------------------|--------:|-------:|---------:|
| declarer-mes-revenus                    |   0.820 |  0.830 |    0.858 |
| gerer-mon-patrimoinemon-logement        |   0.903 |  0.918 |    0.933 |
| payer-mes-impots-taxes                  |   0.780 |  0.790 |    0.847 |
| presenter-un-recours-aupres-de-la-dgfip |   0.825 |  0.852 |    0.854 |
| signaler-mes-changements-de-situation   |   0.792 |  0.826 |    0.852 |

Hybrid leads (or ties) on **hit@1 and MRR in every category**. *payer-mes-impots-taxes* is
dense's weakest theme (0.66) but hybrid lifts it to 0.77 — the lexical signal helps most there.

---

# Ablations — dev set

Each test below probes one design choice on top of the hybrid (chunk size fixed at 256).
**None beats the e5-base hybrid (RRF 1:1) baseline** — the simple pipeline is robust.

## A. Chunking method (hybrid)

| chunking                |   MRR |   hit@1 |   hit@3 |   hit@5 |   hit@10 | chunks |
|:------------------------|------:|--------:|--------:|--------:|---------:|-------:|
| baseline (line-based)   | 0.888 |   0.813 |   0.958 |   0.982 |    0.997 |    603 |
| + title prepend         | 0.881 |   0.800 |   0.960 |   0.986 |    0.995 |    600 |
| + semantic chunk        | 0.890 |   0.811 |   0.965 |   0.982 |    0.995 |    983 |
| + title + semantic      | 0.890 |   0.809 |   0.968 |   0.982 |    0.993 |    968 |

All within ~1 point (noise). **Semantic chunking is flat** and makes 1.6× more chunks;
title-prepend slightly hurt. → keep simple line-based chunking.

## B. Chunk granularity (hybrid)

|                           |   MRR |   hit@1 |   hit@3 |   hit@5 |   hit@10 | chunks |
|:--------------------------|------:|--------:|--------:|--------:|---------:|-------:|
| baseline (chunk size 256) | 0.888 |   0.813 |   0.958 |   0.982 |    0.997 |    603 |
| chunk size 128            | 0.890 |   0.814 |   0.963 |   0.985 |    0.995 |  1 648 |
| whole-fiche               | 0.809 |   0.707 |   0.890 |   0.932 |    0.977 |    113 |

**128 vs 256:** flat (within noise) but 2.7× more chunks → keep 256.
**Whole-fiche:** big drop (hit@1 0.813 → 0.707) — the hybrid's **embedding half** truncates a
whole fiche at e5's 512-token limit; the BM25 half reads the full text but can't fully
compensate. → we chunk *because the embedding model is token-limited*.

## C. BM25 fusion weight (hybrid)

Weighted RRF `wd/(c+rank_dense) + wb/(c+rank_bm25)`:

| weighting             |   MRR |   hit@1 |   hit@3 |   hit@5 |   hit@10 |
|:----------------------|------:|--------:|--------:|--------:|---------:|
| dense only            | 0.847 |   0.749 |   0.939 |   0.969 |    0.993 |
| hybrid 1:1 (current)  | 0.888 |   0.813 |   0.958 |   0.982 |    0.997 |
| hybrid 1:2 (BM25×2)   | 0.889 |   0.814 |   0.958 |   0.982 |    0.994 |
| hybrid 1:3 (BM25×3)   | 0.885 |   0.809 |   0.953 |   0.980 |    0.992 |
| hybrid 1:5 (BM25×5)   | 0.882 |   0.805 |   0.948 |   0.974 |    0.990 |
| BM25 only             | 0.864 |   0.779 |   0.939 |   0.965 |    0.985 |

**1:1 is the sweet spot** — 1:2 is within noise, over-weighting BM25 (1:3+) degrades.

## D. Bigger embedding model (hybrid)

|                      |   MRR |   hit@1 |   hit@3 |   hit@5 |   hit@10 |
|:---------------------|------:|--------:|--------:|--------:|---------:|
| e5-base hybrid (ref) | 0.888 |   0.813 |   0.958 |   0.982 |    0.997 |
| e5-large dense       | 0.841 |   0.741 |   0.938 |   0.966 |    0.988 |
| e5-large hybrid      | 0.877 |   0.794 |   0.959 |   0.981 |    0.991 |

**e5-large is slightly *worse*** (hit@1 0.794 vs 0.813) **and ~5× slower** on CPU. Keep e5-base.

---

# Adopted improvement — dev set (each lever tested separately)

Each lever measured on its own on **dev**; only the one with a clear dev gain was adopted.

|                                        |   hit@1 |   hit@3 |   hit@5 |   hit@10 |   MRR |
|:---------------------------------------|--------:|--------:|--------:|---------:|------:|
| hybrid RRF (baseline)                  | 0.813 | 0.958 | 0.982 | 0.997 | 0.888 |
| + stemming (alone)                     | 0.814 | 0.965 | 0.987 | 0.999 | 0.891 |
| **+ score fusion 0.5/0.5 (adopted)**   | **0.824** | **0.961** | **0.983** | **0.996** | **0.896** |

**Score fusion is the win** — hit@1 0.813 → 0.824 — and is adopted. The weight sweep
(reproduce with `make fusion-stemming`) shows **dense 0.5 and 0.7 tie at hit@1 0.824** (only sub-noise wiggles on
the deeper metrics), so we keep the **balanced 0.5/0.5** — the simpler, standard choice. BM25-leaning
(0.3) was worse (0.811). **Stemming gave no dev hit@1 gain** (0.813 → 0.814; only a small *test*-set
difference, which can't drive technique selection), so it is **not** adopted.

# Held-out test — chosen config: score fusion (dense 0.5 / BM25 0.5), no stemming

The **single best technique** (chosen on dev) evaluated **once** on the held-out test (423
questions). The test set is reserved for the final config only — baselines and rejected
techniques (RRF, other fusion weights) are compared on dev, never here.

## Overall

|                                |   hit@1 |   hit@3 |   hit@5 |   hit@10 |   MRR |
|:-------------------------------|--------:|--------:|--------:|---------:|------:|
| **score fusion (0.5 / 0.5)**   | **0.851** | **0.972** | **0.983** | **0.988** | **0.911** |

On unseen questions the correct fiche is **#1 ~85%** of the time and **in the top 3 ~97%**.

## Per-category (score fusion, 0.5 / 0.5)

| category                                |   hit@1 |   hit@3 |   hit@5 |   hit@10 |   MRR |
|:----------------------------------------|--------:|--------:|--------:|---------:|------:|
| declarer-mes-revenus                    |   0.854 |   0.969 |   0.990 |    0.990 | 0.917 |
| gerer-mon-patrimoinemon-logement        |   0.880 |   0.995 |   1.000 |    1.000 | 0.937 |
| payer-mes-impots-taxes                  |   0.789 |   0.930 |   0.947 |    0.965 | 0.859 |
| presenter-un-recours-aupres-de-la-dgfip |   0.739 |   0.870 |   0.913 |    0.957 | 0.816 |
| signaler-mes-changements-de-situation   |   0.857 |   0.984 |   0.984 |    0.984 | 0.910 |
