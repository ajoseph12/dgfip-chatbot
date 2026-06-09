# Data

Datasets for the DGFiP fiche-routing POC, sourced from the "espace particulier" of
[impots.gouv.fr](https://www.impots.gouv.fr/). Both files are **comma-separated, UTF-8,
French**.

```
data/
├── raw/         # the two provided CSVs (committed, treated as read-only)
└── processed/   # derived artifacts: cleaned fiches, chunks, embeddings, index (gitignored)
```

The two files form a **labeled retrieval dataset**: a knowledge base of fiches, and a set
of questions each tagged with its expected fiche.

---

## 1. `raw/info_particulier_impot.csv` — knowledge base

**113 fiches pratiques** (one per row), each a help sheet from impots.gouv.

| Column | Meaning |
|---|---|
| *(unnamed)* | **fiche id**, 0–112, contiguous — the join key |
| `niveau_0` | top-level category (stored as a URL); **5 distinct** |
| `niveau_1` | sub-category (stored as a URL); **37 distinct** |
| `URL` | canonical URL of the fiche |
| `Titre` | title (upper-case), e.g. `OBLIGATIONS DÉCLARATIVES` |
| `Texte` | full body text of the fiche |

### Category distribution (`niveau_0`, by number of fiches)

| Fiches | Theme (`niveau_0` slug) |
|---:|---|
| 37 | `gerer-mon-patrimoine` / `mon-logement` — property & housing |
| 32 | `declarer-mes-revenus` — declaring income |
| 20 | `signaler-mes-changements-de-situation` — life changes |
| 16 | `payer-mes-impots-taxes` — paying taxes |
| 8  | `presenter-un-recours-aupres-de-la-dgfip` — recours / complaints |

### `Texte` characteristics

- **Length:** min 460 / median ~4,900 / mean ~5,870 / max ~25,800 chars
  (≈ median ~1,200 tokens, max ~6,400).
- **86%** of fiches exceed ~512 tokens, **59%** exceed ~1k, **0%** exceed ~8k.
  → a long-context embedder could take a fiche whole; a 512-token one would truncate most.
- **Cleaning artifacts** (mild, consistent):
  - non-breaking spaces (`\xa0`) in 106/113 fiches
  - curly quotes (`’`, `«`) in 110/113
  - the **title is repeated as the first line** of `Texte` in **all 113**
  - structure is single `\n` line breaks (~38/doc median, max 198) — headings/bullets,
    **not** blank-line-separated paragraphs
  - **no** HTML tags, carriage returns, tabs, double blank lines, or runs of 2+ spaces

---

## 2. `raw/questions_fiches_fip.csv` — evaluation set

**1,427 user questions** (one per row), generated semi-synthetically by an LLM, each
tagged with the fiche that should answer it.

| Column | Meaning |
|---|---|
| *(unnamed)* | question id |
| `question` | the natural-language user question (French) |
| `num_texte` | id of the expected fiche → **joins to the fiche id above** |

### Coverage & distribution

- Every `num_texte` is valid (∈ 0–112); **all 113 fiches are referenced**.
- Questions per fiche: **min 6 / median 10 / max 55** — **skewed** (e.g. fiche 40
  *SUIVRE MES PAIEMENTS* has 55).
- Distribution of the **1,427 questions by theme** (via each question's fiche `niveau_0`):

| Questions | Share | Theme |
|---:|---:|---|
| 623 | 43.7% | property & housing |
| 323 | 22.6% | declaring income |
| 210 | 14.7% | life changes |
| 194 | 13.6% | paying taxes |
| 77  | 5.4%  | recours |

- Questions are **specific / extractive** — they target facts buried inside a fiche (often
  quoting thresholds, dates, or even UI navigation steps), so matching titles alone is
  insufficient; the body text must be searched.

---

## Using the two together

`questions_fiches_fip.num_texte` → `info_particulier_impot` *(unnamed id)*. This gives a
ground-truth **(question → correct fiche)** mapping, suitable for retrieval metrics
(Top-k accuracy, MRR, recall@k), ideally reported **per theme** given the skew above.

> The largest theme is **property & housing**, ahead of income declaration — the task is
> broader than "tax declaration help."
