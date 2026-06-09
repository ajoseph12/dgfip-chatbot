# DGFiP Chatbot

A retrieval-first FAQ chatbot for the **DGFiP "espace particulier"**: given a citizen's
question, it routes to the **most relevant *fiche pratique*** (from impots.gouv) and cites
the official source. Built as a POC for a technical test.

- **Why & for whom:** [`docs/business_context.md`](docs/business_context.md)
- **Build plan:** [`docs/phases/global_phase.md`](docs/phases/global_phase.md)
- **Datasets:** [`data/README.md`](data/README.md)

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) (it will fetch Python 3.12 automatically).

```bash
git clone <repo> && cd dgfip-chatbot
make setup        # uv sync — creates .venv from the lockfile (base + dev)
make data         # build the chunk table + dev/test split into data/processed/
make test         # run the test suite
```

> **Why uv?** A single tool replacing pip + venv + pyenv + pip-tools. The committed
> `uv.lock` pins every direct **and transitive** dependency (with hashes, cross-platform),
> so `uv sync` reproduces the *exact* environment — `git clone && uv sync` just works.

## Common commands

| Command | Does |
|---|---|
| `make setup` | Sync the environment (base + dev) |
| `make data` | Build processed chunks + dev/test split *(Phase 1)* |
| `make index` | Embed + cache the chunk index *(Phase 2; needs `uv sync --group ml`)* |
| `make lint` / `make format` | Lint / format with ruff |
| `make test` | Run pytest |
| `make eval` | Retrieval eval harness *(Phase 3)* |
| `make app` | Streamlit demo *(Phases 5–6)* |

Heavier dependencies are installed per phase: `uv sync --group ml` (Phase 2),
`--group llm` (Phase 4, optional), `--group app` (Phases 5–6).

## Project structure

```
src/dgfip_chatbot/
  config.py        # paths, model names, params, secrets (pydantic-settings)
  data/            # Phase 1 — load / clean / chunk
  retrieval/       # Phase 2 — embeddings, index, retriever
  eval/            # Phase 3 — metrics harness
  generation/      # Phase 4 (OPTIONAL) — grounded answer layer (Mistral); not required
  app/             # Phases 5–6 — Streamlit app (UI + serving; no separate backend)
data/raw/          # provided CSVs (KB + eval questions) — see data/README.md
data/processed/    # `make data` output: chunks.parquet + question splits (gitignored)
tests/
reports/           # `make eval` output (eval.md)
docs/              # business context + phased build plan
```

## Data exploration

A first look (see [`notebooks/phase1_eda.ipynb`](notebooks/phase1_eda.ipynb)):

- **Themes:** 5 categories; **property & housing is the largest (~44% of questions)**, not income declaration.
- **Length:** fiches vary widely (median ~4.9k chars) → motivates chunking.
- **Skew:** questions per fiche range from 6 to 55.
- **Vocabulary overlap (key signal):** most questions share many content words with their target fiche (favourable to **BM25**), but a meaningful tail does not (needs **semantic embeddings**) → together motivating a **hybrid** retriever.
- **Caveat:** questions are LLM-generated from the fiches, so this overlap is likely **optimistic** versus real user phrasing.

## Results

Methods evaluated on **dev** (test held out): **dense** (e5-base), **BM25**, **hybrid — RRF**,
**hybrid — score fusion**. Levers tried and **rejected** (no dev gain): stemming, semantic
chunking, title-prepend, cross-encoder rerank, e5-large, chunk-size 128.

**Best: hybrid with confidence-aware score fusion (dense 0.5 / BM25 0.5)** — the only lever that
beat the RRF baseline on dev. Scored **once** on the held-out test:

| score fusion (0.5 / 0.5) | hit@1 | hit@3 | MRR |
|:--|--:|--:|--:|
| dev (1 004 q) | 0.824 | 0.961 | 0.896 |
| **test (423 q)** | **0.851** | **0.972** | **0.911** |

Full tables: [`reports/eval.md`](reports/eval.md)
