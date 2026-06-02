# DGFiP Chatbot

A retrieval-first FAQ chatbot for the **DGFiP "espace particulier"**: given a citizen's
question, it routes to the **most relevant *fiche pratique*** (from impots.gouv) and cites
the official source. Built as a POC for a technical test.

- **Why & for whom:** [`docs/business_context.md`](docs/business_context.md)
- **Build plan:** [`docs/phases/global_phase.md`](docs/phases/global_phase.md)

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) (it will fetch Python 3.12 automatically).

```bash
git clone <repo> && cd dgfip-chatbot
make setup        # uv sync — creates .venv from the lockfile (base + dev)
make test         # run the smoke tests
```

> **Why uv?** A single tool replacing pip + venv + pyenv + pip-tools. The committed
> `uv.lock` pins every direct **and transitive** dependency (with hashes, cross-platform),
> so `uv sync` reproduces the *exact* environment — `git clone && uv sync` just works.

## Common commands

| Command | Does |
|---|---|
| `make setup` | Sync the environment (base + dev) |
| `make lint` / `make format` | Lint / format with ruff |
| `make test` | Run pytest |
| `make eval` | Retrieval eval harness *(Phase 3)* |
| `make app` | Streamlit demo *(Phase 6)* |

Heavier dependencies are installed per phase: `uv sync --group ml` (Phase 2),
`--group llm` (Phase 4), `--group app` (Phase 6).

## Project structure

```
src/dgfip_chatbot/
  config.py        # paths, model names, params, secrets (pydantic-settings)
  data/            # Phase 1 — load / clean / chunk
  retrieval/       # Phase 2 — embeddings, index, retriever
  eval/            # Phase 3 — metrics harness
  generation/      # Phase 4 — optional Mistral answer layer
  app/             # Phase 6 — Streamlit demo
data/raw/          # provided CSVs (KB + eval questions)
data/processed/    # derived artifacts (gitignored)
tests/
docs/              # business context + phased build plan
```

## Configuration

Copy `.env.example` to `.env` and fill in as needed (the `MISTRAL_API_KEY` is only used
from Phase 4). The real `.env` is gitignored.

## Status

**Phase 0 — scaffolding.** Retrieval core (Phases 1–3) is the graded deliverable; the
chat/UI layer is a thin demo on top. See the build plan for details.
