.PHONY: setup lint format test data index eval experiments app

setup:        ## Create/sync the environment (base + dev)
	uv sync

data:         ## Build processed chunks + dev/test split (Phase 1)
	uv run python -m dgfip_chatbot.data.build

index:        ## Embed + cache the chunk index (Phase 2; needs `uv sync --group ml`)
	uv run python -m dgfip_chatbot.retrieval.build

lint:         ## Lint with ruff
	uv run ruff check .

format:       ## Auto-format with ruff
	uv run ruff format .

test:         ## Run the test suite
	uv run pytest

eval:         ## FINAL test-set eval of the chosen config — run once (Phase 3)
	uv run python -m dgfip_chatbot.eval.run

experiments:  ## Dev-set hybrid ablations: title prepend / semantic chunking
	uv run python -m dgfip_chatbot.eval.experiments

app:          ## Launch the Streamlit demo (Phase 6)
	@echo "Streamlit demo — implemented in Phase 6."
