.PHONY: setup lint format test data index eval app

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

eval:         ## Run the retrieval eval harness (Phase 3)
	@echo "Eval harness — implemented in Phase 3."

app:          ## Launch the Streamlit demo (Phase 6)
	@echo "Streamlit demo — implemented in Phase 6."
