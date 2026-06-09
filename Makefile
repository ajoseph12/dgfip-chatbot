.PHONY: setup lint format test data index eval experiments fusion-stemming app

setup:        ## Create/sync the environment (base + dev)
	uv sync

data:         ## Build processed chunks + dev/test split
	uv run python -m dgfip_chatbot.data.build

index:        ## Embed + cache the chunk index (needs `uv sync --group ml`)
	uv run python -m dgfip_chatbot.retrieval.build

lint:         ## Lint with ruff
	uv run ruff check .

format:       ## Auto-format with ruff
	uv run ruff format .

test:         ## Run the test suite
	uv run pytest

eval:         ## FINAL test-set eval of the chosen config — run once
	uv run python -m dgfip_chatbot.eval.run

experiments:  ## Dev-set hybrid ablations: title prepend / semantic chunking
	uv run python -m dgfip_chatbot.eval.experiments

fusion-stemming:  ## Dev-set fusion x stemming sweep (reproduces reports/fusion_stemming.md)
	uv run python -m dgfip_chatbot.eval.experiments fusion-stemming

app:          ## Launch the Streamlit retrieval-chat demo (needs the ml + app groups)
	uv run --group ml --group app streamlit run src/dgfip_chatbot/app/main.py
