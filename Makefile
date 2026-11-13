-include .env
export

.PHONY: install test test-llm lint fmt doctor play eval viewer

install:  ## Install the environment and git hooks
	uv sync
	uv run pre-commit install

test:  ## Offline test suite (no real model)
	uv run pytest -m "not llm" -q

test-llm:  ## Tests that call the local Ollama models
	uv run pytest -m llm -q

lint:  ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

fmt:  ## Auto-fix lint and format
	uv run ruff check --fix .
	uv run ruff format .

doctor:  ## Check that Ollama is up and the configured models are pulled
	uv run prefecture doctor

play:  ## Play the préfecture yourself in the terminal
	@echo "Not available yet: arrives at the end of step 1 (world engine)."

eval:  ## Run a batch of episodes and write JSONL traces to runs/
	@echo "Not available yet: arrives at step 5 (evaluation)."

viewer:  ## Serve the trace viewer
	@echo "Not available yet: arrives at step 5 (evaluation)."
