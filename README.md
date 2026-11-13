# Préfecture

An LLM agent must obtain a document from a procedurally generated, absurd French préfecture: inconsistent guichets, pièces justificatives that require other pièces, impossible opening hours, and sometimes a Catch-22 where the right answer is to declare the démarche impossible. The world is a deterministic engine with exact ground truth; LLM clerks only verbalise its verdicts. That makes it a rigorous benchmark for agent architectures.

> Work in progress. The full description, results and trace viewer will come later.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) for the local models (no paid API involved)

## Installation

```bash
git clone https://github.com/martinlepadan/prefecture.git
cd prefecture
make install
```

Pull the models listed in `config.yaml`, start Ollama, and check the setup:

```bash
ollama pull qwen3.5:9b
ollama serve          # in another terminal, if not already running
make doctor
```

## Development

```bash
make test       # offline tests, no model needed
make test-llm   # tests that call the local models
make lint       # ruff
```

LLM outputs are cached in `.cache/llm/`, keyed by model configuration and messages, so an episode replayed with the same world seed is identical. Copy `.env.example` to `.env` to point to another Ollama server.

## License

MIT
