"""Chat model factory and on-disk LLM output cache.

Every LLM in the project is built through `make_chat_model(role)`. A role (`agent`, `clerk`)
maps to an Ollama model and its sampling parameters in `config.yaml`. All models share a disk
cache keyed by a hash of (model configuration, call parameters such as bound tools, messages):
replaying an episode with the same world seed hits the cache and reproduces it exactly, without
the GPU.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from langchain_core._api import LangChainBetaWarning
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.load import dumps, loads
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

Role = Literal["agent", "clerk"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ModelConfig(BaseModel):
    """Sampling configuration of one role."""

    model: str
    temperature: float = 0.0
    seed: int = 0
    num_ctx: int = 8192
    num_predict: int | None = None
    # None lets Ollama decide; False disables the "thinking" of reasoning models.
    reasoning: bool | None = None


class LLMConfig(BaseModel):
    """Configuration of all roles, plus where Ollama and the cache live."""

    ollama_url: str = "http://localhost:11434"
    cache_dir: Path = Path(".cache/llm")
    roles: dict[str, ModelConfig] = Field(default_factory=dict)

    def role(self, role: Role) -> ModelConfig:
        try:
            return self.roles[role]
        except KeyError:
            raise KeyError(f"Role {role!r} is not configured in config.yaml") from None


def load_config(path: Path | None = None) -> LLMConfig:
    """Read `config.yaml`, then apply environment overrides.

    `PREFECTURE_OLLAMA_URL` and `PREFECTURE_CACHE_DIR` override the file, so that the same
    config can point to a remote Ollama (e.g. Onyxia) without editing it. A relative
    `cache_dir` is resolved against the project root.
    """
    path = path or DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    config = LLMConfig.model_validate(raw)
    if url := os.environ.get("PREFECTURE_OLLAMA_URL"):
        config.ollama_url = url
    if cache_dir := os.environ.get("PREFECTURE_CACHE_DIR"):
        config.cache_dir = Path(cache_dir)
    if not config.cache_dir.is_absolute():
        config.cache_dir = PROJECT_ROOT / config.cache_dir
    return config


class DiskCache(BaseCache):
    """LangChain cache storing each LLM output as a JSON file.

    The key is the SHA-256 of LangChain's `llm_string` (serialized model + call kwargs,
    including bound tools) and `prompt` (serialized messages, ids stripped). Files are sharded
    by the first two hex digits and written atomically, so concurrent episodes are safe.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    @staticmethod
    def key(prompt: str, llm_string: str) -> str:
        return hashlib.sha256(f"{llm_string}\x00{prompt}".encode()).hexdigest()

    def _path(self, prompt: str, llm_string: str) -> Path:
        key = self.key(prompt, llm_string)
        return self.directory / key[:2] / f"{key}.json"

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        path = self._path(prompt, llm_string)
        if not path.exists():
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LangChainBetaWarning)
            return loads(path.read_text(), allowed_objects="core")

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        path = self._path(prompt, llm_string)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dumps(list(return_val))
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as f:
            f.write(payload)
        Path(f.name).replace(path)

    def clear(self, **kwargs: Any) -> None:
        for file in self.directory.glob("*/*.json"):
            file.unlink()

    def __len__(self) -> int:
        return sum(1 for _ in self.directory.glob("*/*.json"))


def make_chat_model(
    role: Role,
    *,
    config: LLMConfig | None = None,
    cache: bool = True,
    **overrides: Any,
) -> ChatOllama:
    """Build the chat model of a role.

    Args:
        role: which configured role to build (`agent` or `clerk`).
        config: configuration to use; read from `config.yaml` when omitted.
        cache: whether to read and write the disk cache. Disable it to sample fresh outputs.
        **overrides: fields of `ModelConfig` to override for this model only
            (e.g. `temperature=0.7`, `seed=3`).
    """
    config = config or load_config()
    params = config.role(role).model_copy(update=overrides)
    return ChatOllama(
        base_url=config.ollama_url,
        cache=DiskCache(config.cache_dir) if cache else False,
        **params.model_dump(exclude_none=True),
    )


def configured_models(config: LLMConfig | None = None) -> Sequence[str]:
    """Distinct Ollama model names used by the configured roles."""
    config = config or load_config()
    return sorted({role.model for role in config.roles.values()})


def ollama_status(config: LLMConfig | None = None) -> dict[str, Any]:
    """Ask the Ollama server which models are available.

    Returns `{"reachable": bool, "available": [...], "missing": [...]}`; `missing` lists the
    configured models that have not been pulled.
    """
    import ollama

    config = config or load_config()
    wanted = configured_models(config)
    try:
        listed = ollama.Client(host=config.ollama_url).list()
    except Exception as exc:  # connection refused, timeout...
        return {"reachable": False, "error": str(exc), "available": [], "missing": list(wanted)}
    available = sorted(m.model for m in listed.models if m.model)
    # Ollama names an untagged model "name:latest".
    normalized = set(available) | {name.removesuffix(":latest") for name in available}
    missing = [name for name in wanted if name not in normalized]
    return {"reachable": True, "available": available, "missing": missing}


__all__ = [
    "DiskCache",
    "LLMConfig",
    "ModelConfig",
    "Role",
    "configured_models",
    "load_config",
    "make_chat_model",
    "ollama_status",
]
