from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prefecture.llm import (
    DiskCache,
    LLMConfig,
    ModelConfig,
    load_config,
    make_chat_model,
    ollama_status,
)


def fake_model(cache: DiskCache, *replies: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter(AIMessage(r) for r in replies), cache=cache)


def make_config(tmp_path: Path, url: str = "http://localhost:11434") -> LLMConfig:
    return LLMConfig(
        ollama_url=url,
        cache_dir=tmp_path / "cache",
        roles={
            "agent": ModelConfig(model="agent-model", temperature=0.0, seed=7),
            "clerk": ModelConfig(model="clerk-model", temperature=0.8),
        },
    )


def test_cache_miss_then_hit_replays_the_same_output(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    messages = [SystemMessage("You are a clerk."), HumanMessage("Bonjour")]

    first = fake_model(cache, "Guichet fermé.", "something else").invoke(messages)
    # A fresh model with other scripted replies: the cache must answer instead.
    second = fake_model(cache, "never used").invoke(messages)

    assert first.content == second.content == "Guichet fermé."
    assert len(cache) == 1


def test_cache_survives_a_new_cache_instance(tmp_path: Path) -> None:
    messages = [HumanMessage("Formulaire 12-B ?")]
    fake_model(DiskCache(tmp_path), "Au guichet 7.").invoke(messages)

    replay = fake_model(DiskCache(tmp_path), "never used").invoke(messages)

    assert replay.content == "Au guichet 7."


def test_different_messages_do_not_collide(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    a = fake_model(cache, "A").invoke([HumanMessage("question A")])
    b = fake_model(cache, "B").invoke([HumanMessage("question B")])

    assert (a.content, b.content) == ("A", "B")
    assert len(cache) == 2


def test_different_call_parameters_do_not_collide(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    messages = [HumanMessage("même question")]
    plain = fake_model(cache, "plain").invoke(messages)
    with_stop = fake_model(cache, "with stop").invoke(messages, stop=["\n"])

    assert (plain.content, with_stop.content) == ("plain", "with stop")


def test_clear_empties_the_cache(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    fake_model(cache, "x").invoke([HumanMessage("q")])
    cache.clear()
    assert len(cache) == 0


def test_make_chat_model_applies_role_config_and_overrides(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    agent = make_chat_model("agent", config=config)
    clerk = make_chat_model("clerk", config=config, temperature=0.2, cache=False)

    assert (agent.model, agent.temperature, agent.seed) == ("agent-model", 0.0, 7)
    assert isinstance(agent.cache, DiskCache)
    assert agent.cache.directory == tmp_path / "cache"
    assert (clerk.model, clerk.temperature, clerk.cache) == ("clerk-model", 0.2, False)


def test_unknown_role_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="not configured"):
        make_chat_model("narrator", config=make_config(tmp_path))  # type: ignore[arg-type]


def test_load_config_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("roles:\n  agent:\n    model: m\n")
    monkeypatch.setenv("PREFECTURE_OLLAMA_URL", "http://gpu-node:11434")
    monkeypatch.setenv("PREFECTURE_CACHE_DIR", str(tmp_path / "elsewhere"))

    config = load_config(path)

    assert config.ollama_url == "http://gpu-node:11434"
    assert config.cache_dir == tmp_path / "elsewhere"
    assert config.role("agent").model == "m"


def test_repository_config_defines_both_roles() -> None:
    config = load_config()
    assert {"agent", "clerk"} <= config.roles.keys()


def test_ollama_status_reports_unreachable_server(tmp_path: Path) -> None:
    status = ollama_status(make_config(tmp_path, url="http://127.0.0.1:9"))
    assert status["reachable"] is False
    assert status["missing"] == ["agent-model", "clerk-model"]


@pytest.mark.llm
def test_real_agent_model_answers_and_replays_from_cache(tmp_path: Path) -> None:
    config = load_config()
    config.cache_dir = tmp_path
    messages = [HumanMessage("Reply with exactly one word: tampon")]

    first = make_chat_model("agent", config=config).invoke(messages)
    second = make_chat_model("agent", config=config).invoke(messages)

    assert "tampon" in str(first.content).lower()
    assert first.content == second.content
    assert len(DiskCache(tmp_path)) == 1
