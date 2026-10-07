"""Offline tests: rendering, slots, request accounting, and answer shaping.

The readout is exercised against a stub tokenizer and a mock transport, so the
suite runs with no network, no GPU, and no tokenizer download. `test_live.py`
covers the real endpoint and is opt-in.
"""

from __future__ import annotations

import base64
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Event, Lock
from types import SimpleNamespace

import httpx
import pytest

from shisa_de import Choice, DecisionModel, Noul, Readout, ReadoutError, Score, softmax
from shisa_de.calibration import (
    CALIBRATION_FILES,
    calibration_for,
    confidence,
    load_calibration,
    resolve_calibration,
    temper_binary,
    temper_distribution,
)
from shisa_de.family import family_is_explicit, model_family, resolve_family
from shisa_de.client import Decision, _MultiLabel, _question_from_head
from shisa_de.images import ImageError, prepare_image, validate_image_url
from shisa_de.policy import ReadOptions, read_policy, resolve_options
from shisa_de.questions import MAX_OPTIONS, QuestionError, render_option
from shisa_de.readout import (
    ANSWER_PREFIX,
    CALIBRATION_COMPATIBLE,
    DE2_READOUT_VERSION,
    DIRECT_SYSTEM,
    INPUT_REPEAT,
    MAX_CODES,
    READOUT_VERSION,
    READOUT_VERSIONS,
    THINK_SYSTEM,
    LetterRead,
    calibration_compatible,
    codebook,
    codes_for,
)


class StubTokenizer:
    """Character-level stand-in: every character is one token."""

    def __init__(self) -> None:
        self.templates: list[list[dict[str, str]]] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        self.templates.append(messages)
        return f"<sys>{messages[0]['content']}<user>{messages[1]['content']}<model>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, ids) -> str:
        return "".join(chr(value) for value in ids)


def make_readout(handler, **kwargs) -> Readout:
    readout = Readout(base_url="http://test.local", model="test-model", **kwargs)
    readout._client = httpx.Client(base_url="http://test.local", transport=httpx.MockTransport(handler))
    readout._tokenizer = StubTokenizer()
    return readout


def top_logprobs(entries: dict[str, float], prompt_tokens: int = 12) -> dict:
    return {
        "choices": [{"text": max(entries, key=entries.get), "logprobs": {"top_logprobs": [entries]}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1},
    }


# -- probability intent -------------------------------------------------------

@pytest.mark.parametrize("method", ["decide", "classify", "system_one"])
@pytest.mark.parametrize("calibrated", [None, False, True])
def test_probability_preserves_single_read_and_calibration(method, calibrated):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=top_logprobs({"A": -0.5, "B": -1.5}))

    with DecisionModel(readout=make_readout(handler)) as model:
        ask = getattr(model, method)
        questions = {"forecast": Noul("Will it rain tomorrow?")}
        ordinary = ask("state", questions, calibrated=calibrated)
        result = ask("state", questions, probability=True, calibrated=calibrated)
    assert calls[0] == calls[1]
    assert "probability" not in calls[1]  # SDK intent, not a server extension.
    assert result.answers["forecast"].to_dict() == ordinary.answers["forecast"].to_dict()
    assert result.to_wire() == ordinary.to_wire()
    assert ordinary.meta["probability"] is False
    assert result.meta["probability"] is True
    assert result.usage["logical_reads"] == result.usage["requests"] == 1
    raw_yes = softmax({"A": -0.5, "B": -1.5})["A"]
    expected = (raw_yes if calibrated is False else
                temper_binary(raw_yes, model.calibration.temperature_for("noul")))
    assert result["forecast"] == pytest.approx(expected)


@pytest.mark.parametrize("method", ["decide", "classify"])
def test_probability_rejects_overflow_before_any_head_is_sent(method):
    calls = []
    with DecisionModel(readout=make_readout(lambda request: calls.append(request))) as model:
        with pytest.raises(QuestionError, match="probability=True requires a single read"):
            getattr(model, method)("state", {
                "small": Noul("Will it rain?"),
                "wide": list(map(str, range(27))),
            }, probability=True)
    assert calls == []


@pytest.mark.parametrize("value", [None, 1, "true", "false"])
def test_probability_requires_boolean(value):
    calls = []
    with DecisionModel(readout=make_readout(lambda request: calls.append(request))) as model:
        with pytest.raises(ValueError, match="probability must be a boolean"):
            model.decide("state", {"forecast": Noul("Will it rain?")}, probability=value)
    assert calls == []


def test_probability_allows_letter_recovery_not_a_second_logical_read():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if "prompt_logprobs" in body:
            assert body["prompt"] == calls[0]["prompt"] + "B"
            return httpx.Response(200, json={
                "choices": [{"prompt_logprobs": [None, {"66": {"logprob": -1.5, "rank": 21}}]}],
                "usage": {"prompt_tokens": 13},
            })
        return httpx.Response(200, json=top_logprobs({"A": -0.5}))

    with DecisionModel(readout=make_readout(handler)) as model:
        result = model.decide("state", {"forecast": Noul("Will it rain?")},
                              probability=True, calibrated=False)
    assert len(calls) == result.usage["requests"] == 2
    assert result.usage["logical_reads"] == 1
    assert result.answers["forecast"].missing_from_top == ["B"]
    assert result["forecast"] == pytest.approx(softmax({"A": -0.5, "B": -1.5})["A"])


def test_probability_multilabel_and_display_options_remain_independent():
    with DecisionModel(readout=make_readout(lambda request: httpx.Response(
        200, json=top_logprobs({"A": -0.5, "B": -1.5})
    ))) as model:
        result = model.classify("state", {
            "tags": {"labels": ["rain", "wind"], "multi_label": True},
            "weather": ["wet", "dry"],
        }, probability=True, include_probabilities=True, calibrated=False)
    assert result.usage["logical_reads"] == 3
    assert len(result["tags"]) == 2
    assert result["weather"]["label"] == "wet"
    assert sum(result["weather"]["probabilities"].values()) == pytest.approx(1)


@pytest.mark.parametrize("method", ["decide", "classify"])
def test_probability_image_retains_single_read_and_uncalibrated_default(method):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response({"A": -0.5, "B": -1.5}))

    with DecisionModel(readout=make_image_readout(handler)) as model:
        result = getattr(model, method)("state", {"forecast": Noul("Will it rain?")},
                                        probability=True, image=IMAGE_URL)
    assert len(calls) == result.usage["logical_reads"] == 1
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert result.answers["forecast"].calibrated is False
    assert result.meta["probability"] is True


# -- lazy tokenizer initialization --------------------------------------------

@pytest.mark.parametrize("separate_readouts", [False, True])
def test_tokenizer_initialization_is_serialized(monkeypatch, separate_readouts):
    import shisa_de.readout as readout_module

    second_attempt = Event()

    class ObservedLock:
        def __init__(self):
            self.lock = Lock()
            self.counter_lock = Lock()
            self.attempts = 0

        def __enter__(self):
            with self.counter_lock:
                self.attempts += 1
                if self.attempts == 2:
                    second_attempt.set()
            self.lock.acquire()

        def __exit__(self, *exc):
            self.lock.release()

    init_lock = ObservedLock()
    monkeypatch.setattr(readout_module, "_TOKENIZER_INIT_LOCK", init_lock)
    calls = []

    def load(source, **kwargs):
        assert init_lock.lock.locked()
        # Hold the first load until another caller reaches the shared lock.
        # No sleeps or assumptions about thread scheduling are needed.
        assert second_attempt.wait(timeout=5)
        tokenizer = StubTokenizer()
        calls.append((source, kwargs, tokenizer))
        return tokenizer

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load),
    ))
    with ExitStack() as stack:
        first = stack.enter_context(Readout("http://test.local", "test-model"))
        second = (stack.enter_context(Readout("http://test.local", "other-model"))
                  if separate_readouts else first)
        assert first._tokenizer is None
        assert second._tokenizer is None
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(readout.ensure_tokenizer) for readout in (first, second)]
            results = [future.result(timeout=10) for future in futures]
        assert len(calls) == (2 if separate_readouts else 1)
        assert results[0] is first._tokenizer
        assert results[1] is second._tokenizer
        assert (results[0] is results[1]) is not separate_readouts
        assert first.ensure_tokenizer() is results[0]
        assert second.ensure_tokenizer() is results[1]
        assert init_lock.attempts == 2  # Cached calls do not acquire the lock.


def test_tokenizer_initialization_retries_after_failure(monkeypatch):
    calls = []
    tokenizer = StubTokenizer()

    def load(source, **kwargs):
        calls.append((source, kwargs))
        if len(calls) == 1:
            raise OSError("tokenizer unavailable")
        return tokenizer

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load),
    ))
    with Readout("http://test.local", "test-model", tokenizer="custom-tokenizer",
                 tokenizer_revision="revision", local_files_only=True) as readout:
        assert calls == []
        with pytest.raises(OSError, match="tokenizer unavailable"):
            readout.ensure_tokenizer()
        assert readout._tokenizer is None
        assert readout.ensure_tokenizer() is tokenizer
        assert readout.ensure_tokenizer() is tokenizer
    assert calls == [("custom-tokenizer", {"revision": "revision", "local_files_only": True})] * 2


@pytest.mark.parametrize("method", ["decide", "classify"])
def test_multi_question_first_call_loads_tokenizer_once(monkeypatch, method):
    calls = []

    def load(source, **kwargs):
        calls.append(source)
        return StubTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load),
    ))
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=top_logprobs({"A": -0.1, "B": -3.0}))
    )
    with DecisionModel(transport=transport) as model:
        result = getattr(model, method)("state", {
            "route": Choice("Choose a route", {"a": None, "b": None}),
            "urgent": Noul("Is this urgent?"),
        })
        assert result["route"] == "a"
        assert result["urgent"] > 0.5
        assert result.usage["requests"] == 2
    assert len(calls) == 1


# -- hosted configuration and health ------------------------------------------

@pytest.mark.parametrize("constructor", [DecisionModel, DecisionModel.from_pretrained])
@pytest.mark.parametrize("key_source", ["platform", "de", "explicit"])
def test_hosted_defaults_and_api_key_precedence(monkeypatch, constructor, key_source):
    monkeypatch.delenv("SHISA_DE_ENDPOINT", raising=False)
    monkeypatch.delenv("SHISA_DE_API_KEY", raising=False)
    monkeypatch.setenv("SHISA_API_KEY", "platform-key")
    kwargs = {}
    if key_source in {"de", "explicit"}:
        monkeypatch.setenv("SHISA_DE_API_KEY", "de-key")
    if key_source == "explicit":
        kwargs["api_key"] = "explicit-key"
    calls = []

    def handler(request):
        calls.append(request)
        assert str(request.url) == "https://api.shisa.ai/openai/v1/completions"
        assert request.headers["Authorization"] == f"Bearer {key_source}-key"
        assert json.loads(request.content)["model"] == "shisa-ai/shisa-de-1"
        return httpx.Response(200, json=top_logprobs({"A": -0.1, "B": -3.0}))

    with constructor(transport=httpx.MockTransport(handler), **kwargs) as model:
        assert model.readout._tokenizer is None
        assert model.readout.tokenizer_source == "shisa-ai/shisa-de-1"
        model.readout._tokenizer = StubTokenizer()
        result = model.classify("Win a prize!", {"intent": ["spam", "ham"]})
        assert result["intent"] == "spam"
    assert len(calls) == 1


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("suffix", ["/", "/v1", "/v1/"])
def test_endpoint_overrides_preserve_custom_model(monkeypatch, explicit, suffix):
    monkeypatch.setenv("SHISA_DE_ENDPOINT", "http://env.local" + suffix)
    kwargs = {"base_url": "http://explicit.local" + suffix} if explicit else {}
    expected = "http://explicit.local" if explicit else "http://env.local"

    def handler(request):
        assert str(request.url) == expected + "/v1/completions"
        assert json.loads(request.content)["model"] == "custom-model"
        return httpx.Response(200, json=top_logprobs({"A": -0.1, "B": -3.0}))

    with DecisionModel(model="custom-model", family="de1",
                       transport=httpx.MockTransport(handler), **kwargs) as model:
        assert model.base_url == expected
        model.readout._tokenizer = StubTokenizer()
        assert model.classify("state", {"intent": ["a", "b"]})["intent"] == "a"


def test_local_endpoint_can_disable_environment_authentication(monkeypatch):
    monkeypatch.setenv("SHISA_API_KEY", "hosted-key")
    monkeypatch.setenv("SHISA_DE_API_KEY", "de-key")
    calls = []

    def handler(request):
        calls.append(request)
        assert "Authorization" not in request.headers
        assert str(request.url) == "http://localhost:8021/v1/completions"
        assert json.loads(request.content)["model"] == "local-alias"
        return httpx.Response(200, json=top_logprobs({"A": -0.1, "B": -3.0}))

    with DecisionModel.from_endpoint(
        "http://localhost:8021/v1", model="local-alias", api_key="",
        tokenizer="shisa-ai/shisa-de-1", transport=httpx.MockTransport(handler),
    ) as model:
        assert model.readout.tokenizer_source == "shisa-ai/shisa-de-1"
        model.readout._tokenizer = StubTokenizer()
        assert model.classify("state", {"intent": ["a", "b"]})["intent"] == "a"
    assert len(calls) == 1


@pytest.mark.parametrize("status,listed,boundary_ok", [
    (200, True, True),
    (200, False, True),
    (401, False, True),
    (503, False, True),
    (200, True, False),
])
def test_health_requires_model_access_and_boundary(monkeypatch, status, listed, boundary_ok):
    def handler(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(status, json={"data": [{"id": "test-model"}] if listed else []})

    with DecisionModel(base_url="http://test.local", model="test-model", family="de1",
                       transport=httpx.MockTransport(handler)) as model:
        model.readout._tokenizer = StubTokenizer()
        if not boundary_ok:
            def fail_boundary(*args):
                raise ReadoutError("answer boundary moves")
            monkeypatch.setattr(model.readout, "check_boundary", fail_boundary)
        report = model.health()
    assert report["ok"] is (status == 200 and listed and boundary_ok)
    assert (report["boundary_check"] == "passed") is boundary_ok


# -- command line ------------------------------------------------------------

def test_cli_ask_uses_the_supplied_prompt(monkeypatch, capsys):
    from shisa_de import cli

    def handler(request):
        prompt = json.loads(request.content)["prompt"]
        payload = json.loads(prompt.split("<user>", 1)[1].split("<model>")[0])
        assert payload["criterion"] == "Is this spam?"
        return httpx.Response(200, json=top_logprobs({"A": -0.1, "B": -3.0}))

    model = DecisionModel(transport=httpx.MockTransport(handler))
    model.readout._tokenizer = StubTokenizer()
    monkeypatch.setattr(cli, "DecisionModel", lambda **kwargs: model)
    assert cli.main(["ask", "--state", "prize", "--labels", "spam,ham",
                     "--prompt", "Is this spam?"]) == 0
    assert json.loads(capsys.readouterr().out) == {"intent": "spam"}


@pytest.mark.parametrize("authenticated", [False, True])
def test_cli_explain_curl_uses_environment_references_not_secrets(monkeypatch, capsys, authenticated):
    from shisa_de import cli

    monkeypatch.delenv("SHISA_API_KEY", raising=False)
    monkeypatch.delenv("SHISA_DE_API_KEY", raising=False)
    if authenticated:
        monkeypatch.setenv("SHISA_API_KEY", "secret-not-for-output")
    model = DecisionModel(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=top_logprobs({"A": -0.1, "B": -3.0}))
    ))
    model.readout._tokenizer = StubTokenizer()
    monkeypatch.setattr(cli, "DecisionModel", lambda **kwargs: model)
    assert cli.main(["explain", "--labels", "spam,ham"]) == 0
    output = capsys.readouterr().out
    assert "secret-not-for-output" not in output
    assert ('Authorization: Bearer ${SHISA_DE_API_KEY:-$SHISA_API_KEY}' in output) is authenticated


def test_cli_ask_with_an_image_sends_the_prompt_and_prints_no_image_or_key(tmp_path, monkeypatch, capsys):
    from shisa_de import cli

    payload = b"\x89PNG\r\n\x1a\n" + b"pretend image bytes"
    path = tmp_path / "cat.png"
    path.write_bytes(payload)
    monkeypatch.setenv("SHISA_API_KEY", "secret-not-for-output")
    calls: list[dict] = []
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response({"A": -0.1, "B": -3.0}))

    def factory(**kwargs):
        """Build the model the way the CLI does, so the flag wiring is exercised."""
        captured.update(kwargs)
        built = DecisionModel(transport=httpx.MockTransport(handler), **kwargs)
        built.readout._tokenizer = ImageStubTokenizer()
        return built

    monkeypatch.setattr(cli, "DecisionModel", factory)
    assert cli.main(["ask", "--base-url", "http://test.local",
                     "--image", str(path), "--prompt", "Is this spam?",
                     "--labels", "spam,ham", "--image-top-logprobs", "48"]) == 0

    output = capsys.readouterr().out
    assert json.loads(output) == {"intent": "spam"}
    assert "secret-not-for-output" not in output
    assert "cat.png" not in output
    assert base64.b64encode(payload).decode() not in output

    # The image really was sent, so the checks above are not vacuous.
    sent = json.dumps(calls[0])
    assert base64.b64encode(payload).decode() in sent
    assert captured["image_top_logprobs"] == 48
    assert calls[0]["top_logprobs"] == 48
    question = json.loads(calls[0]["messages"][1]["content"][1]["text"])
    assert question["criterion"] == "Is this spam?"
    assert question["evidence"] == {}  # an image call with no --state sends no invented state


# -- questions ---------------------------------------------------------------

def test_wire_shapes_match_the_system_one_api():
    assert Noul("Is this urgent?").to_wire() == {"type": "noul", "instructions": "Is this urgent?"}
    assert Noul("Is this urgent?", criteria={"true": "time sensitive"}).to_wire()["criteria"] == {"true": "time sensitive"}
    assert Choice("Which queue?", {"billing": "Charges"}).to_wire() == {
        "type": "choice", "instructions": "Which queue?", "criteria": {"billing": "Charges"}
    }
    assert Score("How urgent?", ["low", "high"]).to_wire() == {
        "type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]
    }


def test_option_without_a_description_shows_its_key():
    assert render_option("refund_request", None) == "refund_request"
    assert render_option("refund_request", "   ") == "refund_request"
    assert render_option("refund_request", "Money back") == "Money back"


def test_question_limits():
    with pytest.raises(QuestionError):
        Choice("Too many?", {f"label{index}": None for index in range(MAX_OPTIONS + 1)}).validate()
    with pytest.raises(QuestionError):
        Noul("   ").validate()
    Choice("Fine?", {f"label{index}": None for index in range(MAX_OPTIONS)}).validate()


def test_noul_options_are_yes_and_no():
    assert Noul("Is this true?").options() == [("yes", "Yes"), ("no", "No")]


# -- rendering ---------------------------------------------------------------

def test_prompt_carries_the_state_the_question_and_lettered_options():
    readout = make_readout(lambda request: httpx.Response(200, json=top_logprobs({"A": -0.1, "B": -3.0})))
    prompt = readout.render({"ticket": "late"}, Choice("Which queue?", {"billing": None, "logistics": "Delivery"}))
    payload = json.loads(prompt.split("<user>", 1)[1].split("<model>")[0])
    assert payload["evidence"] == {"ticket": "late"}
    assert payload["criterion"] == "Which queue?"
    assert payload["options"] == [
        {"letter": "A", "description": "billing"},
        {"letter": "B", "description": "Delivery"},
    ]
    assert "only its uppercase letter" in prompt


def test_every_letter_is_a_single_token_and_the_boundary_holds():
    readout = make_readout(lambda request: httpx.Response(200, json=top_logprobs({"A": -0.1, "B": -3.0})))
    slots = readout.slots(MAX_OPTIONS)
    assert [slot.letter for slot in slots] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    prompt = readout.render("state", Choice("q", {f"l{index}": None for index in range(MAX_OPTIONS)}))
    readout.check_boundary(prompt, slots)


def test_a_tokenizer_that_disagrees_fails_loudly():
    class MergingTokenizer(StubTokenizer):
        """A tokenizer where `prompt + A` merges into a different token."""

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            ids = [ord(character) for character in text]
            if len(ids) > 1 and text.endswith("A"):
                return ids[:-1] + [0]
            return ids

    readout = make_readout(lambda request: httpx.Response(200, json=top_logprobs({"A": -0.1, "B": -3.0})))
    readout._tokenizer = MergingTokenizer()
    with pytest.raises(ReadoutError, match="answer boundary moves"):
        readout.read("prompt", 2)


# -- reading -----------------------------------------------------------------

def test_one_request_when_every_letter_is_in_the_top_k():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=top_logprobs({"A": -0.5, "B": -1.5, "C": -4.0}))

    readout = make_readout(handler)
    read = readout.read("prompt", 3)
    assert len(calls) == 1
    assert read.requests == 1
    assert read.missing_from_top == []
    assert read.answer() == "A"
    assert math.isclose(sum(read.probabilities.values()), 1.0)
    assert calls[0]["max_tokens"] == 1 and calls[0]["temperature"] == 0 and calls[0]["logprobs"] == 20


def test_a_letter_outside_the_top_k_costs_one_fallback_request():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "prompt_logprobs" in body:
            assert body["prompt"].endswith("C")
            return httpx.Response(200, json={
                "choices": [{"prompt_logprobs": [None, {"67": {"logprob": -6.0, "rank": 41, "decoded_token": "C"}}]}],
                "usage": {"prompt_tokens": 13},
            })
        return httpx.Response(200, json=top_logprobs({"A": -0.5, "B": -1.5}))

    readout = make_readout(handler)
    read = readout.read("prompt", 3)
    assert len(calls) == 2
    assert read.requests == 2
    assert read.missing_from_top == ["C"]
    assert read.ranks == {"C": 41}
    assert read.logprobs["C"] == -6.0
    assert read.answer() == "A"
    assert read.prompt_tokens == 25  # both requests are charged


def test_tokens_outside_the_option_list_are_discarded():
    readout = make_readout(lambda request: httpx.Response(200, json=top_logprobs({"!": 0.0, "A": -1.0, "B": -2.0})))
    read = readout.read("prompt", 2)
    assert set(read.probabilities) == {"A", "B"}
    assert math.isclose(read.probabilities["A"], math.e ** -1 / (math.e ** -1 + math.e ** -2))


def test_http_errors_are_reported_with_the_body():
    readout = make_readout(lambda request: httpx.Response(400, json={"error": {"message": "nope"}}))
    with pytest.raises(ReadoutError, match="HTTP 400"):
        readout.read("prompt", 2)


def test_softmax_rejects_an_empty_distribution():
    with pytest.raises(ReadoutError):
        softmax({})


# -- images ------------------------------------------------------------------

IMAGE_URL = "data:image/png;base64,AAAA"
ANSWER_TOKEN_ID = "101"


class ImageStubTokenizer(StubTokenizer):
    """A stub where the generation-prompt terminator is one token.

    The image readout confirms the answer boundary from the server's side, so
    the terminator has to resolve to exactly one token the way it does in the
    checkpoint's tokenizer.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == ANSWER_PREFIX:
            return [int(ANSWER_TOKEN_ID)]
        return super().encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids) -> str:
        if list(ids) == [int(ANSWER_TOKEN_ID)]:
            return ANSWER_PREFIX
        return super().decode(ids)


def chat_response(
    entries: dict[str, float],
    *,
    sampled: str | None = None,
    prompt_token: str | None = ANSWER_PREFIX,
    prompt_token_id: str = ANSWER_TOKEN_ID,
    prompt_tokens: int = 512,
) -> dict:
    """A `/v1/chat/completions` body in the shape the hosted probe returned."""
    best = sampled if sampled is not None else max(entries, key=entries.get)
    prompt_logprobs: list = [None]
    if prompt_token is not None:
        prompt_logprobs.append({prompt_token_id: {"logprob": -0.02, "decoded_token": prompt_token}})
    return {
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": best},
            "logprobs": {"content": [{
                "token": best,
                "top_logprobs": [{"token": token, "logprob": value} for token, value in entries.items()],
            }]},
            "finish_reason": "length",
        }],
        "prompt_logprobs": prompt_logprobs,
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1,
                  "total_tokens": prompt_tokens + 1},
    }


def broken(mutate) -> dict:
    """A valid chat response with one part damaged."""
    response = chat_response({"A": -0.1, "B": -3.0})
    mutate(response)
    return response


def make_image_readout(handler, **kwargs) -> Readout:
    readout = make_readout(handler, **kwargs)
    readout._tokenizer = ImageStubTokenizer()
    return readout


def image_question(count: int = 2) -> Choice:
    return Choice("Is this spam?", {f"label{index}": None for index in range(count)})


# -- preparing an image ------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://example.com/cat.png",
    "http://example.com/cat.jpg",
    "data:image/webp;base64,AAAA",
])
def test_urls_pass_through_unchanged(url):
    assert prepare_image(url) == url


def test_a_local_png_becomes_a_base64_data_url(tmp_path):
    payload = b"\x89PNG\r\n\x1a\n" + b"pretend image bytes"
    path = tmp_path / "cat.png"
    path.write_bytes(payload)
    url = prepare_image(path)
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == payload


def test_a_local_file_is_sent_as_what_it_is_not_what_it_is_called(tmp_path):
    path = tmp_path / "mislabelled.jpg"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"png bytes")
    assert prepare_image(path).startswith("data:image/png;base64,")


def test_webp_and_jpeg_are_supported(tmp_path):
    webp = tmp_path / "shot.webp"
    webp.write_bytes(b"RIFF\x00\x00\x00\x00WEBP" + b"payload")
    assert prepare_image(webp).startswith("data:image/webp;base64,")
    jpeg = tmp_path / "shot.jpeg"
    jpeg.write_bytes(b"\xff\xd8\xff\xe0" + b"payload")
    assert prepare_image(jpeg).startswith("data:image/jpeg;base64,")


def test_a_string_path_is_accepted_the_same_way(tmp_path):
    path = tmp_path / "cat.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"bytes")
    assert prepare_image(str(path)) == prepare_image(path)


def test_prepare_image_rejects_bad_values_without_a_request(tmp_path):
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    not_an_image = tmp_path / "fake.png"
    not_an_image.write_bytes(b"this is not an image")

    for value, expected in [
        ("", "empty"),
        ("   ", "empty"),
        ("photo.gif", "extension"),
        ("photo.bmp", "extension"),
        ("ftp://example.com/cat.png", "scheme"),
        ("file:///tmp/cat.png", "scheme"),
    ]:
        with pytest.raises(ImageError, match=expected):
            prepare_image(value)
    with pytest.raises(ImageError, match="not found"):
        prepare_image(tmp_path / "missing.png")
    with pytest.raises(ImageError, match="empty"):
        prepare_image(empty)
    with pytest.raises(ImageError, match="not a PNG"):
        prepare_image(not_an_image)
    with pytest.raises(ImageError, match="file path or a URL string"):
        prepare_image(b"bytes")


def test_a_local_file_name_is_not_a_url(tmp_path):
    path = tmp_path / "cat.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"bytes")
    with pytest.raises(ImageError, match="prepare_image"):
        validate_image_url(str(path))
    with pytest.raises(ImageError, match="empty"):
        validate_image_url("   ")
    with pytest.raises(ImageError, match="not an image"):
        validate_image_url("data:text/plain;base64,AAAA")
    with pytest.raises(ImageError, match="no payload"):
        validate_image_url("data:image/png;base64,")
    with pytest.raises(ImageError, match="no host"):
        validate_image_url("https://")


# -- reading an image --------------------------------------------------------

def test_the_image_request_matches_the_probed_shape():
    calls: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=chat_response({"A": -0.1, "B": -3.0, "1": -9.0}))

    readout = make_image_readout(handler)
    read = readout.read_image({"sms": "hi"}, Choice("Is this spam?", {"spam": None, "ham": None}), IMAGE_URL)

    assert len(calls) == 1
    path, body = calls[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "test-model"
    assert body["max_tokens"] == 1
    assert body["temperature"] == 0
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 20
    assert body["prompt_logprobs"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}

    system, user = body["messages"]
    assert system == {"role": "system", "content": DIRECT_SYSTEM}
    assert user["role"] == "user"
    image_part, text_part = user["content"]
    assert image_part == {"type": "image_url", "image_url": {"url": IMAGE_URL}}
    assert text_part["type"] == "text"
    payload = json.loads(text_part["text"])
    assert payload["evidence"] == {"sms": "hi"}
    assert payload["criterion"] == "Is this spam?"
    assert payload["options"] == [
        {"letter": "A", "description": "spam"},
        {"letter": "B", "description": "ham"},
    ]

    assert read.requests == 1
    assert read.prompt_tokens == 512
    assert read.sampled == "A"
    assert read.missing_from_top == []
    assert read.answer() == "A"
    assert set(read.logprobs) == {"A", "B"}
    assert math.isclose(sum(read.probabilities.values()), 1.0)


def test_the_image_readout_reads_the_same_distribution_as_the_text_readout():
    readout = make_image_readout(lambda request: httpx.Response(200, json=chat_response({"A": -0.5, "B": -1.5})))
    read = readout.read_image("state", image_question(), IMAGE_URL)
    assert read.probabilities["A"] == pytest.approx(math.e ** -0.5 / (math.e ** -0.5 + math.e ** -1.5))


def test_the_answer_boundary_token_must_be_one_token():
    readout = make_image_readout(lambda request: httpx.Response(200, json=chat_response({"A": -0.1, "B": -3.0})))
    assert readout.answer_prefix_id() == int(ANSWER_TOKEN_ID)

    class MergingTokenizer(ImageStubTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            if text == ANSWER_PREFIX:
                return [1, 2]
            return super().encode(text, add_special_tokens=add_special_tokens)

    readout._tokenizer = MergingTokenizer()
    with pytest.raises(ReadoutError, match="not one token"):
        readout.answer_prefix_id()


def test_a_wrong_last_prompt_token_fails_the_boundary_check():
    readout = make_image_readout(lambda request: httpx.Response(
        200, json=chat_response({"A": -0.1, "B": -3.0}, prompt_token="<bos>")))
    with pytest.raises(ReadoutError, match="not read at the answer boundary"):
        readout.read_image("state", image_question(), IMAGE_URL)


def test_a_boundary_token_from_another_tokenizer_fails():
    readout = make_image_readout(lambda request: httpx.Response(
        200, json=chat_response({"A": -0.1, "B": -3.0}, prompt_token_id="999")))
    with pytest.raises(ReadoutError, match="not read at the answer boundary"):
        readout.read_image("state", image_question(), IMAGE_URL)


def test_a_missing_boundary_entry_fails():
    for response in [
        broken(lambda r: r.pop("prompt_logprobs")),
        broken(lambda r: r.update({"prompt_logprobs": [None]})),
        broken(lambda r: r["prompt_logprobs"].append(None)),
    ]:
        readout = make_image_readout(lambda request: httpx.Response(200, json=response))
        with pytest.raises(ReadoutError, match="boundary"):
            readout.read_image("state", image_question(), IMAGE_URL)


def test_an_option_outside_the_returned_top_k_is_a_hard_error():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response({"A": -0.5, "B": -1.5}))

    readout = make_image_readout(handler)
    with pytest.raises(ReadoutError) as excinfo:
        readout.read_image("state", image_question(3), IMAGE_URL)
    message = str(excinfo.value)
    assert "C" in message
    assert "no letter fallback" in message
    assert "max-logprobs" in message
    assert len(calls) == 1  # no fallback request, and no renormalized answer


def test_more_options_than_requested_logprobs_fails_before_any_request():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=chat_response({"A": -0.1, "B": -3.0}))

    readout = make_image_readout(handler)
    with pytest.raises(ReadoutError, match="no letter fallback"):
        readout.read_image("state", image_question(3), IMAGE_URL, top_logprobs=2)
    assert calls == []


def test_a_bad_image_is_rejected_before_any_request():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=chat_response({"A": -0.1, "B": -3.0}))

    readout = make_image_readout(handler)
    with pytest.raises(ImageError, match="prepare_image"):
        readout.read_image("state", image_question(), "cat.png")
    with pytest.raises(ReadoutError, match="positive integer"):
        readout.read_image("state", image_question(), IMAGE_URL, top_logprobs=0)
    assert calls == []


@pytest.mark.parametrize("response,match", [
    ({"choices": []}, "no choices"),
    (broken(lambda r: r["choices"][0].update({"logprobs": None})), "no logprobs.content"),
    (broken(lambda r: r["choices"][0].update({"logprobs": ["invalid"]})), "no logprobs.content"),
    (broken(lambda r: r["choices"][0].update({"logprobs": {"content": []}})), "no logprobs.content"),
    (broken(lambda r: r["choices"][0]["logprobs"]["content"][0].update({"top_logprobs": []})), "no top_logprobs"),
    (broken(lambda r: r["choices"][0]["logprobs"]["content"][0].update({"top_logprobs": [{"token": "A"}]})), "malformed"),
    (broken(lambda r: r["choices"][0]["logprobs"]["content"][0].update({"top_logprobs": [{"logprob": -1.0}]})), "malformed"),
    (chat_response({"A": float("nan"), "B": -1.0}), "malformed"),
    (chat_response({"A": float("inf"), "B": -1.0}), "malformed"),
])
def test_malformed_chat_responses_are_reported(response, match):
    readout = make_image_readout(lambda request: httpx.Response(200, text=json.dumps(response)))
    with pytest.raises(ReadoutError, match=match):
        readout.read_image("state", image_question(), IMAGE_URL)


def test_a_non_json_or_failed_chat_response_is_reported():
    readout = make_image_readout(lambda request: httpx.Response(200, text="<html>not json</html>"))
    with pytest.raises(ReadoutError, match="did not return JSON"):
        readout.read_image("state", image_question(), IMAGE_URL)

    readout = make_image_readout(lambda request: httpx.Response(400, json={"error": {"message": "nope"}}))
    with pytest.raises(ReadoutError, match="HTTP 400"):
        readout.read_image("state", image_question(), IMAGE_URL)


def test_debug_carries_the_messages_and_the_full_top_logprobs():
    handler = lambda request: httpx.Response(200, json=chat_response({"A": -0.1, "B": -3.0, "1": -9.0}))
    readout = make_image_readout(handler)
    plain = readout.read_image("state", image_question(), IMAGE_URL)
    assert plain.messages is None
    assert plain.top_logprobs == {}
    assert plain.prompt is None  # the server rendered the prompt on this path

    debug = readout.read_image("state", image_question(), IMAGE_URL, debug=True)
    assert debug.messages[1]["content"][0] == {"type": "image_url", "image_url": {"url": IMAGE_URL}}
    assert debug.top_logprobs["1"] == -9.0  # non-letters are visible, never answers
    assert set(debug.probabilities) == {"A", "B"}


def test_the_image_never_enters_the_evidence():
    state = {"note": "a red square"}
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response({"A": -0.1, "B": -3.0}))

    readout = make_image_readout(handler)
    readout.read_image(state, image_question(), IMAGE_URL)
    assert state == {"note": "a red square"}  # the caller's state is not mutated
    text_part = calls[0]["messages"][1]["content"][1]
    assert json.loads(text_part["text"])["evidence"] == {"note": "a red square"}


# -- calibration -------------------------------------------------------------

def test_confidence_matches_the_documented_formula():
    assert math.isclose(confidence({"a": 0.88, "b": 0.12, "c": 0.0}), (3 * 0.88 - 1) / 2)
    assert confidence({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}) == 0.0
    assert confidence({"a": 1.0, "b": 0.0}) == 1.0
    assert confidence({"a": 0.9, "b": 0.1}) == pytest.approx(0.8)


def test_tempering_preserves_the_argmax_and_moves_toward_the_middle():
    probabilities = {"a": 0.9, "b": 0.1}
    tempered = temper_distribution(probabilities, 1.9)
    assert max(tempered, key=tempered.get) == "a"
    assert tempered["a"] < probabilities["a"]
    assert temper_binary(0.9, 1.9) < 0.9
    assert temper_binary(0.9, 1.0) == 0.9
    assert math.isclose(sum(temper_distribution(probabilities, 2.5).values()), 1.0)


# -- head parsing ------------------------------------------------------------

def test_label_list_becomes_a_choice_with_a_generated_question():
    question, options = _question_from_head("intent", ["refund", "cancel"])
    assert isinstance(question, Choice)
    assert question.instructions == "What is the intent?"
    assert list(question.criteria) == ["refund", "cancel"]
    assert options == {"labels": ["refund", "cancel"]}


def test_described_labels_and_an_explicit_prompt():
    question, _ = _question_from_head("answer", {"labels": {"yes": None, "no": None}, "prompt": "Did it enter force?"})
    assert question.instructions == "Did it enter force?"
    assert question.criteria == {"yes": None, "no": None}


def test_levels_become_a_score():
    question, options = _question_from_head("severity", {"levels": ["low", "medium", "high"]})
    assert isinstance(question, Score)
    assert options == {"levels": ["low", "medium", "high"]}


def test_multi_label_expands_to_one_noul_per_label():
    question, options = _question_from_head("topics", {"labels": ["hvac", "billing"], "multi_label": True, "cls_threshold": 0.4})
    assert isinstance(question, _MultiLabel)
    assert options["cls_threshold"] == 0.4
    assert question.noul_for("hvac").instructions == "Is hvac one of the topics?"


def test_a_question_object_passes_through():
    original = Noul("Is this spam?")
    question, _ = _question_from_head("spam", original)
    assert question is original


def test_a_bare_string_is_rejected():
    with pytest.raises(QuestionError, match="bare string"):
        _question_from_head("intent", "refund")


# -- the client --------------------------------------------------------------

class FakeReadout:
    """Returns a distribution that favours a chosen letter, and counts calls."""

    def __init__(self, favourite: dict[str, str] | None = None, noul: float = 0.9) -> None:
        self.favourite = favourite or {}
        self.noul = noul
        self.calls: list[str] = []

    def evaluate(self, state, question, debug=False):
        self.calls.append(question.instructions)
        options = question.options()
        if question.type == "noul":
            logprobs = {"A": math.log(self.noul), "B": math.log(1 - self.noul)}
        else:
            target = self.favourite.get(question.instructions, "A")
            letters = [chr(ord("A") + index) for index in range(len(options))]
            logprobs = {letter: (0.0 if letter == target else -3.0) for letter in letters}
        read = LetterRead(logprobs=logprobs, probabilities=softmax(logprobs), requests=1, prompt_tokens=10)
        return read, options

    def close(self) -> None:
        pass


def test_classify_returns_a_label_by_default():
    model = DecisionModel(readout=FakeReadout())
    result = model.classify({"ticket": "late"}, {"intent": ["refund", "cancel"]})
    assert isinstance(result, Decision)
    assert result["intent"] == "refund"
    assert result.answers["intent"].confidence > 0.6


def test_classify_can_return_confidence_and_probabilities():
    model = DecisionModel(readout=FakeReadout())
    with_confidence = model.classify({"ticket": "late"}, {"intent": ["refund", "cancel"]}, include_confidence=True)
    assert with_confidence["intent"]["label"] == "refund"
    assert 0.0 <= with_confidence["intent"]["confidence"] <= 1.0
    with_probabilities = model.classify({"ticket": "late"}, {"intent": ["refund", "cancel"]}, include_probabilities=True)
    assert math.isclose(sum(with_probabilities["intent"]["probabilities"].values()), 1.0)


def test_classify_on_an_ordered_scale_returns_the_level():
    model = DecisionModel(readout=FakeReadout())
    result = model.classify({"ticket": "late"}, {"severity": {"levels": ["low", "high"]}})
    assert result["severity"] == "low"
    assert result.answers["severity"].score == pytest.approx(0.171, abs=1e-3)
    raw = model.classify({"ticket": "late"}, {"severity": {"levels": ["low", "high"]}}, calibrated=False)
    assert raw.answers["severity"].score == pytest.approx(0.047, abs=1e-3)


def test_classify_multi_label_returns_the_labels_above_the_threshold():
    model = DecisionModel(readout=FakeReadout(noul=0.9))
    result = model.classify(
        {"ticket": "late"}, {"topics": {"labels": ["hvac", "billing"], "multi_label": True, "cls_threshold": 0.4}}
    )
    assert result["topics"] == ["hvac", "billing"]
    strict = DecisionModel(readout=FakeReadout(noul=0.2))
    assert strict.classify(
        {"ticket": "late"}, {"topics": {"labels": ["hvac"], "multi_label": True, "cls_threshold": 0.4}}
    )["topics"] == []


def test_decide_returns_typed_answers_and_the_wire_shape():
    model = DecisionModel(readout=FakeReadout(noul=0.93))
    result = model.decide(
        {"ticket": "late"},
        {"refund": Noul("Does the customer ask for a refund?"), "route": Choice("Which queue?", {"billing": None, "logistics": None})},
        calibrated=False,
    )
    assert result["refund"] == pytest.approx(0.93, abs=0.01)
    assert result["route"] == "billing"
    assert result.nouls["refund"].type == "noul"
    assert result.choices["route"].type == "choice"
    wire = result.to_wire()
    assert wire["answers"]["refund"]["type"] == "noul"
    assert wire["answers"]["route"]["choice"] == "billing"
    assert set(wire["usage"]) == {"input_tokens", "output_tokens"}


def test_every_answer_reports_what_it_cost():
    model = DecisionModel(readout=FakeReadout())
    result = model.classify({"ticket": "late"}, {"intent": ["a", "b"], "queue": ["x", "y"]})
    assert result.usage["requests"] == 2
    assert result.usage["input_tokens"] == 20
    assert result.meta["readout_version"]
    assert result.meta["calibration"]


def test_calibration_is_applied_by_default_and_can_be_turned_off():
    model = DecisionModel(readout=FakeReadout())
    calibrated = model.classify({"ticket": "late"}, {"intent": ["a", "b"]}, include_probabilities=True)
    raw = model.classify({"ticket": "late"}, {"intent": ["a", "b"]}, include_probabilities=True, calibrated=False)
    assert calibrated["intent"]["probabilities"]["a"] < raw["intent"]["probabilities"]["a"]
    assert calibrated.answers["intent"].calibrated is True
    assert raw.answers["intent"].calibrated is False


# -- DE-1 vs DE-2: family, readout identity, calibration --------------------

@pytest.mark.parametrize("model_id,expected,explicit", [
    ("shisa-ai/shisa-de-1", "de1", True),
    ("de-1", "de1", True),
    ("de1-cont-v1-lr2e5-s7", "de1", True),
    ("shisa-ai/shisa-de-2", "de2", True),
    ("de2-v4-lr5e5-e3-s7", "de2", True),
    # No slug: DE-2 is assumed, and reported as an assumption.
    ("gemma-4-26b-a4b-it", "de2", False),
    ("w2-lr1e-4-s13", "de2", False),
    # A slug has to be its own token, so these are not read as DE-1.
    ("de-13", "de2", False),
    ("de1x", "de2", False),
    # An id that names both families names neither.
    ("de-2-vs-de-1-ablation", "de2", False),
])
def test_model_family_defaults_to_de2_without_a_de1_slug(model_id, expected, explicit):
    assert model_family(model_id) == expected
    assert family_is_explicit(model_id) is explicit


@pytest.mark.parametrize("model_id,tokenizer,declared,expected", [
    ("shisa-ai/shisa-de-1", None, None, ("de1", "model id")),
    ("local-alias", "shisa-ai/shisa-de-1", None, ("de1", "tokenizer")),
    ("local-alias", "google/gemma-4-26B-A4B-it", None, ("de2", "assumed")),
    ("local-alias", "google/gemma-4-26B-A4B-it", "DE-1", ("de1", "declared")),
    ("shisa-ai/shisa-de-1", None, "de2", ("de2", "declared")),
])
def test_the_family_is_declared_then_read_from_the_id_then_the_tokenizer(model_id, tokenizer, declared, expected):
    assert resolve_family(model_id, tokenizer, declared) == expected


def test_an_assumed_family_warns_and_a_declared_one_does_not(recwarn):
    with pytest.warns(UserWarning, match="names neither DE-1 nor DE-2"):
        assumed = DecisionModel(model="my-arm")
    assert (assumed.family, assumed.family_source, assumed.family_explicit) == ("de2", "assumed", False)
    recwarn.clear()
    declared = DecisionModel(model="my-arm", family="de2")
    assert (declared.family, declared.family_source) == ("de2", "declared")
    assert not recwarn.list
    with pytest.raises(ValueError, match="unknown family"):
        DecisionModel(model="my-arm", family="de3")


def test_each_family_has_one_readout_version_and_its_own_policy():
    assert READOUT_VERSIONS == {"de1": READOUT_VERSION, "de2": DE2_READOUT_VERSION}
    assert READOUT_VERSION != DE2_READOUT_VERSION
    de1 = DecisionModel(model="shisa-ai/shisa-de-1")
    de2 = DecisionModel(model="de2-v4-lr5e5-e3-s7")
    assert (de1.readout_version, de1.policy) == (READOUT_VERSION, "direct")
    assert (de2.readout_version, de2.policy) == (DE2_READOUT_VERSION, "repeat-think")
    assert DecisionModel(model="de2-x", policy="repeat").policy == "repeat"
    # DE-1 was measured on one read; the DE-2 policies are not defined for it.
    with pytest.raises(ValueError, match="not defined for de1"):
        DecisionModel(model="shisa-ai/shisa-de-1", policy="repeat-think")


def test_a_fit_against_an_answer_identical_readout_still_applies():
    """The DE-1 record was fitted against v1. v2 and v3 added paths, not changes to that read."""
    assert calibration_for("de1").readout_version == "de1-letter-slots-v1"
    assert calibration_compatible("de1-letter-slots-v1", READOUT_VERSION)
    assert calibration_compatible(READOUT_VERSION, READOUT_VERSION)
    # Nothing fitted on a DE-1 readout describes the DE-2 one, in either direction.
    assert not calibration_compatible(READOUT_VERSION, DE2_READOUT_VERSION)
    assert not calibration_compatible(DE2_READOUT_VERSION, READOUT_VERSION)
    assert not calibration_compatible("some-other-readout", READOUT_VERSION)
    assert set(CALIBRATION_COMPATIBLE) == set(READOUT_VERSIONS.values())


def test_every_shipped_record_loads_and_applies_to_its_own_family():
    for family, name in CALIBRATION_FILES.items():
        record = load_calibration(name)
        assert record.family == family, name
        # The identity has to carry the fitted model, or a reader of one answer
        # cannot tell which checkpoint the number beside it came from.
        assert record.model and record.model in record.id
        assert record.readout_version in record.id
        level, reasons = record.applicability(record.model, READOUT_VERSIONS[family], family=family)
        assert level in {"checkpoint", "unfitted"} and reasons == [], name


def test_the_de2_record_is_unfitted_and_keeps_the_withheld_fit_as_provenance():
    record = calibration_for("de2")
    assert record.fitted is False
    assert record.temperatures == {"noul": 1.0, "choice": 1.0}
    assert record.readout_version == DE2_READOUT_VERSION
    assert record.withheld["noul"]["nll_optimum"] == 1.36
    assert calibration_for("de1").fitted is True


def test_a_record_applies_to_its_checkpoint_and_says_why_otherwise():
    record = calibration_for("de1")
    assert record.applicability("shisa-ai/shisa-de-1", READOUT_VERSION) == ("checkpoint", [])
    # A server that aliases the model id is recognised by its tokenizer source.
    assert record.applicability("alias", READOUT_VERSION, family="de1",
                                tokenizer="shisa-ai/shisa-de-1") == ("checkpoint", [])

    level, reasons = record.applicability("de1-cont-v1-lr2e5-s7", READOUT_VERSION)
    assert level == "family" and "another checkpoint" in reasons[0]

    level, reasons = record.applicability("de2-v4-lr5e5-e3-s7", DE2_READOUT_VERSION)
    assert level == "mismatch"
    assert any("family" in reason for reason in reasons) and any("readout" in reason for reason in reasons)

    ok, reasons = record.matches("shisa-ai/shisa-de-1", "some-other-readout")
    assert not ok and any("readout" in reason for reason in reasons)
    assert record.matches("de1-cont-v1-lr2e5-s7", READOUT_VERSION) == (True, [])
    # Nothing fitted, so nothing to mismatch on: any DE-2 checkpoint may carry it.
    assert calibration_for("de2").applicability("any-arm", DE2_READOUT_VERSION, family="de2") == ("unfitted", [])


def test_the_bundled_record_is_applied_only_to_the_checkpoint_it_was_fitted_on():
    hosted = DecisionModel(readout=FakeReadout())
    assert hosted.calibration_applied is True
    answer = hosted.decide("s", {"n": Noul("q?")}).answers["n"]
    assert answer.calibrated is True and answer.temperature == 1.69

    with pytest.warns(UserWarning, match="is not applied"):
        other = DecisionModel(model="de1-cont-v1-lr2e5-s7", readout=FakeReadout())
    assert (other.calibration_level, other.calibration_applied) == ("family", False)
    result = other.decide("s", {"n": Noul("q?")})
    assert result.answers["n"].calibrated is False and result.answers["n"].noul == pytest.approx(0.9)
    # The metadata says what the answers carry, not what was asked for.
    assert result.meta["calibrated"] is False

    # Passing the record is the caller vouching for it.
    vouched = DecisionModel(model="de1-cont-v1-lr2e5-s7", readout=FakeReadout(),
                            calibration=calibration_for("de1"))
    assert vouched.calibration_applied is True
    assert vouched.decide("s", {"n": Noul("q?")}).answers["n"].calibrated is True


def test_an_aliased_de1_server_keeps_its_calibration_through_the_tokenizer(recwarn):
    model = DecisionModel(model="local-alias", tokenizer="shisa-ai/shisa-de-1")
    assert (model.family, model.family_source) == ("de1", "tokenizer")
    assert (model.calibration_level, model.calibration_applied) == ("checkpoint", True)
    assert not recwarn.list


@pytest.mark.parametrize("spec,family", [("de1", "de1"), ("DE-1", "de1"), ("de2", "de2"), ("DE2", "de2")])
def test_a_calibration_spec_accepts_common_family_spellings(spec, family):
    assert resolve_calibration(spec).family == family


def test_a_calibration_spec_that_is_neither_a_family_nor_a_file_is_an_error(capsys):
    from shisa_de import cli

    assert resolve_calibration(None) is None and resolve_calibration("auto") is None
    with pytest.raises(ValueError, match="neither a family"):
        resolve_calibration("de-9")
    with pytest.raises(SystemExit):
        cli.main(["doctor", "--calibration", "no/such/record.json"])
    assert "neither a family" in capsys.readouterr().err


def test_health_passes_for_the_default_de1_model():
    """The shipped DE-1 record was fitted against v1 and has to pass under v3."""
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "shisa-ai/shisa-de-1"}]})

    with DecisionModel(base_url="http://test.local", transport=httpx.MockTransport(handler)) as model:
        model.readout._tokenizer = StubTokenizer()
        report = model.health()
    assert report["readout_version"] == READOUT_VERSION
    assert report["calibration_readout_version"] == "de1-letter-slots-v1"
    assert (report["calibration_level"], report["calibration_applied"]) == ("checkpoint", True)
    assert report["calibration_match"] is True and report["ok"] is True


def test_health_fails_when_the_calibration_does_not_match_the_model():
    """A DE-1 record forced onto a DE-2 endpoint fails the report."""
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "de2-v4-lr5e5-e3-s7"}]})

    with DecisionModel(base_url="http://test.local", model="de2-v4-lr5e5-e3-s7",
                       calibration=calibration_for("de1"),
                       transport=httpx.MockTransport(handler)) as model:
        model.readout._tokenizer = CodeStubTokenizer()
        report = model.health()
    assert report["boundary_check"] == "passed"
    assert report["model_listed"] is True
    assert report["calibration_match"] is False
    assert len(report["calibration_mismatch"]) == 2
    assert report["ok"] is False


def test_health_passes_for_de2_and_checks_every_code():
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "de2-v4-lr5e5-e3-s7"}]})

    with DecisionModel(base_url="http://test.local", model="de2-v4-lr5e5-e3-s7",
                       transport=httpx.MockTransport(handler)) as model:
        model.readout._tokenizer = CodeStubTokenizer()
        report = model.health()
    assert (report["model_family"], report["family_source"]) == ("de2", "model id")
    assert (report["readout_version"], report["policy"]) == (DE2_READOUT_VERSION, "repeat-think")
    assert (report["calibration_level"], report["calibration_applied"]) == ("unfitted", False)
    assert report["slots"]["count"] == MAX_CODES
    assert report["calibration_match"] is True and report["ok"] is True


def test_a_checkpoint_from_the_same_family_passes_health_with_a_note():
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "de1-cont-v1-lr2e5-s7"}]})

    with pytest.warns(UserWarning):
        model = DecisionModel(base_url="http://test.local", model="de1-cont-v1-lr2e5-s7",
                              transport=httpx.MockTransport(handler))
    with model:
        model.readout._tokenizer = StubTokenizer()
        report = model.health()
    assert (report["calibration_level"], report["calibration_applied"]) == ("family", False)
    assert "another checkpoint" in report["calibration_note"][0]
    assert report["ok"] is True


# -- the client on an image --------------------------------------------------

def make_image_model(handler, **kwargs) -> DecisionModel:
    model = DecisionModel(base_url="http://test.local", model="test-model", family="de1",
                          transport=httpx.MockTransport(handler), **kwargs)
    model.readout._tokenizer = ImageStubTokenizer()
    return model


def dual_handler(chat: dict | None = None, text: dict | None = None):
    """Answer the chat and the completions paths, so one test can compare them."""
    chat_body = chat if chat is not None else chat_response({"A": -0.1, "B": -3.0})
    text_body = text if text is not None else top_logprobs({"A": -0.1, "B": -3.0})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=chat_body)
        return httpx.Response(200, json=text_body)

    return handler


def test_classify_with_an_image_returns_the_label():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response({"A": -0.1, "B": -3.0}))

    with make_image_model(handler) as model:
        result = model.classify({"note": "a red square"}, {"color": ["red", "blue"]}, image=IMAGE_URL)
    assert result["color"] == "red"
    assert result.answers["color"].type == "choice"
    assert result.answers["color"].requests == 1
    assert result.meta["input_type"] == "image"
    assert len(calls) == 1


def test_decide_with_an_image_returns_typed_answers():
    with make_image_model(dual_handler()) as model:
        result = model.decide(
            {"note": "a red square"},
            {"ok": Noul("Is this a valid image?"),
             "route": Choice("Which queue?", {"left": None, "right": None})},
            image=IMAGE_URL,
        )
    assert result["ok"] == pytest.approx(0.948, abs=0.005)
    assert result["route"] == "left"
    assert result.answers["ok"].type == "noul"
    assert result.answers["route"].type == "choice"
    assert result.meta["input_type"] == "image"


def test_each_image_head_costs_one_request_and_is_charged():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response({"A": -0.1, "B": -3.0}, prompt_tokens=512))

    with make_image_model(handler) as model:
        result = model.classify({"note": "x"}, {"color": ["red", "blue"], "shape": ["square", "circle"]},
                                image=IMAGE_URL)
    assert len(calls) == 2
    assert result.usage["requests"] == 2
    assert result.usage["input_tokens"] == 1024
    assert result.usage["output_tokens"] == 2
    assert result["color"] == "red" and result["shape"] == "square"


def test_image_multi_label_asks_one_question_per_label():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response({"A": -0.1, "B": -3.0}))

    with make_image_model(handler) as model:
        result = model.classify(
            {"note": "x"},
            {"topics": {"labels": ["hvac", "billing"], "multi_label": True, "cls_threshold": 0.4}},
            image=IMAGE_URL,
        )
    assert len(calls) == 2
    assert result["topics"] == ["hvac", "billing"]
    assert result.usage["requests"] == 2
    assert set(result.raw) == {"topics:hvac", "topics:billing"}


def test_images_default_to_raw_probabilities_and_can_opt_into_calibration():
    # The record is passed explicitly: this test is about the image path opting
    # into the same scaling the text path applies. The fixture id is not the
    # checkpoint the bundled DE-1 record was fitted on, so the client would not
    # apply that record on its own.
    with make_image_model(dual_handler(), calibration=calibration_for("de1")) as model:
        image_default = model.classify({"note": "x"}, {"color": ["red", "blue"]},
                                       image=IMAGE_URL, include_probabilities=True)
        image_tempered = model.classify({"note": "x"}, {"color": ["red", "blue"]},
                                        image=IMAGE_URL, include_probabilities=True, calibrated=True)
        text_default = model.classify({"note": "x"}, {"color": ["red", "blue"]}, include_probabilities=True)
        text_raw = model.classify({"note": "x"}, {"color": ["red", "blue"]},
                                  include_probabilities=True, calibrated=False)

    assert image_default.answers["color"].calibrated is False
    assert image_default.answers["color"].temperature == 1.0
    assert image_default.meta["calibrated"] is False
    assert image_default.meta["input_type"] == "image"
    # The image default is the raw distribution, the same one the text path returns on request.
    assert image_default["color"]["probabilities"]["red"] == pytest.approx(
        text_raw["color"]["probabilities"]["red"])
    # An explicit calibrated=True applies the same scaling the text path applies by default.
    assert image_tempered.answers["color"].calibrated is True
    assert image_tempered["color"]["probabilities"]["red"] < image_default["color"]["probabilities"]["red"]
    assert text_default.meta["calibrated"] is True
    assert text_default.meta["input_type"] == "text"
    assert text_default["color"]["probabilities"]["red"] == pytest.approx(
        image_tempered["color"]["probabilities"]["red"])


def test_an_image_does_not_mutate_or_enter_the_evidence(tmp_path):
    path = tmp_path / "cat.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"pretend image bytes")
    state = {"note": "a red square"}
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response({"A": -0.1, "B": -3.0}))

    with make_image_model(handler) as model:
        result = model.classify(state, {"color": ["red", "blue"]}, image=path)

    assert result["color"] == "red"
    assert state == {"note": "a red square"}
    body = json.dumps(calls[0])
    assert "cat.png" not in body
    assert str(tmp_path) not in body
    image_part, text_part = calls[0]["messages"][1]["content"]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert json.loads(text_part["text"])["evidence"] == {"note": "a red square"}


def test_a_local_image_is_prepared_once_for_many_heads(tmp_path, monkeypatch):
    from shisa_de import client as client_module

    path = tmp_path / "cat.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"pretend image bytes")
    prepared: list = []
    original = client_module.prepare_image

    def counting(image):
        prepared.append(image)
        return original(image)

    monkeypatch.setattr(client_module, "prepare_image", counting)
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response({"A": -0.1, "B": -3.0}))

    with make_image_model(handler) as model:
        model.classify({"note": "x"}, {"color": ["red", "blue"], "shape": ["square", "circle"]}, image=path)
    assert len(prepared) == 1
    assert len(calls) == 2


def test_a_bad_image_is_rejected_before_the_client_sends_anything():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=chat_response({"A": -0.1, "B": -3.0}))

    with make_image_model(handler) as model:
        with pytest.raises(ImageError, match="extension"):
            model.classify({"note": "x"}, {"color": ["red", "blue"]}, image="cat.gif")
        with pytest.raises(ImageError, match="empty"):
            model.classify({"note": "x"}, {"color": ["red", "blue"]}, image="")
    assert calls == []


def test_image_top_logprobs_is_forwarded_to_the_readout():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response({"A": -0.1, "B": -3.0}))

    with make_image_model(handler, image_top_logprobs=64) as model:
        model.classify({"note": "x"}, {"color": ["red", "blue"]}, image=IMAGE_URL)
    assert calls[0]["top_logprobs"] == 64


def test_a_missing_image_letter_fails_the_client_call_rather_than_guessing():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response({"A": -0.1, "B": -3.0}))

    with make_image_model(handler) as model:
        with pytest.raises(ReadoutError, match="no letter fallback"):
            model.classify({"note": "x"}, {"color": ["red", "blue", "green"]}, image=IMAGE_URL)
    assert len(calls) == 1


def test_debug_keeps_the_image_messages_out_of_the_default_raw_record():
    with make_image_model(dual_handler()) as model:
        plain = model.classify({"note": "x"}, {"color": ["red", "blue"]}, image=IMAGE_URL)
        debug = model.classify({"note": "x"}, {"color": ["red", "blue"]}, image=IMAGE_URL, debug=True)
    assert "prompt" not in plain.raw["color"]
    assert "top_logprobs" not in plain.raw["color"]
    assert debug.raw["color"]["prompt"] is None  # the server rendered the prompt
    assert debug.raw["color"]["top_logprobs"]["A"] == -0.1


# -- text overflow ------------------------------------------------------------

@pytest.mark.parametrize("count", [27, 52, 53, 77, 129, 151, 676])
def test_overflow_balances_chunks_and_accounts_for_all_reads(count):
    calls = []
    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["prompt"].split("<user>")[1].split("<model>")[0])
        calls.append(payload)
        n = len(payload["options"])
        assert 2 <= n <= 26
        return httpx.Response(200, json=top_logprobs({chr(65+i): -float(i) for i in range(n)}))
    with DecisionModel(readout=make_readout(handler)) as model:
        result = model.decide("state", {"wide": Choice("Which?", {f"k{i}": None for i in range(count)})})
    answer = result.answers["wide"]
    blocks = (count + 25) // 26
    assert len(calls) == blocks + 1
    sizes = [len(c["options"]) for c in calls[:-1]]
    assert max(sizes) - min(sizes) <= 1 and sum(sizes) == count
    assert result.raw["wide"]["strategy"] == "finalist-top1"
    assert all("prompt" not in c for c in result.raw["wide"]["components"])
    assert calls[0]["options"][0]["description"] == "k0"
    assert result["wide"] == "k0"
    assert len(answer.probabilities) == count
    assert sum(answer.probabilities.values()) == pytest.approx(1)
    assert answer.confidence is None and not answer.calibrated
    assert answer.temperature == 1 and answer.stages == 2
    assert answer.logical_reads == answer.requests == blocks + 1
    assert result.usage["input_tokens"] == 12 * (blocks + 1)
    assert result.usage["output_tokens"] == blocks + 1
    assert result.meta["calibrated"] is False
    assert answer.to_dict()["strategy"] == "finalist-top1"
    assert "confidence" not in answer.to_wire()
    assert answer.to_wire()["score_semantics"] == "conditional-on-finalists"
    assert result.to_wire()["answers"]["wide"]["calibrated"] is False


@pytest.mark.parametrize("kind", ["strict", "too-wide", "image", "score", "calibration"])
def test_overflow_rejects_unsupported_shapes_before_any_requests(kind):
    calls = []
    def handler(request):
        calls.append(request)
        raise AssertionError("unexpected request")
    with DecisionModel(readout=make_readout(handler), overflow="error" if kind == "strict" else "finalist-top1") as model:
        q = Choice("Which?", {str(i): None for i in range(677 if kind == "too-wide" else 27)})
        if kind == "score":
            q = Score("Level?", list(range(27)))
        kwargs = {"image": IMAGE_URL} if kind == "image" else {"calibrated": True} if kind == "calibration" else {}
        with pytest.raises(QuestionError):
            model.decide("state", {"small": Noul("Yes?"), "wide": q}, **kwargs)
    assert not calls


def test_overflow_ties_keep_order_and_shorthand_marks_conditional_scores():
    def handler(request):
        payload = json.loads(json.loads(request.content)["prompt"].split("<user>")[1].split("<model>")[0])
        return httpx.Response(200, json=top_logprobs({chr(65+i): -1.0 for i in range(len(payload["options"]))}))
    with DecisionModel(readout=make_readout(handler)) as model:
        result = model.classify("state", {"wide": [f"k{i}" for i in range(27)], "small": ["yes", "no"]}, include_probabilities=True)
        raw = model.classify("state", {"wide": [f"k{i}" for i in range(27)], "small": ["yes", "no"]}, calibrated=False)
    assert raw.meta["calibrated"] is False
    answer = result.answers["wide"]
    assert answer.finalists == ["k0", "k14"]
    assert result["wide"]["label"] == "k0"
    assert result["wide"]["confidence"] is None
    assert result["wide"]["score_semantics"] == "conditional-on-finalists"
    assert result.meta["calibrated"] is None
    assert result.meta["calibration_by_head"] == {"wide": False, "small": True}


def test_overflow_fails_without_partial_answer():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="unavailable")
    with DecisionModel(readout=make_readout(handler)) as model:
        with pytest.raises(ReadoutError):
            model.classify("state", {"wide": list(map(str, range(27)))})
    assert len(calls) == 1


def test_overflow_fallback_accounting_and_incomplete_scores():
    from shisa_de.overflow import _check_read
    with pytest.raises(ReadoutError, match="incomplete"):
        _check_read(LetterRead(logprobs={"A": 0}, probabilities={"A": 1}), 2)
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if "prompt_logprobs" in body:
            letter = body["prompt"][-1]
            return httpx.Response(200, json={"choices": [{"prompt_logprobs": [None, {str(ord(letter)): {"logprob": -5.0, "rank": 40}}]}], "usage": {"prompt_tokens": 13}})
        payload = json.loads(body["prompt"].split("<user>")[1].split("<model>")[0])
        n = len(payload["options"])
        return httpx.Response(200, json=top_logprobs({chr(65+i): -float(i) for i in range(n-1)}))
    with DecisionModel(readout=make_readout(handler)) as model:
        result = model.classify("state", {"wide": list(map(str, range(27)))}, debug=True)
    assert result.usage["requests"] == 6
    assert result.usage["logical_reads"] == 3
    assert result.usage["input_tokens"] == 75
    assert len(result.answers["wide"].missing_from_top) == 3
    assert all(c["prompt"] for c in result.raw["wide"]["components"])


def test_max_logprobs_is_forwarded_without_changing_default():
    model = DecisionModel(max_logprobs=40)
    assert model.readout.max_logprobs == 40
    model.close()
    with pytest.raises(ValueError):
        DecisionModel(max_logprobs=True)
    with pytest.raises(ValueError):
        DecisionModel(overflow="truncate")


def test_overflow_replays_measured_component_distributions():
    from pathlib import Path
    from shisa_de.readout import LETTERS
    fixture = json.loads((Path(__file__).parent / "fixtures" / "overflow-replay.json").read_text())
    for case in fixture["cases"]:
        class Replay:
            def __init__(self):
                self.index = 0
            def evaluate(self, state, question, debug=False):
                assert state == case["state"]
                assert question.instructions == case["question"]["instructions"]
                component = case["components"][self.index]
                self.index += 1
                options = question.options()
                assert [key for key, _ in options] == component["keys"]
                for key, description in options:
                    assert description == render_option(key, case["question"]["criteria"][key])
                return LetterRead(
                    logprobs={LETTERS[i]: component["logprobs"][key] for i, key in enumerate(component["keys"])},
                    probabilities={LETTERS[i]: component["probabilities"][key] for i, key in enumerate(component["keys"])},
                    requests=component["requests"], prompt_tokens=component["prompt_tokens"]), options
            def close(self):
                pass
        replay = Replay()
        with DecisionModel(readout=replay) as model:
            result = model.decide(case["state"], {"wide": Choice(case["question"]["instructions"], case["question"]["criteria"])})
        assert result["wide"] == case["expected"]
        assert result.answers["wide"].probabilities == case["scores"]
        assert replay.index == len(case["components"])


# -- the DE-2 readout ---------------------------------------------------------

class CodeStubTokenizer(StubTokenizer):
    """A stub where the answer boundary and the two-letter codes are single tokens.

    DE-2 reads codes past `Z` and reads every answer after the terminator, so
    both have to be one token here the way they are in the checkpoint's tokenizer.
    Everything else stays one character per token.
    """

    def __init__(self) -> None:
        super().__init__()
        words = [ANSWER_PREFIX] + [code for code in codebook() if len(code) > 1]
        self.ids = {word: 0x110000 + index for index, word in enumerate(words)}
        self.ids[ANSWER_PREFIX] = int(ANSWER_TOKEN_ID)
        self.words = {value: word for word, value in self.ids.items()}
        self.pattern = re.compile("|".join(re.escape(word) for word in sorted(words, key=len, reverse=True)))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        self.templates.append(messages)
        opened = f"<sys>{messages[0]['content']}<user>{messages[1]['content']}<model>"
        return opened if enable_thinking else opened + ANSWER_PREFIX

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids, position = [], 0
        while position < len(text):
            match = self.pattern.match(text, position)
            if match:
                ids.append(self.ids[match.group()])
                position = match.end()
            else:
                ids.append(ord(text[position]))
                position += 1
        return ids

    def decode(self, ids) -> str:
        return "".join(self.words.get(value) or chr(value) for value in ids)


class StubDE2Server:
    """Answers the three DE-2 request shapes the way vLLM does.

    `logprob(code)` scores a code; `thought` is the token ids a thinking request
    generates. `candidate_ids` chooses how the server treats `logprob_token_ids`:
    honoured, ignored (an older server), or rejected with a 400.
    """

    def __init__(self, logprob=None, thought=None, candidate_ids="honoured", top_k=20, after_thought=None):
        self.tokenizer = CodeStubTokenizer()
        self.logprob = logprob or (lambda code: 0.0 if code == "A" else -6.0)
        self.after_thought = after_thought or self.logprob
        self.thought = thought
        self.candidate_ids = candidate_ids
        self.top_k = top_k
        self.calls: list[dict] = []
        self.code_of = {self.tokenizer.encode(code)[0]: code for code in codebook()}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append(body)
        prompt = body["prompt"]
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        usage = {"prompt_tokens": len(ids), "completion_tokens": 1}
        if "stop_token_ids" in body:
            assert body["return_token_ids"] is True and body["temperature"] == 0
            generated = list(self.thought)[:body["max_tokens"]]
            if int(ANSWER_TOKEN_ID) in generated:
                generated = generated[:generated.index(int(ANSWER_TOKEN_ID)) + 1]
            return httpx.Response(200, json={"choices": [{"text": "thinking", "token_ids": generated}], "usage": usage})
        score = self.after_thought if isinstance(prompt, list) else self.logprob
        if "prompt_logprobs" in body:
            token = ids[-1]
            entry = {str(token): {"logprob": score(self.code_of[token]), "rank": 30, "decoded_token": self.code_of[token]}}
            return httpx.Response(200, json={"choices": [{"prompt_logprobs": [None, entry]}], "usage": usage})
        assert body["max_tokens"] == 1 and body["temperature"] == 0
        if "logprob_token_ids" in body and self.candidate_ids == "rejected":
            return httpx.Response(400, json={"error": {"message": "extra_forbidden: logprob_token_ids"}})
        if "logprob_token_ids" in body and self.candidate_ids == "honoured":
            assert len(body["logprob_token_ids"]) <= 128
            top = {f"token_id:{token}": score(self.code_of[token]) for token in body["logprob_token_ids"]}
        else:
            ranked = sorted(codebook(), key=lambda code: -score(code))[:self.top_k]
            top = {code: score(code) for code in ranked}
        best = max(codebook(), key=score)
        return httpx.Response(200, json={"choices": [{"text": best, "logprobs": {"top_logprobs": [top]}}], "usage": usage})


def make_de2_model(server: StubDE2Server, **kwargs) -> DecisionModel:
    model = DecisionModel(base_url="http://test.local", model="de2-test", api_key="",
                          transport=httpx.MockTransport(server), **kwargs)
    model.readout._tokenizer = server.tokenizer
    return model


def wide_choice(count: int) -> Choice:
    return Choice("Which?", {f"k{index}": None for index in range(count)})


def test_the_codebook_is_the_letters_then_pinned_pairs():
    codes = codebook()
    assert len(codes) == MAX_CODES == 256 and len(set(codes)) == 256
    assert "".join(codes[:26]) == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert codes[26:29] == ("AA", "AB", "AC") and codes[-1] == "IW"
    assert "GZ" not in codes  # fails the boundary audit on the pinned tokenizer
    assert codes_for(3) == ["A", "B", "C"] and codes_for(27)[-1] == "AA"
    with pytest.raises(QuestionError):
        codes_for(257)


def test_a_small_question_renders_the_same_under_either_contract():
    """Up to 26 options the codes are the letters, so the scaffold is shared."""
    server = StubDE2Server()
    with make_de2_model(server) as model:
        question = wide_choice(26)
        assert model.readout.render("s", question, max_options=MAX_CODES) == model.readout.render("s", question)


def test_the_repeated_and_thinking_prompts():
    server = StubDE2Server()
    with make_de2_model(server) as model:
        readout = model.readout
        once = readout.render({"a": 1}, wide_choice(3))
        user = once.split("<user>")[1].split("<model>")[0]
        twice = readout.render({"a": 1}, wide_choice(3), repeat=2)
        assert twice == once.replace(user, user + INPUT_REPEAT + user)
        assert INPUT_REPEAT == "\n\nRead the same input again before answering:\n"
        thinking = readout.render({"a": 1}, wide_choice(3), thinking=True)
        # The thought is generated, so the prompt stops before the empty thought block.
        assert thinking == f"<sys>{THINK_SYSTEM}<user>{user}<model>"
        assert "no explanation or reasoning" not in THINK_SYSTEM
        wide = json.loads(readout.render("s", wide_choice(28), max_options=MAX_CODES)
                          .split("<user>")[1].split("<model>")[0])
        assert [option["letter"] for option in wide["options"][25:]] == ["Z", "AA", "AB"]
        with pytest.raises(QuestionError):
            readout.render("s", wide_choice(28))  # the DE-1 scaffold stops at the letters


@pytest.mark.parametrize("count,requests", [(2, 1), (26, 1), (27, 1), (128, 1), (129, 2), (256, 2)])
def test_every_code_is_read_by_token_id_without_a_fallback(count, requests):
    server = StubDE2Server(logprob=lambda code: 0.0 if code == "C" else -4.0)
    with make_de2_model(server, policy="direct") as model:
        result = model.decide("state", {"wide": wide_choice(max(count, 3))})
    answer = result.answers["wide"]
    count = max(count, 3)
    assert result["wide"] == "k2"
    assert len(answer.probabilities) == count and sum(answer.probabilities.values()) == pytest.approx(1)
    assert answer.requests == requests == len(server.calls) and answer.missing_from_top == []
    assert (answer.strategy, answer.score_semantics, answer.stages) == ("direct", "option-softmax", 1)
    assert answer.confidence is not None
    sent = [token for call in server.calls for token in call["logprob_token_ids"]]
    assert sent == [server.tokenizer.encode(code)[0] for code in codes_for(count)]
    assert all(call["return_tokens_as_token_ids"] is True and call["logprobs"] == 20 for call in server.calls)
    assert len({call["prompt"] for call in server.calls}) == 1
    # A wide DE-2 answer is an ordinary choice on the wire, not an overflow extension.
    assert set(answer.to_wire()) == {"type", "choice", "probabilities", "confidence"}
    assert result.meta["readout_version"] == DE2_READOUT_VERSION and result.meta["family"] == "de2"
    assert result.raw["wide"]["strategy"] == "direct"


@pytest.mark.parametrize("mode", ["ignored", "rejected"])
def test_a_server_without_candidate_ids_falls_back_to_the_top_k(mode):
    server = StubDE2Server(logprob=lambda code: -0.1 * codebook().index(code), candidate_ids=mode)
    with make_de2_model(server, policy="direct") as model:
        first = model.decide("state", {"wide": wide_choice(30)})
        second = model.decide("state", {"wide": wide_choice(30)})
    answer = first.answers["wide"]
    assert first["wide"] == "k0" and len(answer.probabilities) == 30
    # 20 codes arrive in the top-k; the other 10 cost one request each.
    assert len(answer.missing_from_top) == 10 and answer.requests == 11
    assert first.raw["wide"]["ranks"]["AD"] == 30
    assert sum(answer.probabilities.values()) == pytest.approx(1)
    if mode == "rejected":
        # The rejection is remembered: the next question does not ask again.
        assert sum("logprob_token_ids" in call for call in server.calls) == 1
        assert second.answers["wide"].requests == 11


def test_a_server_that_tokenizes_the_prompt_differently_is_an_error():
    server = StubDE2Server()
    def handler(request):
        response = server(request)
        body = response.json()
        body["usage"]["prompt_tokens"] += 1
        return httpx.Response(200, json=body)
    with make_de2_model(server) as model:
        model.readout._client = httpx.Client(base_url="http://test.local", transport=httpx.MockTransport(handler))
        with pytest.raises(ReadoutError, match="counted .* prompt tokens"):
            model.decide("state", {"q": Noul("Yes?")})


def test_a_prompt_that_does_not_end_at_the_answer_boundary_is_an_error():
    server = StubDE2Server()
    with make_de2_model(server) as model:
        slots = model.readout.code_slots(2)
        with pytest.raises(ReadoutError, match="does not end on"):
            model.readout.read_codes("<sys>x<user>y<model>", slots)
    assert not server.calls


def test_a_code_that_is_not_one_token_fails_loudly():
    server = StubDE2Server()
    with make_de2_model(server) as model:
        model.readout._tokenizer = ImageStubTokenizer()  # two-letter codes are two tokens here
        with pytest.raises(ReadoutError, match="not one token"):
            model.decide("state", {"wide": wide_choice(27)})
    assert not server.calls


def test_a_confident_repeated_read_is_the_answer():
    server = StubDE2Server(logprob=lambda code: 0.0 if code == "B" else -5.0)
    with make_de2_model(server) as model:
        result = model.decide("state", {"q": wide_choice(3)}, debug=True)
    answer = result.answers["q"]
    assert result["q"] == "k1" and len(server.calls) == 1
    assert (answer.strategy, answer.stages, answer.requests, answer.thought_tokens) == ("repeat2", 1, 1, 0)
    assert answer.thought_closed is None and "thought_tokens" not in answer.to_dict()
    assert server.calls[0]["prompt"].count(INPUT_REPEAT) == 1
    assert result.usage["output_tokens"] == 1 and result.usage["thought_tokens"] == 0
    assert result.meta["policy"] == "repeat-think" and result.meta["think_gate"] == 0.7
    assert result.meta["strategy_by_head"] == {"q": "repeat2"}
    assert result.raw["q"]["components"][0]["prompt"] == server.calls[0]["prompt"]


def test_an_unsure_read_thinks_and_answers_after_the_thought():
    close = int(ANSWER_TOKEN_ID)
    thought = [ord(character) for character in "<think>3+4=7"] + [close, ord("B")]
    server = StubDE2Server(logprob=lambda code: {"A": -0.7, "B": -0.8}.get(code, -3.0), thought=thought,
                           after_thought=lambda code: 0.0 if code == "B" else -7.0)
    with make_de2_model(server) as model:
        result = model.decide("state", {"q": wide_choice(3)}, debug=True)
    answer = result.answers["q"]
    first, think, last = server.calls
    assert result["q"] == "k1"  # the repeated read preferred k0; the thought changed it
    assert (answer.strategy, answer.stages, answer.logical_reads, answer.requests) == ("repeat2-think", 2, 2, 3)
    assert (answer.thought_tokens, answer.thought_closed) == (len(thought) - 2, True)
    assert answer.calibrated is False
    # The thought is generated from the thinking prompt, greedily, up to the budget.
    assert think["prompt"].startswith(f"<sys>{THINK_SYSTEM}<user>") and think["prompt"].endswith("<model>")
    assert think["max_tokens"] == 1024 and think["stop_token_ids"] == [close]
    # The codes are read after the thought and the close token, token for token.
    tokenizer = server.tokenizer
    assert last["prompt"] == tokenizer.encode(think["prompt"]) + thought[:-2] + [close]
    assert len(last["logprob_token_ids"]) == 3
    assert result.usage["requests"] == 3 and result.usage["thought_tokens"] == len(thought) - 2
    assert result.usage["output_tokens"] == 2 + len(thought) - 2
    raw = result.raw["q"]
    assert [stage["stage"] for stage in raw["components"]] == ["repeat2", "think"]
    assert raw["components"][0]["probabilities"]["A"] > raw["components"][0]["probabilities"]["B"]
    assert raw["thought"] == "thinking" and raw["thought_closed"] is True
    assert answer.to_dict()["thought_tokens"] == len(thought) - 2


def test_a_thought_cut_by_the_budget_is_closed_by_the_client():
    thought = [ord("x")] * 50
    server = StubDE2Server(logprob=lambda code: -1.0, thought=thought)
    with make_de2_model(server, think_budget=8) as model:
        result = model.decide("state", {"q": Noul("Yes?")})
    answer = result.answers["q"]
    assert (answer.thought_tokens, answer.thought_closed) == (8, False)
    assert server.calls[1]["max_tokens"] == 8
    assert server.calls[2]["prompt"][-9:] == [ord("x")] * 8 + [int(ANSWER_TOKEN_ID)]


@pytest.mark.parametrize("policy,gate,count,thinks", [
    ("repeat-think", 0.7, 3, True),
    ("repeat-think", 0.2, 3, False),   # the top probability clears a lower gate
    ("repeat-think", 0.7, 27, False),  # questions above 26 options never think
    ("repeat", 0.7, 3, False),
    ("direct", 0.7, 3, False),
])
def test_the_gate_the_option_cap_and_the_policy_decide_whether_to_think(policy, gate, count, thinks):
    server = StubDE2Server(logprob=lambda code: -1.0, thought=[ord("x"), int(ANSWER_TOKEN_ID)])
    with make_de2_model(server, policy=policy, think_gate=gate) as model:
        answer = model.decide("state", {"q": wide_choice(count)}).answers["q"]
    assert (answer.strategy == "repeat2-think") is thinks
    assert len(server.calls) == (3 if thinks else 1)
    assert (INPUT_REPEAT in server.calls[0]["prompt"]) is (policy != "direct")


def test_a_server_that_returns_no_thought_tokens_is_an_error():
    server = StubDE2Server(logprob=lambda code: -1.0)
    def handler(request):
        if "stop_token_ids" in json.loads(request.content):
            return httpx.Response(200, json={"choices": [{"text": "thinking"}], "usage": {}})
        return server(request)
    with make_de2_model(server) as model:
        model.readout._client = httpx.Client(base_url="http://test.local", transport=httpx.MockTransport(handler))
        with pytest.raises(ReadoutError, match="return_token_ids"):
            model.decide("state", {"q": Noul("Yes?")})


def test_de2_reads_typed_heads_and_keeps_their_limits():
    server = StubDE2Server(logprob=lambda code: 0.0 if code == "A" else -5.0)
    with make_de2_model(server) as model:
        result = model.decide("state", {"n": Noul("Yes?"), "s": Score("Level?", ["low", "mid", "high"])})
        assert result["n"] > 0.99 and result.answers["s"].level == "low"
        assert result.to_wire()["answers"]["n"] == {"type": "noul", "noul": result["n"]}
        assert result.meta["calibrated"] is False and result.answers["n"].temperature == 1.0
        tags = model.classify("state", {"tags": {"labels": ["x", "y"], "multi_label": True}})
        assert tags["tags"] == ["x", "y"]
        before = len(server.calls)
        # Only a text choice goes past the letters.
        for question, kwargs in ((wide_choice(257), {}), (Score("Level?", list(range(27))), {}),
                                 (wide_choice(27), {"image": IMAGE_URL})):
            with pytest.raises(QuestionError):
                model.decide("state", {"small": Noul("Yes?"), "wide": question}, **kwargs)
        # Nothing is fitted for DE-2, so asking for calibration returns raw answers.
        asked = model.decide("state", {"n": Noul("Yes?")}, calibrated=True)
        assert asked.answers["n"].calibrated is False and asked.meta["calibrated"] is False
    assert len(server.calls) == before + 1


def test_de1_still_overflows_where_de2_reads_natively():
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        n = len(json.loads(calls[-1]["prompt"].split("<user>")[1].split("<model>")[0])["options"])
        return httpx.Response(200, json=top_logprobs({chr(65 + i): -float(i) for i in range(n)}))
    with DecisionModel(readout=make_readout(handler)) as de1:
        assert de1.decide("state", {"wide": wide_choice(77)}).answers["wide"].strategy == "finalist-top1"
    assert len(calls) == 4 and all("logprob_token_ids" not in call for call in calls)
    server = StubDE2Server()
    with make_de2_model(server) as de2:
        assert de2.decide("state", {"wide": wide_choice(77)}).answers["wide"].strategy == "repeat2"
    assert len(server.calls) == 1


def test_probability_on_de2_stops_at_the_repeated_read_and_accepts_wide_choices():
    server = StubDE2Server(logprob=lambda code: -1.0, thought=[ord("x"), int(ANSWER_TOKEN_ID)])
    with make_de2_model(server) as model:
        unsure = model.decide("state", {"q": wide_choice(3)}, probability=True)
        wide = model.decide("state", {"q": wide_choice(77)}, probability=True)
        ordinary = model.decide("state", {"q": wide_choice(3)})
    # One logical read each: an unsure question would otherwise have thought.
    assert unsure.answers["q"].strategy == wide.answers["q"].strategy == "repeat2"
    assert unsure.usage["logical_reads"] == unsure.usage["requests"] == 1
    assert (unsure.meta["probability"], unsure.meta["policy"]) == (True, "repeat")
    assert len(wide.answers["q"].probabilities) == 77
    assert (ordinary.meta["policy"], ordinary.answers["q"].strategy) == ("repeat-think", "repeat2-think")


def test_read_policy_rejects_an_unknown_policy():
    server = StubDE2Server()
    with make_de2_model(server) as model:
        with pytest.raises(ValueError, match="unknown policy"):
            read_policy(model.readout, "state", Noul("Yes?"), policy="twice")


# -- the system turn on the image path ----------------------------------------

def image_prompt_logprobs(tokenizer, prompt: str, *, space: bool) -> list:
    """Server-side prompt tokens for a chat render, with or without the parts-format space."""
    if space:
        prompt = prompt.replace(DIRECT_SYSTEM, DIRECT_SYSTEM + " ")
    ids = tokenizer.encode(prompt)
    return [None] + [{str(token): {"logprob": -0.1, "decoded_token": tokenizer.decode([token])}} for token in ids[1:]]


@pytest.mark.parametrize("space,render", [(False, "string"), (True, "differs")])
def test_the_image_read_records_how_the_server_rendered_the_system_turn(space, render):
    tokenizer = CodeStubTokenizer()
    def handler(request):
        body = chat_response({"A": -0.1, "B": -3.0})
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": DIRECT_SYSTEM}, {"role": "user", "content": "<image>{}"}])
        body["prompt_logprobs"] = image_prompt_logprobs(tokenizer, prompt, space=space)
        return httpx.Response(200, json=body)

    # Both families record the render and answer either way: a stock vLLM writes
    # the space on every image request, and that is the image path as served.
    for model_id, readout_version in (("shisa-ai/shisa-de-1", READOUT_VERSION), ("de2-test", DE2_READOUT_VERSION)):
        with DecisionModel(base_url="http://test.local", model=model_id,
                           transport=httpx.MockTransport(handler)) as model:
            model.readout._tokenizer = tokenizer
            result = model.decide({}, {"q": image_question()}, image=IMAGE_URL)
            assert result.raw["q"]["system_render"] == render and result["q"] == "label0"
            # An image is one direct read under either contract: no repeat, no thought.
            assert (result.answers["q"].strategy, result.answers["q"].requests) == ("direct", 1)
            assert result.meta["input_type"] == "image" and result.meta["readout_version"] == readout_version
            # A caller with a fixed template can make the difference an error.
            payload = (model.readout, {}, image_question(), IMAGE_URL)
            if space:
                with pytest.raises(ReadoutError, match="system render: differs"):
                    payload[0].read_image(*payload[1:], require_string_system=True)
            else:
                assert payload[0].read_image(*payload[1:], require_string_system=True).system_render == "string"


def test_a_response_without_the_prompt_tokens_leaves_the_render_unverified():
    with make_image_model(dual_handler()) as model:
        assert model.decide({}, {"q": image_question()}, image=IMAGE_URL).raw["q"]["system_render"] == "unverified"
        with pytest.raises(ReadoutError, match="system render: unverified"):
            model.readout.read_image({}, image_question(), IMAGE_URL, require_string_system=True)


def test_doctor_probe_reports_the_read_and_the_chat_render(capsys):
    from shisa_de import cli

    server = StubDE2Server()
    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "de2-test"}]})
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            prompt = server.tokenizer.apply_chat_template(body["messages"])
            return httpx.Response(200, json={
                "choices": [], "prompt_logprobs": image_prompt_logprobs(server.tokenizer, prompt, space=True)})
        return server(request)

    def factory(**kwargs):
        built = DecisionModel(transport=httpx.MockTransport(handler), **kwargs)
        built.readout._tokenizer = server.tokenizer
        return built

    import shisa_de.cli as cli_module
    original = cli_module.DecisionModel
    cli_module.DecisionModel = factory
    try:
        assert cli.main(["doctor", "--probe", "--base-url", "http://test.local", "--model", "de2-test"]) == 0
    finally:
        cli_module.DecisionModel = original
    output = capsys.readouterr().out
    assert "read probe      passed (repeat2, 1 requests)" in output
    assert "chat system     differs" in output
    assert "family          de2 (model id)" in output and "de2-codebook-v1, policy repeat-think" in output


# -- read options: the model's defaults and one call's overrides -----------------


def test_each_family_has_its_default_read_options():
    de1, de2 = resolve_options("de1"), resolve_options("de2")
    assert (de1.policy, de1.reads, de1.reasoning, de1.compound) == ("direct", "single", False, True)
    assert (de2.policy, de2.reads, de2.reasoning, de2.compound) == ("repeat-think", "double", True, False)
    assert (de2.reasoning_prob, de2.reasoning_len) == (0.7, 1024)


@pytest.mark.parametrize("given,policy", [
    ({}, "repeat-think"),
    ({"reasoning": False}, "repeat"),
    ({"reads": "single"}, "direct"),           # a single read never reasons
    ({"reads": "double"}, "repeat-think"),
    ({"policy": "direct", "reasoning": False}, "direct"),
    ({"policy": "repeat", "reads": "double"}, "repeat"),
])
def test_reads_and_reasoning_select_one_of_the_three_policies(given, policy):
    assert resolve_options("de2", **given).policy == policy


def test_a_call_inherits_what_it_does_not_override():
    model = resolve_options("de2", policy="direct", reasoning_prob=0.9, reasoning_len=64)
    assert resolve_options("de2", model) == model
    eager = resolve_options("de2", model, reasoning=True)  # reasoning reads twice first
    assert (eager.policy, eager.think_gate, eager.think_budget) == ("repeat-think", 0.9, 64)
    assert resolve_options("de2", eager, reasoning_len=8) == ReadOptions("de2", "repeat-think", 0.9, 8)
    assert resolve_options("de1", compound=False).overflow == "error"
    assert resolve_options("de1", resolve_options("de1", overflow="error"), compound=True).compound


@pytest.mark.parametrize("family,given,match", [
    ("de2", {"reads": "single", "reasoning": True}, "single"),
    ("de2", {"policy": "direct", "reasoning": True}, "disagrees"),
    ("de2", {"policy": "repeat-think", "reads": "single"}, "disagrees"),
    ("de2", {"policy": "twice"}, "unknown policy"),
    ("de2", {"reads": "triple"}, "reads"),
    ("de2", {"reasoning": "yes"}, "boolean"),
    ("de2", {"compound": True}, "compound"),
    ("de2", {"reasoning_prob": 1.5}, "probability"),
    ("de2", {"reasoning_len": 0}, "positive integer"),
    ("de2", {"reasoning_prob": 0.5, "think_gate": 0.6}, "pass one"),
    ("de2", {"reasoning_len": 8, "think_budget": 16}, "pass one"),
    ("de1", {"compound": True, "overflow": "error"}, "pass one"),
    ("de1", {"overflow": "truncate"}, "overflow"),
    ("de1", {"reads": "double"}, "not defined for de1"),
    ("de1", {"reasoning": True}, "not defined for de1"),
])
def test_a_combination_no_contract_defines_is_an_error(family, given, match):
    with pytest.raises(ValueError, match=match):
        resolve_options(family, **given)


def test_the_constructor_takes_either_name_for_a_setting():
    new = DecisionModel(model="de2-x", reasoning_prob=0.5, reasoning_len=64, reads="double", reasoning=False)
    old = DecisionModel(model="de2-x", think_gate=0.5, think_budget=64, policy="repeat")
    assert new.options == old.options
    assert (new.policy, new.reads, new.reasoning) == ("repeat", "double", False)
    assert (new.think_gate, new.reasoning_prob, new.think_budget, new.reasoning_len) == (0.5, 0.5, 64, 64)
    strict = DecisionModel(model="shisa-ai/shisa-de-1", compound=False)
    assert (strict.overflow, strict.compound) == ("error", False)
    assert DecisionModel(model="shisa-ai/shisa-de-1").compound is True
    for kwargs in ({"reads": "double"}, {"reasoning": True}):
        with pytest.raises(ValueError, match="not defined for de1"):
            DecisionModel(model="shisa-ai/shisa-de-1", **kwargs)
    with pytest.raises(ValueError, match="compound"):
        DecisionModel(model="de2-x", compound=True)


@pytest.mark.parametrize("override,strategy,calls,repeated", [
    ({}, "repeat2-think", 3, True),
    ({"reasoning": False}, "repeat2", 1, True),
    ({"policy": "repeat"}, "repeat2", 1, True),
    ({"reads": "single"}, "direct", 1, False),
    ({"policy": "direct"}, "direct", 1, False),
    ({"reasoning_prob": 0.2}, "repeat2", 1, True),  # the read clears a lower gate
])
def test_one_call_overrides_the_models_read(override, strategy, calls, repeated):
    server = StubDE2Server(logprob=lambda code: -1.0, thought=[ord("x"), int(ANSWER_TOKEN_ID)])
    with make_de2_model(server) as model:
        result = model.decide("state", {"q": wide_choice(3)}, **override)
        assert result.answers["q"].strategy == strategy and len(server.calls) == calls
        assert (INPUT_REPEAT in server.calls[0]["prompt"]) is repeated
        assert result.meta["policy"] == resolve_options("de2", **override).policy
        assert result.meta["think_gate"] == override.get("reasoning_prob", 0.7)
        # The override was for that call; the model's default is unchanged.
        assert model.policy == "repeat-think" and model.think_gate == 0.7
        server.calls.clear()
        assert model.classify("state", {"q": ["a", "b", "c"]}).answers["q"].strategy == "repeat2-think"


def test_a_call_can_turn_reasoning_on_and_bound_the_thought():
    server = StubDE2Server(logprob=lambda code: -1.0, thought=[ord("x")] * 20)
    with make_de2_model(server, policy="direct") as model:
        assert model.decide("state", {"q": wide_choice(3)}).answers["q"].strategy == "direct"
        server.calls.clear()
        result = model.classify("state", {"q": ["a", "b", "c"]}, reasoning=True, reasoning_len=8)
        answer = result.answers["q"]
        assert (answer.strategy, answer.thought_tokens) == ("repeat2-think", 8)
        assert server.calls[1]["max_tokens"] == 8 and result.meta["think_budget"] == 8
        with pytest.raises(ValueError, match="one logical read"):
            model.decide("state", {"q": wide_choice(3)}, reasoning=True, probability=True)
        with pytest.raises(ValueError, match="compound"):
            model.decide("state", {"q": wide_choice(3)}, compound=True)


def test_compound_is_set_per_call_on_de1():
    calls = []
    def handler(request):
        calls.append(request)
        raise AssertionError("unexpected request")
    with DecisionModel(readout=make_readout(handler)) as model:
        with pytest.raises(QuestionError, match="rejects choices above 26"):
            model.decide("state", {"wide": wide_choice(27)}, compound=False)
        with pytest.raises(ValueError, match="not defined for de1"):
            model.decide("state", {"q": Noul("Yes?")}, reasoning=True)
    assert not calls


def test_the_cli_passes_the_read_options_through(capsys):
    from shisa_de import cli

    args = cli.build_parser().parse_args(
        ["doctor", "--model", "de2-x", "--no-reasoning", "--reasoning-prob", "0.5", "--reasoning-len", "64"])
    assert (args.reads, args.reasoning, args.reasoning_prob, args.reasoning_len, args.compound) == (
        None, False, 0.5, 64, None)
    with pytest.raises(SystemExit):
        cli.main(["doctor", "--model", "shisa-ai/shisa-de-1", "--reads", "double"])
    assert "not defined for de1" in capsys.readouterr().err
