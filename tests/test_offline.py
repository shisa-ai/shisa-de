"""Offline tests: rendering, slots, request accounting, and answer shaping.

The readout is exercised against a stub tokenizer and a mock transport, so the
suite runs with no network, no GPU, and no tokenizer download. `test_live.py`
covers the real endpoint and is opt-in.
"""

from __future__ import annotations

import base64
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Event, Lock
from types import SimpleNamespace

import httpx
import pytest

from shisa_de import Choice, DecisionModel, Noul, Readout, ReadoutError, Score, softmax
from shisa_de.calibration import confidence, temper_binary, temper_distribution
from shisa_de.client import Decision, _MultiLabel, _question_from_head
from shisa_de.images import ImageError, prepare_image, validate_image_url
from shisa_de.questions import MAX_OPTIONS, QuestionError, render_option
from shisa_de.readout import ANSWER_PREFIX, DIRECT_SYSTEM, LetterRead


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

    with DecisionModel(model="custom-model", transport=httpx.MockTransport(handler), **kwargs) as model:
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

    with DecisionModel(base_url="http://test.local", model="test-model",
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


# -- the client on an image --------------------------------------------------

def make_image_model(handler, **kwargs) -> DecisionModel:
    model = DecisionModel(base_url="http://test.local", model="test-model",
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
    with make_image_model(dual_handler()) as model:
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
