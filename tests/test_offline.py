"""Offline tests: rendering, slots, request accounting, and answer shaping.

The readout is exercised against a stub tokenizer and a mock transport, so the
suite runs with no network, no GPU, and no tokenizer download. `test_live.py`
covers the real endpoint and is opt-in.
"""

from __future__ import annotations

import json
import math

import httpx
import pytest

from shisa_de import Choice, DecisionModel, Noul, Readout, ReadoutError, Score, softmax
from shisa_de.calibration import confidence, temper_binary, temper_distribution
from shisa_de.client import Decision, _MultiLabel, _question_from_head
from shisa_de.questions import MAX_OPTIONS, QuestionError, render_option
from shisa_de.readout import LetterRead


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
