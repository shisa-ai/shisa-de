"""Live tests against a served DE-1 or DE-2. Opt in with SHISA_DE_LIVE=1.

    SHISA_DE_LIVE=1 python -m pytest tests/test_live.py -v
    SHISA_DE_LIVE=1 SHISA_DE_ENDPOINT=http://127.0.0.1:8021 SHISA_DE_MODEL=de2-v4-lr5e5-e3-s7 \
        SHISA_DE_TOKENIZER=google/gemma-4-26B-A4B-it python -m pytest tests/test_live.py -v

The default is hosted DE-1, which needs SHISA_API_KEY. `SHISA_DE_MODEL` and
`SHISA_DE_TOKENIZER` point the suite at another served model; tests that assert
one family's contract skip on the other. These tests spend real requests, so
they stay out of the default run.
"""

from __future__ import annotations

import json
import os
import struct
import zlib

import pytest

from shisa_de import DEFAULT_MODEL, DE2_READOUT_VERSION, Choice, DecisionModel, Noul, Score

pytestmark = pytest.mark.skipif(
    os.environ.get("SHISA_DE_LIVE") != "1",
    reason="set SHISA_DE_LIVE=1 to run against a live endpoint",
)

SPAM = {
    "sms": "WINNER!! You have won a $1000 gift card. Claim it now: bit.ly/xyz",
    "sender": "+1-555-0199",
}


@pytest.fixture(scope="module")
def model():
    with DecisionModel.from_pretrained(
        os.environ.get("SHISA_DE_MODEL") or DEFAULT_MODEL,
        tokenizer=os.environ.get("SHISA_DE_TOKENIZER") or None,
    ) as served:
        yield served


def only(model, family):
    if model.family != family:
        pytest.skip(f"asserts the {family} contract; the served model is read as {model.family}")


def test_text_choice_overflow_returns_conditional_scores(model):
    only(model, "de1")
    criteria = {f"label_{i}": f"Category number {i}" for i in range(27)}
    result = model.decide({"category": 7}, {"wide": Choice("Which category number is stated?", criteria)})
    answer = result.answers["wide"]
    assert answer.choice in criteria
    assert set(answer.probabilities) == set(criteria)
    assert sum(answer.probabilities.values()) == pytest.approx(1)
    assert answer.strategy == "finalist-top1"
    assert answer.confidence is None and not answer.calibrated
    assert answer.logical_reads == 3 and answer.requests >= 3
    assert result.to_wire()["answers"]["wide"]["score_semantics"] == "conditional-on-finalists"


def test_health_reports_the_endpoint_and_passes_the_boundary_check(model):
    report = model.health()
    assert report["models_status"] == 200, report
    assert report["model_listed"] is True, report
    assert report["boundary_check"] == "passed", report
    assert report["ok"] is True


def test_classify_answers_an_obvious_case(model):
    result = model.classify(SPAM, {"intent": {"spam": "Unsolicited bulk or scam message", "ham": "Ordinary message"}})
    assert result["intent"] == "spam"
    assert 0.0 <= result.answers["intent"].confidence <= 1.0


def test_one_call_can_ask_several_heads(model):
    result = model.decide(
        SPAM,
        {
            "is_spam": Noul("Is this message spam?"),
            "ask": Choice("What does the sender want?", {"card_details": "Payment card details",
                                                          "callback": "A support callback",
                                                          "nothing": "Nothing; it is routine"}),
            "risk": Score("How risky is acting on this message?", ["Safe", "Suspicious", "Dangerous"]),
        },
    )
    assert result["is_spam"] > 0.5
    assert result["ask"] == "card_details"
    # `decide` returns the probability-weighted position, like the System One API;
    # the level the answer sits on is on the answer object.
    assert result["risk"] > 0.5
    assert result.answers["risk"].level in {"Suspicious", "Dangerous"}
    # One request per head, plus two for any DE-2 head unsure enough to think.
    thinking = sum(answer.thought_closed is not None for answer in result.answers.values())
    assert result.usage["requests"] == 3 + 2 * thinking
    assert result.usage["input_tokens"] > 0


def test_the_model_card_example_still_renders_the_same_prompt(model):
    """The card's worked example is a 161-token prompt that answers A at 0.9973."""
    only(model, "de1")
    payload = {
        "evidence": "From: billing@acme-support.example\nSubject: Your invoice is overdue\n\n"
                    "We could not charge your card. Reply with your full card number, expiry, and CVV "
                    "so we can release the pending refund.",
        "criterion": "What is this message trying to obtain?",
    }
    question = Choice(
        instructions=payload["criterion"],
        criteria={
            "payment card details": None,
            "a support callback": None,
            "nothing; it is a routine notice": None,
        },
    )
    prompt = model.readout.render(payload["evidence"], question)
    tokenizer = model.readout.ensure_tokenizer()
    assert len(tokenizer.encode(prompt, add_special_tokens=False)) == 161
    read = model.readout.read(prompt, 3)
    assert read.answer() == "A"
    assert read.probabilities["A"] > 0.99
    assert read.requests == 1


@pytest.mark.parametrize("color,rgb", [("red", b"\xff\x00\x00"), ("blue", b"\x00\x00\xff")])
def test_image_classification_reads_pixels_not_filenames(model, tmp_path, color, rgb):
    def chunk(kind, data):
        return (struct.pack("!I", len(data)) + kind + data
                + struct.pack("!I", zlib.crc32(kind + data) & 0xffffffff))

    width = height = 32
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress((b"\x00" + rgb * width) * height))
           + chunk(b"IEND", b""))
    path = tmp_path / "image.png"
    path.write_bytes(png)
    result = model.classify(
        {}, {"color": {"labels": ["red", "blue"], "prompt": "What is the dominant color in this image?"}},
        image=path,
    )
    assert result["color"] == color
    assert result.answers["color"].calibrated is False
    assert result.meta["input_type"] == "image"
    assert result.usage["requests"] == 1
    assert result.usage["input_tokens"] > 0
    assert sum(result.answers["color"].probabilities.values()) == pytest.approx(1)


def test_answers_are_json_serializable(model):
    result = model.classify(SPAM, {"intent": ["spam", "ham"]}, include_probabilities=True)
    assert json.loads(json.dumps(dict(result)))["intent"]["label"] in {"spam", "ham"}


def test_health_probe_serves_a_read(model):
    report = model.health(probe=True)
    assert report["read_probe"].startswith("passed"), report
    assert report["chat_system_render"] in {"string", "differs"}, report
    assert report["ok"] is True


@pytest.mark.parametrize("count,requests", [(27, 1), (128, 1), (256, 2)])
def test_de2_reads_a_wide_choice_in_one_prompt(model, count, requests):
    only(model, "de2")
    criteria = {f"label_{i}": f"Category number {i}" for i in range(count)}
    result = model.decide({"category": 7}, {"wide": Choice("Which category number is stated?", criteria)})
    answer = result.answers["wide"]
    assert answer.choice == "label_7"
    assert set(answer.probabilities) == set(criteria)
    assert sum(answer.probabilities.values()) == pytest.approx(1)
    assert answer.strategy == "repeat2" and answer.missing_from_top == []
    assert answer.requests == requests and answer.confidence is not None
    assert result.meta["readout_version"] == DE2_READOUT_VERSION
    assert set(result.to_wire()["answers"]["wide"]) == {"type", "choice", "probabilities", "confidence"}


def test_de2_thinks_when_the_repeated_read_is_unsure(model):
    only(model, "de2")
    question = Choice("Which option is correct?", {"a": "270", "b": "280", "c": "290", "d": "391"})
    # A gate of 1.0 sends every question that is not certain through a thought.
    with DecisionModel(base_url=model.base_url, model=model.model, think_gate=1.0,
                       tokenizer=os.environ.get("SHISA_DE_TOKENIZER") or None) as eager:
        result = eager.decide("What is 17 * 23 - 111?", {"sum": question})
    answer = result.answers["sum"]
    assert answer.strategy == "repeat2-think" and answer.requests == 3
    assert 0 < answer.thought_tokens <= eager.think_budget and answer.thought_closed is not None
    assert result["sum"] == "b"
    assert result.usage["thought_tokens"] == answer.thought_tokens
    assert answer.calibrated is False
