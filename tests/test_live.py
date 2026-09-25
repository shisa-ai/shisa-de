"""Live tests against a served DE-1. Opt in with SHISA_DE_LIVE=1.

    SHISA_DE_LIVE=1 SHISA_DE_ENDPOINT=http://127.0.0.1:8021 python -m pytest tests/test_live.py -v

The default endpoint is the hosted one, which needs SHISA_API_KEY. These tests
spend real requests, so they stay out of the default run.
"""

from __future__ import annotations

import json
import os

import pytest

from shisa_de import Choice, DecisionModel, Noul, Score

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
    with DecisionModel.from_pretrained() as de1:
        yield de1


def test_health_reports_the_endpoint_and_passes_the_boundary_check(model):
    report = model.health()
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
    assert result.usage["requests"] == 3
    assert result.usage["input_tokens"] > 0


def test_the_model_card_example_still_renders_the_same_prompt(model):
    """The card's worked example is a 161-token prompt that answers A at 0.9973."""
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


def test_answers_are_json_serializable(model):
    result = model.classify(SPAM, {"intent": ["spam", "ham"]}, include_probabilities=True)
    assert json.loads(json.dumps(dict(result)))["intent"]["label"] in {"spam", "ham"}
