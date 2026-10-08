"""Optional integration checks: put the pinned Decision Index kit on PYTHONPATH."""
import json
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("decision_index")
from scripts.decision_index_engine import ShisaDE2Engine, Unsupported


class FakeReadout:
    def ensure_tokenizer(self):
        return SimpleNamespace(encode=lambda prompt, **kw: list(range(int(prompt))))

    def render(self, state, question, **kwargs):
        assert kwargs == {"repeat": 2, "max_options": 256}
        return str(state[question.instructions])


def engine():
    result = ShisaDE2Engine.__new__(ShisaDE2Engine)
    result.max_tokens = 10
    result.repeat = 2
    calls = []

    def decide(state, questions, **kwargs):
        calls.append((state, questions, kwargs))
        return SimpleNamespace(to_wire=lambda: {"answers": {}}, meta={}, usage={})

    result.model = SimpleNamespace(readout=FakeReadout(), decide=decide)
    return result, calls


def choice(instructions="first", count=2):
    return {"type": "choice", "instructions": instructions,
            "criteria": {str(i): str(i) for i in range(count)}}


def test_preflights_all_questions_before_sending():
    e, calls = engine()
    with pytest.raises(Unsupported, match="Context window"):
        e({"first": 2, "second": 10}, {"a": choice(), "b": choice("second")})
    assert calls == []


def test_accepts_exact_context_limit_and_preserves_option_order():
    e, calls = engine()
    e({"first": 9}, {"a": choice()})
    assert list(calls[0][1]["a"].criteria) == ["0", "1"]
    assert calls[0][2] == {"calibrated": False}


@pytest.mark.parametrize("count", [1, 257])
def test_refuses_option_capacity_without_filtering(count):
    e, calls = engine()
    with pytest.raises(Unsupported, match="capacity"):
        e({"first": 2}, {"a": choice(count=count)})
    assert calls == []


def test_noul_uses_existing_yes_no_contract():
    e, calls = engine()
    e({"first": 2}, {"a": {"type": "noul", "instructions": "first",
                           "criteria": {"true": "provided true", "false": "provided false"}}})
    assert calls[0][1]["a"].options() == [("yes", "Yes"), ("no", "No")]


@pytest.mark.parametrize("status,text,capacity", [
    (400, "maximum context length exceeded", True),
    (422, "max_model_len exceeded", True),
    (400, "invalid field", False),
    (500, "maximum context length", False),
])
def test_only_declared_context_errors_become_unsupported(status, text, capacity):
    e, _ = engine()
    response = httpx.Response(status, text=text, request=httpx.Request("POST", "http://localhost:8026"))

    def fail(*args, **kwargs):
        response.raise_for_status()

    e.model.decide = fail
    with pytest.raises(Unsupported if capacity else httpx.HTTPStatusError):
        e({"first": 2}, {"a": choice()})
