# shisa-de

Talk to Shisa DE-1 decision models. Ask typed questions about a state, get typed
answers back, with the distribution and the request cost attached.

```python
from shisa_de import DecisionModel

de = DecisionModel()  # uses SHISA_API_KEY and the hosted DE-1 endpoint

de.classify(
    {"sms": "WINNER!! You have won a $1000 gift card. Claim it now: bit.ly/xyz",
     "sender": "+1-555-0199"},
    {"intent": {"spam": "Unsolicited bulk or scam message", "ham": "Ordinary message"}},
)
# {'intent': 'spam'}
```

DE-1 runs on a GPU server: this package holds an HTTP client and a tokenizer,
and downloads no weights. Nothing is fetched at import time. The tokenizer is
loaded lazily when first needed and may require a download from Hugging Face.

## Install

```bash
pip install -e .        # from a checkout; not published to PyPI yet
```

For the hosted service, set only your Shisa API key:

```bash
export SHISA_API_KEY=...
shisa-de doctor
shisa-de ask --state 'WINNER! Claim your free prize now!' --labels spam,ham
```

`DecisionModel()` and the CLI default to `shisa-ai/shisa-de-1` at
`https://api.shisa.ai/openai`. No endpoint or model argument is required.
`doctor` checks access to the model list, that DE-1 is listed, and the local
tokenizer's answer boundary; `ask` verifies an actual completion request.

Explicit `api_key=` takes precedence over `SHISA_DE_API_KEY`, then
`SHISA_API_KEY`. Explicit `base_url=` takes precedence over
`SHISA_DE_ENDPOINT`, then the hosted default. Leave the DE-specific environment
variables unset to use just `SHISA_API_KEY` with the hosted defaults.

## Classify

`classify` takes a state and a mapping of head name to label set. Each head is
one question, sent as its own request.

```python
ticket = {"subject": "Charged twice for order 4812",
          "body": "I was billed twice for the same order and I want my money back."}

de.classify(ticket, {"intent": ["refund_request", "cancel_order", "order_status", "speak_to_human"]})
# {'intent': 'refund_request'}

de.classify(ticket, {"intent": ["refund_request", "cancel_order"]}, include_confidence=True)
# {'intent': {'label': 'refund_request', 'confidence': 0.954}}

de.classify(ticket, {"intent": ["refund_request", "cancel_order"]}, include_probabilities=True)
# {'intent': {'label': 'refund_request', 'confidence': 0.954,
#             'probabilities': {'refund_request': 0.977, 'cancel_order': 0.023}}}
```

Several heads in one call run concurrently and come back together:

```python
comment = "The dashboard has been broken for three days and nobody has replied to my email."

de.classify(comment, {
    "sentiment": ["positive", "negative", "mixed"],
    "urgency": ["low", "normal", "high"],
    "escalate": {"labels": ["yes", "no"], "prompt": "Should a human read this today?"},
})
# {'sentiment': 'negative', 'urgency': 'high', 'escalate': 'yes'}
```

### Label set forms

| Form | Meaning |
| --- | --- |
| `["a", "b"]` | A choice over the labels. The question is derived from the head name. |
| `{"a": "description", "b": "description"}` | A choice with descriptions shown to the model. |
| `{"labels": ["a", "b"], "prompt": "..."}` | The same, with your own question. |
| `{"labels": ["a", "b"], "multi_label": True, "cls_threshold": 0.5}` | One yes/no question per label; returns the labels above the threshold. |
| `{"levels": ["low", "medium", "high"]}` | An ordered scale; returns the level, with the weighted position on the answer object. |
| `Noul(...)`, `Choice(...)`, `Score(...)` | The question object itself, for full control. |

A head name becomes the question: `"urgency"` asks "What is the urgency?".
Pass `prompt` when the default reads badly. One question can offer at most 26
labels, because the model reads one answer letter per question.

## Decide

`decide` is the System One call: typed questions in, typed answers out. The dict
view returns each answer's primary value; `result.answers` carries the
distribution, confidence, and cost behind it.

```python
from shisa_de import Choice, Noul, Score

msg = {"sms": "WINNER!! You have won a $1000 gift card. Claim it now: bit.ly/xyz", "sender": "+1-555-0199"}

result = de.decide(msg, {
    "is_spam": Noul("Is this message spam?"),
    "ask": Choice("What does the sender want?", {
        "card_details": "Payment card details",
        "callback": "A support callback",
        "nothing": "Nothing; it is routine",
    }),
    "risk": Score("How risky is acting on this message?", ["Safe", "Suspicious", "Dangerous"]),
})

dict(result)
# {'is_spam': 0.99, 'ask': 'card_details', 'risk': 1.75}
```

```python
result.answers["ask"].probabilities
# {'card_details': 0.94, 'callback': 0.03, 'nothing': 0.03}
result.answers["ask"].confidence
# 0.9
result.answers["risk"].level, result.answers["risk"].score
# ('Dangerous', 1.75)
result.usage
# {'input_tokens': 427, 'output_tokens': 3, 'requests': 3, 'wall_ms': 131.96}
result.to_wire()          # the System One response shape: model, answers, usage
```

Answer values: a noul returns `P(yes)`, a choice returns the option key, and a
score returns the probability-weighted position over the levels. `classify`
returns the level description for an ordered scale instead, so a scale reads as
one of its levels.

## Calibration and confidence

Probabilities are tempered by default with the temperatures fitted for DE-1
(noul 1.69, choice and score 1.90), which sharpens the distribution without
changing the answer. Every answer records `calibrated` and `temperature`, so
raw and tempered scores are never mixed by accident. Pass `calibrated=False` for
raw logprob-derived probabilities.

`confidence` is `(K * p_max - 1) / (K - 1)`, the statistic the hosted System One
API returns. It summarizes the distribution; a decision that depends on the
shape of the distribution should read `probabilities` instead.

Repeated identical requests on the hosted endpoint returned bit-identical
logprobs, with one exception in fifteen that moved the probability by 0.001.
Treat a threshold inside that band as undecided; [docs/READOUT.md](docs/READOUT.md)
records the measurement.

## Endpoints

```python
DecisionModel.from_pretrained("shisa-ai/shisa-de-1")                  # hosted, from the environment
DecisionModel.from_endpoint("http://127.0.0.1:8021", model="shisa-ai/shisa-de-1")   # local vLLM
DecisionModel(base_url=..., model=..., api_key=..., timeout=..., max_workers=8)
```

`from_pretrained` resolves the endpoint from `SHISA_DE_ENDPOINT`, then the
hosted default, and downloads no weights. `max_workers` bounds how many
questions run at once; each question is one request.

## Command line

```bash
shisa-de doctor                          # endpoint, served model, boundary check
shisa-de ask --labels spam,ham           # one classification, JSON out
shisa-de explain --labels spam,ham       # every step of the readout, printed
```

`shisa-de explain` prints the rendered prompt, the answer slot token ids, the
top logprobs with the answer slots marked, the normalized distribution, and the
equivalent `curl`.

## The readout

DE-1 answers by reading one letter per question: the prompt carries lettered
options, the request asks for one token with `logprobs`, and the answer is the
highest-probability option letter. [docs/READOUT.md](docs/READOUT.md) is the
full specification — request, rendered prompt, token ids, fallback for letters
outside the top-k, the arithmetic from logprobs to answers, calibration
provenance, measured costs, and a dependency-light reference implementation.

## Tests

```bash
python -m pytest tests/                              # offline, no network
SHISA_DE_LIVE=1 python -m pytest tests/test_live.py  # against a live endpoint
```

## Links

- [docs/READOUT.md](docs/READOUT.md) — the readout contract, with samples
- [shisa-ai/shisa-de-1](https://huggingface.co/shisa-ai/shisa-de-1) — model card, serving scripts, readout contract in prose
- [shisa-ai/jevbench-results](https://github.com/shisa-ai/jevbench-results) — the harness and published JevBench numbers behind the model card

Apache-2.0; see [LICENSE](LICENSE).
