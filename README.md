# shisa-de

A Python client for [Shisa DE-1](https://huggingface.co/shisa-ai/shisa-de-1), a
model for classification and typed decisions. Use it through the hosted
[Shisa Platform](https://platform.shisa.ai/) or a local OpenAI-compatible server
serving DE-1.

Give it text, images, or structured data and a set of labels or questions. It returns
answers, probabilities, and request usage—not generated prose. The client loads
only the tokenizer; model weights stay on the server.

## Benchmark snapshot

| Benchmark | Shisa DE-1 | Jev |
| --- | ---: | ---: |
| [JevBench standard accuracy](https://github.com/fstandhartinger/jevbench) | 98.6% | 98.6% |
| [JevBench hard accuracy](https://github.com/fstandhartinger/jevbench) | 64.9% | 73.0% |
| [AG News accuracy](https://huggingface.co/datasets/fancyzhx/ag_news) | 89.5% | 90.0% |
| [Median latency](https://huggingface.co/shisa-ai/shisa-de-1#evaluation) | 20 ms (local) | 223 ms (hosted) |

## Install

Requires Python 3.10 or newer.

```bash
pip install shisa-de
```

For development, see [Development](#development). Release notes are in
[CHANGELOG.md](CHANGELOG.md).

## Use the Shisa Platform

Get an API key from [platform.shisa.ai](https://platform.shisa.ai/) and set it in
your environment:

```bash
export SHISA_API_KEY="your-api-key"
```

No endpoint or model argument is needed:

```python
from shisa_de import DecisionModel

with DecisionModel() as de:
    result = de.classify(
        "WINNER! Claim your free prize now!",
        {"intent": ["spam", "ham"]},
    )
    print(result["intent"])  # spam
```

The defaults are model `shisa-ai/shisa-de-1` at
`https://api.shisa.ai/openai`. The tokenizer is loaded lazily from Hugging Face
when first needed, so the first call may require a download. Nothing is fetched
at import time. PyTorch and a local GPU are not required for the client.

Check your setup from the command line:

```bash
shisa-de doctor
shisa-de ask --state 'WINNER! Claim your free prize now!' --labels spam,ham
```

`doctor` checks model-list access, the model ID, and the tokenizer's answer
boundary. `ask` makes an actual decision request.

## Use a local server

First start an OpenAI-compatible server with DE-1 loaded. See the
[DE-1 model card](https://huggingface.co/shisa-ai/shisa-de-1) for serving guidance.
Then point this client at it:

```python
from shisa_de import DecisionModel

with DecisionModel.from_endpoint(
    "http://127.0.0.1:8021/v1",
    model="shisa-ai/shisa-de-1",
    api_key="",  # no authentication; do not use a key from the environment
) as de:
    result = de.classify(
        "I was charged twice. Please refund the duplicate payment.",
        {"intent": ["refund_request", "order_status", "cancel_order"]},
    )
    print(result["intent"])
```

Both a server root URL and a URL ending in `/v1` are accepted. For an
authenticated local server, pass its key as `api_key="your-local-key"`.
If the server uses an alias for the model, pass that alias as `model=` and
`tokenizer="shisa-ai/shisa-de-1"` to keep using the checkpoint's tokenizer.

For text, the server needs `/v1/completions` with `logprobs` and
`prompt_logprobs`; `doctor` also needs `/v1/models`. vLLM supports these.
See the [readout contract](docs/READOUT.md) for the exact requests.

For the CLI, use `--base-url` or set `SHISA_DE_ENDPOINT`. In a shell without
hosted credentials:

```bash
export SHISA_DE_ENDPOINT=http://127.0.0.1:8021/v1
shisa-de doctor
shisa-de ask --state 'Please refund my duplicate payment.' --labels refund_request,order_status
```

### Configuration precedence

| Setting | Resolution order |
| --- | --- |
| Endpoint | `base_url=` / CLI `--base-url`, then `SHISA_DE_ENDPOINT`, then the hosted default |
| API key | `api_key=`, then `SHISA_DE_API_KEY`, then `SHISA_API_KEY` |
| Model | `model=` / CLI `--model`, otherwise `shisa-ai/shisa-de-1` |
| Tokenizer | `tokenizer=` / CLI `--tokenizer`, otherwise the model ID |

An explicit `api_key=""` disables authentication. Otherwise, environment keys
are used for local endpoints too. Leave `SHISA_DE_ENDPOINT` and
`SHISA_DE_API_KEY` unset to use the hosted defaults with only `SHISA_API_KEY`.

## Classify with labels

Each named label set is a separate question. Labels can be plain strings or a
mapping from labels to descriptions:

```python
from shisa_de import DecisionModel

with DecisionModel() as de:
    result = de.classify(
        {"subject": "Charged twice", "body": "Please refund the duplicate charge."},
        {"intent": {
            "refund_request": "The customer wants money returned",
            "order_status": "The customer wants a delivery update",
        }},
        include_probabilities=True,
    )
    print(result["intent"]["label"])
    print(result["intent"]["probabilities"])
    print(result.usage)
```

Use `include_confidence=True` for a label and confidence without the full
probability map. The default result maps each head name directly to its label.

Other label-set forms:

| Form | Behavior |
| --- | --- |
| `["a", "b"]` | Choose one label; derive the question from the head name |
| `{"a": "description", "b": "description"}` | Choose one label using its description |
| `{"labels": ["a", "b"], "prompt": "Which applies?"}` | Supply your own question |
| `{"labels": ["a", "b"], "multi_label": True, "cls_threshold": 0.5}` | Ask yes/no for each label; return labels above the threshold |
| `{"levels": ["low", "medium", "high"]}` | Choose a level on an ordered scale |

## Classify images

Pass a local image path, an HTTP(S) image URL, or a base64 image data URL as
`image=`. The same API works with the Shisa Platform and a local DE-1 server
with vision enabled:

```python
from shisa_de import DecisionModel

with DecisionModel() as de:
    result = de.classify(
        {},
        {"animal": ["cat", "dog", "bird"]},
        image="photo.jpg",
        include_probabilities=True,
    )
    print(result["animal"]["label"])
    print(result["animal"]["probabilities"])
```

`decide(..., image="photo.jpg")` supports typed questions about an image too.

Local files are uploaded; URLs are fetched by the server. Supported formats:
PNG, JPEG, and WebP. Image probabilities are uncalibrated by default.

The server must support images and token logprobs on `/v1/chat/completions`.
If an option is missing from the returned logprobs, the client raises an error.
For larger label sets on a local server, raise its `--max-logprobs` and set
`DecisionModel(image_top_logprobs=128)` to match.

```bash
shisa-de ask --image photo.jpg --labels cat,dog,bird --prompt 'What animal is shown?'
```

## Ask typed questions

`decide` supports yes/no probabilities (`Noul`), named choices (`Choice`), and
ordered scores (`Score`):

```python
from shisa_de import Choice, DecisionModel, Noul, Score

with DecisionModel() as de:
    result = de.decide(
        "Reply with your full card number and CVV to claim your prize.",
        {
            "is_spam": Noul("Is this message spam?"),
            "asks_for": Choice("What does the sender want?", {
                "card_details": "Payment card details",
                "callback": "A support callback",
                "nothing": "Nothing; it is routine",
            }),
            "risk": Score("How risky is acting on this message?", [
                "Safe", "Suspicious", "Dangerous",
            ]),
        },
    )
    print(dict(result))
    print(result.answers["asks_for"].probabilities)
    print(result.answers["risk"].level)
    print(result.usage)
```

- `Noul` returns the probability of yes.
- `Choice` returns the selected option key.
- `Score` returns the probability-weighted, zero-based position in the scale.
  The most likely level is available as `result.answers[name].level`.

Each question has at most 26 options. Each question costs one request; text
questions add a fallback request for each option missing from the top logprobs.
Multi-label classification asks one question per label. Questions run
concurrently; `max_workers=` controls concurrency. `result.usage` records the
request and token counts.

## Probabilities and calibration

Text calls apply the bundled temperature scaling by default; image calls return
raw probabilities. Pass `calibrated=False` for raw text probabilities too.
Each answer records `calibrated` and `temperature`; `result.meta` records the
readout version and calibration identity. See the [readout documentation](docs/READOUT.md)
for the scoring and calibration details.

## Development

```bash
git clone https://github.com/shisa-ai/shisa-de.git
cd shisa-de
python -m pip install -e '.[dev]'
python -m pytest tests/
```

Offline tests use a stub tokenizer and mock HTTP transport. Live tests spend
real API requests and are opt-in:

```bash
SHISA_DE_LIVE=1 python -m pytest tests/test_live.py -v
```

Live tests use the same endpoint and API-key environment variables as the
client. `shisa-de explain --labels spam,ham` prints the prompt, token slots,
logprobs, and final distribution for a live request.

## License

This client library is licensed under [Apache 2.0](LICENSE).
See the [model card](https://huggingface.co/shisa-ai/shisa-de-1) for the model's
license and usage requirements.
