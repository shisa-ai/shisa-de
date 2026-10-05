# shisa-de

A Python client for [Shisa DE-1](https://huggingface.co/shisa-ai/shisa-de-1) and
DE-2, models for classification and typed decisions. Use DE-1 through the hosted
[Shisa Platform](https://platform.shisa.ai/) or a local OpenAI-compatible
server. DE-2 is not hosted or released yet; the client reads a DE-2 checkpoint
you serve yourself ([Use DE-2](#use-de-2)).

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

`doctor` checks model-list access, the model ID, the tokenizer's answer
boundary, and whether the calibration record belongs to the served model.
`doctor --probe` also sends one question. `ask` makes an actual decision request.

## Probability-scored questions

Pass `probability=True` when the distribution, rather than just the winning
label, is the result you need:

```python
from shisa_de import DecisionModel, Noul

with DecisionModel() as de:
    result = de.decide(
        {"forecast": "Rain is expected tomorrow afternoon."},
        {"rain": Noul("Will it rain tomorrow?")},
        probability=True,
        calibrated=False,
    )
    p_yes = result["rain"]
    distribution = result.answers["rain"].probabilities
```

The flag applies to every question in the call, including each label in a
multi-label head. It requires one logical read per question. Missing-letter
recovery can still require extra HTTP requests at the same answer position.

- **DE-1** rejects choices above 26 options before any requests are sent:
  overflow's finalist scores are not a distribution over all original options.
  Direct questions already use a single read with thinking disabled.
- **DE-2** skips the thinking read, so every answer is the repeated read's own
  distribution (`strategy` is `repeat2`, and `result.meta["policy"]` reads
  `repeat`). Choices up to 256 options are accepted, because one read returns a
  distribution over all of them.

`decide`, its `system_one` alias, and `classify` accept this flag, defaulting to
`False`. The flag records caller intent in `result.meta["probability"]` and
enforces the single-read restriction. It is not sent as a server parameter and
does not change prompts or answer shapes.
`include_probabilities=True` only changes the `classify` dict view; it does not
set probability intent.

Calibration remains independent. The example requests raw logprob-derived
probabilities with `calibrated=False`; omitting it retains the default text
calibration. The flag neither applies a forecasting-specific scale nor selects
a label-smoothed checkpoint. The bundled temperatures are not a validated fit
for a different adapter, and the client applies them only to the checkpoint they
were fitted on; validate calibration for your serving model and task.
Image calls also accept the flag and retain their uncalibrated default.

## Use a local server

First start an OpenAI-compatible server with DE-1 or DE-2 loaded. See the
[DE-1 model card](https://huggingface.co/shisa-ai/shisa-de-1) for DE-1 serving
guidance and [Use DE-2](#use-de-2) for DE-2. Then point this client at it:

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
`tokenizer="shisa-ai/shisa-de-1"` to keep using the checkpoint's tokenizer. The
tokenizer source also tells the client the alias is DE-1 and that the bundled
calibration belongs to it.

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
| Family | `family=` / CLI `--family`, then a `de-1`/`de-2` slug in the model ID, then one in the tokenizer source, otherwise `de2` with a warning |

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

## Wide text label sets

This section describes DE-1. DE-2 reads up to 256 options in one prompt and
returns an ordinary distribution; see [Use DE-2](#use-de-2).

Requires `shisa-de>=0.2.0`: on DE-1, `DecisionModel` handles
27–676 text choice options with balanced chunks and a final choice among each
chunk's winner. Ordinary choices with up to 26 options keep their direct path.

```python
from shisa_de import Choice, DecisionModel

with DecisionModel(max_logprobs=40) as de:  # requires server support for top-k 40
    result = de.decide("Route this support request", {
        "route": Choice("Which queue applies?", {
            f"queue_{i}": f"Support queue {i}" for i in range(77)
        }),
    })
    answer = result.answers["route"]
    print(answer.choice, answer.finalists, answer.requests)
    print(answer.strategy)  # finalist-top1
```

The returned full-key map contains final-round scores on finalists and zero on
eliminated options. These scores are **conditional on finalist selection**, not
calibrated probabilities over all labels. Overflow answers have
`calibrated=False` and `confidence=None`. Explicit `calibrated=True` is rejected
for overflow; omit it to keep ordinary text heads calibrated in a mixed call.
`include_probabilities=True` also exposes the strategy and score semantics for
wide `classify` heads.

Use `DecisionModel(overflow="error")` for strict rejection above 26. Image and
ordered-score overflow are unsupported. Live quality testing covered up to
151 options; 676 is a tested structural limit, not a quality guarantee. Option
order can change the answer. See the [overflow contract](docs/READOUT.md#13-text-choice-overflow)
for the algorithm, cost, provenance and held-out evidence.

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
An image question is one direct read on both DE-1 and DE-2, with at most 26
options.

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

On DE-1 each question has at most 26 options and costs one request, plus a
fallback request for each option missing from the top logprobs. DE-2 differs:
see [Use DE-2](#use-de-2). Multi-label classification asks one question per
label. Questions run concurrently; `max_workers=` controls concurrency.
`result.usage` records the request and token counts.

## Use DE-2

The same calls work against a served DE-2. The client reads it through a
different contract, [`de2-codebook-v1`](docs/READOUT-DE2.md):

- **Every question is read with the user turn written twice.** If that read's
  top probability is below 0.7 and the question has at most 26 options, the
  model thinks for up to 1,024 tokens and is read again after the thought.
- **Text choices go up to 256 options in one prompt**, with a full distribution
  and a confidence. There is no chunking and no finalist round.
- **Every option's logprob is requested by token id**, so a question costs one
  request up to 128 options and two up to 256, with no fallback requests.
- **Nothing is calibrated.** No temperature has been fitted for this readout, so
  DE-2 probabilities are raw and `calibrated` is `False`.

```python
from shisa_de import Choice, DecisionModel

with DecisionModel.from_endpoint(
    "http://127.0.0.1:8021/v1",
    model="de2-v4-lr5e5-e3-s7",                 # the served id; a `de2` slug selects the contract
    tokenizer="google/gemma-4-26B-A4B-it",
    api_key="",
) as de:
    result = de.decide("Route this support request", {
        "route": Choice("Which queue applies?", {
            f"queue_{i}": f"Support queue {i}" for i in range(200)
        }),
    })
    answer = result.answers["route"]
    print(answer.choice, answer.confidence, answer.requests)   # one of 200, 2 requests
    print(answer.strategy)          # repeat2, or repeat2-think when the question thought
    print(answer.thought_tokens)    # 0 unless it thought
    print(result.meta["readout_version"])   # de2-codebook-v1
```

Choose how much of the policy runs with `policy=`:

| `policy` | Reads | Typical cost |
| --- | --- | --- |
| `"repeat-think"` (default) | Twice, then a thought when unsure | 1 request; 3 when it thinks |
| `"repeat"` | Twice | 1 request |
| `"direct"` | Once, as DE-1 does | 1 request, about half the prompt tokens |

A thought is slow next to a read: about 2 seconds for a 260-token thought
against about 0.04 seconds for a question that does not think, on one local GPU
(2026-10-06, `vllm-0.30.0-709530de`). Use `policy="repeat"` when latency
matters more than the unsure questions, and `think_gate=` / `think_budget=` to
move the gate or shorten the thought.

If the served id does not say which family it is, declare it:

```python
DecisionModel(model="my-arm", family="de2")   # or family="de1"
```

```bash
shisa-de doctor --model my-arm --family de2 --tokenizer google/gemma-4-26B-A4B-it --probe
```

An undeclared id with no `de-1` or `de-2` slug is read as DE-2 and the client
warns. That matters: DE-1 and DE-2 questions are rendered and read differently,
so a checkpoint read through the wrong contract returns answers nobody measured.

**Serving.** DE-2 needs vLLM's `/v1/completions` with `logprob_token_ids`,
`return_token_ids` and token-id prompts; the default `--max-logprobs` is enough.
The research notes report that vLLM 0.30.0's default attention kernel understates
Gemma 4 and recommend `--attention-backend TRITON_ATTN`. Do not combine this
client with a chat template that repeats the input itself. Images work on the
default chat settings; `--chat-template-content-format string` makes vLLM 0.30.0
reject image requests. The [DE-2 readout](docs/READOUT-DE2.md#8-serving) has the
launch command used for the measurements and the details behind each of these.

## Probabilities and calibration

On DE-1, text calls apply the bundled temperature scaling by default; image
calls and DE-2 calls return raw probabilities. Pass `calibrated=False` for raw
DE-1 text probabilities too. Each answer records `calibrated` and `temperature`;
`result.meta` records the family, the readout version and the calibration
identity. See the [DE-1 readout](docs/READOUT.md#8-calibration) for the scoring
and calibration details.

A temperature belongs to the checkpoint, the readout and the serving shape it
was fitted on, so the client applies the bundled record only where it was
fitted:

| Served model | What happens |
| --- | --- |
| `shisa-ai/shisa-de-1`, or an alias with `tokenizer="shisa-ai/shisa-de-1"` | The DE-1 record is applied |
| Another DE-1-family checkpoint, such as `de1-cont-...` | Not applied; the client warns and returns raw probabilities |
| Any DE-2 checkpoint | Nothing to apply: the DE-2 record is unfitted |

To use temperatures you fitted yourself, or to vouch for the bundled record on
another checkpoint, pass a record. It is applied as given:

```python
from shisa_de import DecisionModel, calibration_for, load_calibration_file

DecisionModel(model="de1-cont-v1", calibration=calibration_for("de1"))
DecisionModel(model="my-arm", family="de2", calibration=load_calibration_file("my.json"))
```

```bash
shisa-de doctor --model my-arm --family de2 --calibration ./my.json
```

`doctor` prints the model, readout and serving shape the record was fitted on,
how far it matches the served model, and whether it is applied. It fails when
the record belongs to another family or to a readout whose answers this client
does not reproduce.

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
