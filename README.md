# shisa-de

A Python client for [Shisa DE-1](https://huggingface.co/shisa-ai/shisa-de-1) and
DE-2, models for classification and typed decisions.

Give it text, images, or structured data and a set of labels or questions. It
returns answers, probabilities, and request usage, not generated prose. The
client loads only a tokenizer; model weights stay on the server.

- **DE-1** is available on the hosted [Shisa Platform](https://platform.shisa.ai/)
  or from your own OpenAI-compatible server.
- **DE-2** is not hosted or released yet. The client can read a DE-2 checkpoint
  you serve yourself; see [Use DE-2](#use-de-2).

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

Release notes are in [CHANGELOG.md](CHANGELOG.md).

## Quick start

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

The defaults are model `shisa-ai/shisa-de-1` at `https://api.shisa.ai/openai`.
The tokenizer is downloaded from Hugging Face on the first call, not at import.
The client needs neither PyTorch nor a GPU.

Check your setup from the command line:

```bash
shisa-de doctor
shisa-de ask --state 'WINNER! Claim your free prize now!' --labels spam,ham
```

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

By default each name maps straight to its label. `include_confidence=True`
returns the label with a confidence, and `include_probabilities=True` adds the
full probability map.

Other label-set forms:

| Form | Behavior |
| --- | --- |
| `["a", "b"]` | Choose one label; derive the question from the name |
| `{"a": "description", "b": "description"}` | Choose one label using its description |
| `{"labels": ["a", "b"], "prompt": "Which applies?"}` | Supply your own question |
| `{"labels": ["a", "b"], "multi_label": True, "cls_threshold": 0.5}` | Ask yes/no for each label; return labels above the threshold |
| `{"levels": ["low", "medium", "high"]}` | Choose a level on an ordered scale |

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

Questions run concurrently; `max_workers=` controls how many at once.
Multi-label classification asks one question per label. `result.usage` records
the request and token counts, and `result.answers[name]` holds each answer's
probabilities, confidence, and cost.

On DE-1 a question costs one request, plus one for each option that falls
outside the server's top 20 logprobs.

## Classify images

Pass a local image path, an HTTP(S) image URL, or a base64 image data URL as
`image=`:

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

```bash
shisa-de ask --image photo.jpg --labels cat,dog,bird --prompt 'What animal is shown?'
```

`decide(..., image="photo.jpg")` asks typed questions about an image too. Local
files are uploaded; URLs are fetched by the server. Supported formats are PNG,
JPEG, and WebP.

Image questions take at most 26 options and return uncalibrated probabilities.
If the server does not return a logprob for every option, the client raises an
error rather than guessing. On your own server, raise `--max-logprobs` and set
`DecisionModel(image_top_logprobs=...)` to match for label sets above 20.

## Large label sets

A single DE-1 question holds at most 26 options. For a text choice with 27 to
676 options, the client splits the options into chunks, picks a winner from
each, and asks a final question among the winners:

```python
from shisa_de import Choice, DecisionModel

with DecisionModel() as de:
    result = de.decide("Route this support request", {
        "route": Choice("Which queue applies?", {
            f"queue_{i}": f"Support queue {i}" for i in range(77)
        }),
    })
    answer = result.answers["route"]
    print(answer.choice, answer.finalists, answer.requests)
    print(answer.strategy)  # finalist-top1
```

What to know about these answers:

- The scores cover only the finalists; eliminated options score zero. They are
  **not** probabilities over all labels, so `confidence` is `None` and the
  answer is never calibrated.
- Option order can change the answer.
- Quality was tested up to 151 options. 676 is a structural limit, not a
  quality guarantee.
- Images and ordered scores cannot go above 26.

Use `compound=False` (or the older `overflow="error"`), on the model or on one
call, to reject wide choices instead. The
[overflow section](docs/READOUT.md#13-text-choice-overflow) of the DE-1 readout
has the algorithm, its cost, and the evidence behind it.

DE-2 does not need any of this: it reads up to 256 options in one question and
returns an ordinary probability distribution. See [Use DE-2](#use-de-2).

## Probabilities and calibration

Every answer carries its probabilities, and records whether they were
calibrated in `answer.calibrated` and `answer.temperature`.

| Call | Probabilities returned |
| --- | --- |
| DE-1 text | Calibrated by default; pass `calibrated=False` for raw |
| Images | Raw |
| DE-2 text | Calibrated by default, with a separate fit for each read; pass `calibrated=False` for raw |
| DE-1 large label sets | Finalist scores, never calibrated |

Each bundled calibration is applied only to the model it was fitted on:
`shisa-ai/shisa-de-1` and `shisa-ai/shisa-de-2`. Any other checkpoint gets raw
probabilities and a warning. The DE-2 fit is a first, rough one; probabilities
read after a thinking step are the least reliable
([details](docs/READOUT-DE2.md#7-calibration)). To use a calibration you fitted yourself, or to
vouch for the bundled one on another checkpoint, pass it in:

```python
from shisa_de import DecisionModel, calibration_for, load_calibration_file

DecisionModel(model="my-de1-finetune", family="de1", calibration=calibration_for("de1"))
DecisionModel(model="my-checkpoint", family="de2", calibration=load_calibration_file("my.json"))
```

Calibration depends on the model, the way it is read, and how it is served.
Refit before using these probabilities to drive a threshold on your own
deployment. The [DE-1 readout](docs/READOUT.md#8-calibration) has the details.

### When the probability is the result

Pass `probability=True` when you need the distribution itself, such as a
forecast, rather than the winning label:

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

The flag guarantees that every answer is one distribution over all of its
options, taken from a single pass over the question:

- On DE-1 it rejects choices above 26 options before sending anything, because
  finalist scores are not a distribution over the original options.
- On DE-2 it skips the thinking step. Choices up to 256 options are accepted.

`decide`, `system_one`, and `classify` all accept it. It does not change
prompts, answer shapes, or calibration: the example asks for raw probabilities
with `calibrated=False`, and leaving that out keeps the default. It is separate
from `include_probabilities=True`, which only changes what `classify` returns.

## Use a local server

Start an OpenAI-compatible server with the model loaded, then point the client
at it. The [DE-1 model card](https://huggingface.co/shisa-ai/shisa-de-1) has
serving guidance for DE-1.

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

```bash
export SHISA_DE_ENDPOINT=http://127.0.0.1:8021/v1
shisa-de doctor
shisa-de ask --state 'Please refund my duplicate payment.' --labels refund_request,order_status
```

- A server root URL and a URL ending in `/v1` are both accepted.
- For an authenticated server, pass `api_key="your-local-key"`.
- If the server uses an alias for the model, pass the alias as `model=` and
  `tokenizer="shisa-ai/shisa-de-1"`. The tokenizer name also tells the client
  that the alias is DE-1, so it keeps the DE-1 calibration.
- For DE-1 text the server needs `/v1/completions` with `logprobs` and
  `prompt_logprobs`, and `doctor` needs `/v1/models`. vLLM supports these.
  Images also need `/v1/chat/completions` with vision enabled.

### Configuration

| Setting | Resolution order |
| --- | --- |
| Endpoint | `base_url=` / `--base-url`, then `SHISA_DE_ENDPOINT`, then the hosted default |
| API key | `api_key=`, then `SHISA_DE_API_KEY`, then `SHISA_API_KEY` |
| Model | `model=` / `--model`, otherwise `shisa-ai/shisa-de-1` |
| Tokenizer | `tokenizer=` / `--tokenizer`, otherwise the model ID |
| Model family | `family=` / `--family`, then `de-1` or `de-2` in the model ID, then in the tokenizer name, otherwise DE-2 with a warning |

An explicit `api_key=""` disables authentication. Otherwise keys from the
environment are sent to local endpoints too.

## Use DE-2

The same `classify` and `decide` calls work against a DE-2 checkpoint you serve.
The client asks DE-2 differently from DE-1, the way DE-2 was built to be asked:

- **Each question is shown to the model twice** in one prompt.
- **Unsure questions get a thinking step.** If the top probability is below 0.7
  and the question has at most 26 options, the model reasons for up to 1,024
  tokens and then answers again.
- **Text choices go up to 256 options** in one question, with a full probability
  distribution and a confidence.
- **Probabilities are calibrated per read** on `shisa-ai/shisa-de-2`: answers
  read once, twice, and after thinking each get their own fit. Other DE-2
  checkpoints get raw probabilities.

```python
from shisa_de import Choice, DecisionModel

with DecisionModel.from_endpoint(
    "http://127.0.0.1:8021/v1",
    model="my-checkpoint",                    # the id your server uses
    family="de2",
    tokenizer="google/gemma-4-26B-A4B-it",
    api_key="",
) as de:
    result = de.decide("Route this support request", {
        "route": Choice("Which queue applies?", {
            f"queue_{i}": f"Support queue {i}" for i in range(200)
        }),
    })
    answer = result.answers["route"]
    print(answer.choice, answer.confidence)
    print(answer.strategy)          # repeat2, or repeat2-think if it used a thinking step
    print(answer.thought_tokens)    # 0 unless it did
```

```bash
shisa-de doctor --model my-checkpoint --family de2 --tokenizer google/gemma-4-26B-A4B-it --probe
```

**Say which family the model is.** DE-1 and DE-2 are asked differently, so a
model read as the wrong family returns answers nobody has measured. The client
recognizes `de-1` or `de-2` in the model ID or tokenizer name. Anything else is
treated as DE-2 with a warning, so pass `family="de1"` or `family="de2"` when
the ID does not say.

**Trade accuracy for speed** with the [read settings](#read-settings): a
thinking step takes seconds where an ordinary answer takes tens of milliseconds.
Choices above 128 options cost one extra request.

**Serving DE-2 with vLLM:**

- The default `--max-logprobs` is enough at every label-set size.
- Use `--attention-backend TRITON_ATTN` on vLLM 0.30.0. The DE-2 research notes
  report that the default attention backend lowers Gemma 4's accuracy.
- Do not serve a chat template that repeats the input itself; this client
  already does.
- Leave `--chat-template-content-format` at its default if you use images.
  Setting it to `string` makes vLLM 0.30.0 reject image requests.

The [DE-2 readout](docs/READOUT-DE2.md) has the exact prompts and requests, the
measured costs, and the server command they were measured on.

## Read settings

Each family has a default way of reading a question. You can change it for a
model, or for one call.

| Setting | What it does | DE-2 default | DE-1 default |
| --- | --- | --- | --- |
| `reads` | `"single"` or `"double"`: how often the question is shown in the prompt | `"double"` | `"single"` (fixed) |
| `reasoning` | Think when the answer is unsure, then answer again | `True` | `False` (fixed) |
| `reasoning_prob` | Think when the top probability is below this | `0.7` | not used |
| `reasoning_len` | The most tokens a thinking step may run to | `1024` | not used |
| `compound` | Read a text choice above 26 options in two rounds ([large label sets](#large-label-sets)) | not used | `True` |

Set them on the model to change its default, or on a call to change that call:

```python
# Model default: never think.
de = DecisionModel.from_endpoint(url, model="my-checkpoint", family="de2", reasoning=False)

de.classify(state, labels)                           # shown twice, no thinking
de.classify(state, labels, reads="single")           # this call only: fastest
de.decide(state, questions, reasoning=True,          # this call only: think sooner,
          reasoning_prob=0.9, reasoning_len=256)     # for fewer tokens
```

A call inherits whatever it does not set. `result.meta` records the policy,
gate, and budget that call ran under, and each answer's `strategy` says whether
it actually thought.

**Fastest or best.** On DE-2 the three combinations are also named, as `policy=`:

| Goal | Settings | `policy` | Requests per question |
| --- | --- | --- | --- |
| Best quality (default) | `reads="double", reasoning=True` | `"repeat-think"` | 1, or 3 when it thinks |
| Balanced | `reasoning=False` | `"repeat"` | 1 |
| Lowest latency | `reads="single"` | `"direct"` | 1, with about half the prompt tokens |

DE-1 has one read, so there is nothing to trade: `compound=False` only makes
choices above 26 options an error instead of a two-round read.

**The adaptive policy on a DE-1 checkpoint.** Declaring a DE-1 model as DE-2
reads it the DE-2 way: shown twice, with a thinking step when unsure.

```python
DecisionModel(model="shisa-ai/shisa-de-1", family="de2")
```

DE-1 was not trained to be read this way, and the result has not been measured
or calibrated: probabilities are raw, and answers are recorded as `family: de2`.
The server must also support the [DE-2 requests](docs/READOUT-DE2.md). A DE-2
model is the recommended way to get this policy.

Things to know:

- `reasoning=True` always shows the question twice first, and `reads="single"`
  never thinks. Asking for both at once is an error.
- Questions above 26 options never think, and neither does a call made with
  `probability=True`.
- Settings a family does not have raise `ValueError`: a double read or reasoning
  on DE-1, and `compound=True` on DE-2 (it reads up to 256 options in one
  question).
- The older names still work and mean the same thing: `think_gate` is
  `reasoning_prob`, `think_budget` is `reasoning_len`, and
  `overflow="finalist-top1"` / `"error"` is `compound=True` / `False`.

## Command line

| Command | What it does |
| --- | --- |
| `shisa-de doctor` | Checks the endpoint, the model ID, the tokenizer, and whether the calibration belongs to the served model. Exits non-zero on a problem |
| `shisa-de doctor --probe` | Also sends one question, and reports how the server handles the image path |
| `shisa-de ask` | Classifies `--state` (text or JSON) against `--labels`, optionally with `--image` |
| `shisa-de explain` | Prints the prompt, the answer tokens, the logprobs, and the final distribution for one question |

All of them accept `--base-url`, `--model`, `--tokenizer`, `--family`,
`--policy`, and `--calibration` (a family name or a path to a calibration file).
The [read settings](#read-settings) are `--reads`, `--reasoning` /
`--no-reasoning`, `--reasoning-prob`, `--reasoning-len`, and `--compound` /
`--no-compound`.

## How it works

The client renders each question into a prompt, asks the server for one token,
and reads the probabilities of the option letters at that position. Nothing is
generated except DE-2's optional thinking step. The exact prompts, requests, and
arithmetic are specified so that another client can be written from them:

- [The DE-1 readout](docs/READOUT.md)
- [The DE-2 readout](docs/READOUT-DE2.md)

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
client. Set `SHISA_DE_MODEL` and `SHISA_DE_TOKENIZER` to run them against
another served model.

## License

This client library is licensed under [Apache 2.0](LICENSE).
See the [model card](https://huggingface.co/shisa-ai/shisa-de-1) for the model's
license and usage requirements.
