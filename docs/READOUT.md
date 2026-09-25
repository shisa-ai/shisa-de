# The DE-1 readout

DE-1 answers one typed question about one state and returns a distribution over
the answer options. It does not write prose and it does not answer two questions
in one request. This page is the complete contract: the request, the rendered
prompt, the answer slots, the fallback, and the arithmetic that turns logprobs
into answers. Everything in the `shisa_de` package implements this page, and a
client in another language can be written from it.

Measured numbers on this page come from `shisa-ai/shisa-de-1` served by
`vllm-0.26.0-tp2-c1aff9a1` on 2026-09-25. The readout version is
`de1-letter-slots-v1`. Re-measure if the serving fingerprint changes; logprobs
move slightly between serving shapes.

- Readout version: `de1-letter-slots-v1`
- Maximum options per question: 26
- Requests per question: 1, plus 1 per option letter outside the returned top-k
- Answer position: the first generated token

## The contract in one page

| # | Step | Detail |
| --- | --- | --- |
| 1 | Render the prompt | The checkpoint's chat template, one question, options lettered from `A` |
| 2 | Send one request | `POST /v1/completions`, `max_tokens: 1`, `temperature: 0`, `logprobs: 20` |
| 3 | Read the answer position | `choices[0].logprobs.top_logprobs[0]` holds the token distribution |
| 4 | Keep the option letters | `A`, `B`, ... in order; discard every other token |
| 5 | Fill in missing letters | One extra request per letter outside the top-k, using `prompt_logprobs` |
| 6 | Normalize | Softmax over the option letters only |
| 7 | Answer | The highest-probability letter, mapped back to its option |

## 1. One question per request

A request carries exactly one question. The model was trained with a single
criterion and a single option list per prompt, and the answer is read at one
fixed position: the first generated token. Two questions in one prompt would
need a second answer position, and DE-1 has none.

The hosted System One API accepts a list of questions and returns a list of
answers. A DE-1 client implements that by sending one request per question and
merging the results. `DecisionModel.classify` and `DecisionModel.decide` do this
for you and run the questions over a thread pool.

## 2. The request

```http
POST {base_url}/v1/completions
Content-Type: application/json
Authorization: Bearer $SHISA_API_KEY

{
  "model": "shisa-ai/shisa-de-1",
  "prompt": "<the rendered prompt>",
  "max_tokens": 1,
  "temperature": 0,
  "logprobs": 20
}
```

- `max_tokens: 1` asks for one token. Only the first position is read, so a
  longer generation is wasted work.
- `temperature: 0` removes sampling randomness. The answer is taken from the
  returned distribution, not from the sampled token.
- `logprobs: 20` returns the top 20 tokens at that position with their
  logprobs. 20 covers the option letters for questions with up to about 20
  options in practice; see [The fallback](#6-the-fallback) for the rest.

The same request without Python:

```bash
curl -s https://api.shisa.ai/openai/v1/completions \
  -H "Authorization: Bearer $SHISA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model": "shisa-ai/shisa-de-1", "prompt": "<rendered prompt>", "max_tokens": 1, "temperature": 0, "logprobs": 20}'
```

Response, with the top-logprob map trimmed to the answer letters:

```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "model": "shisa-ai/shisa-de-1",
  "choices": [
    {
      "index": 0,
      "text": "A",
      "logprobs": {
        "text_offset": [0],
        "token_logprobs": [-0.0007664603181183338],
        "tokens": ["A"],
        "top_logprobs": [
          {
            "A": -0.0007664603181183338,
            "B": -8.25076675415039,
            "1": -8.62576675415039,
            "S": -10.56326675415039
          }
        ]
      },
      "finish_reason": "length"
    }
  ],
  "usage": {"prompt_tokens": 137, "completion_tokens": 1, "total_tokens": 138},
  "system_fingerprint": "vllm-0.26.0-tp2-c1aff9a1"
}
```

The map in `top_logprobs[0]` is keyed by the decoded token text, and it is a
top-k slice, not a distribution over all 262,144 tokens. Tokens other than the
option letters (`1`, `S`, and so on) are discarded; a high-probability
non-letter does not become the answer.

## 3. The prompt

The prompt has three parts: a fixed system line, a user message holding one JSON
object, and the template's generation scaffold.

**System line**, byte for byte:

```text
Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.
```

**User message**, one JSON object with three keys:

| Key | Value |
| --- | --- |
| `evidence` | The state, verbatim. A string, an object, or a list. |
| `criterion` | The question, as one sentence. |
| `options` | A list of `{"letter": "A", "description": ...}`, in answer order |

Rendering rules:

- Letters are assigned in list order, starting at `A`. Option `A` is the first
  entry of `criteria` for a choice, `Yes` for a noul, and the first level for a
  score.
- A description is never empty. When a label carries no description, the label
  itself is shown. A prompt with blank descriptions gives the model
  indistinguishable options; in the research harness, dropping the label keys
  from the rendered prompt cost 0.129 to 0.948 accuracy on `dbpedia14`.
- The evidence is serialized as given. Wrapping it in prose changes the input.

**The generation scaffold.** The checkpoint's template appends the assistant
turn and opens an empty thinking channel, because thinking is disabled at
render time. The rendered prompt therefore ends at:

```text
<|turn>model
<|channel>thought
<channel|>
```

That scaffold is why the answer sits at the first generated position: the model
has already emitted the empty thought block before generation starts.

**A complete rendered prompt**, for the 2-option spam question used throughout
this page. `<bos>`, `<|turn>`, `<turn|>`, `<|channel>` and `<channel|>` are the
checkpoint's control tokens; `\n` marks a literal newline:

```text
<bos><|turn>system
Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.<turn|>
<|turn>user
{"evidence": {"sms": "WINNER!! You have won a $1000 gift card. Claim it now: bit.ly/xyz", "sender": "+1-555-0199"}, "criterion": "Is this message spam?", "options": [{"letter": "A", "description": "Unsolicited bulk or scam message"}, {"letter": "B", "description": "Ordinary message"}]}<turn|>
<|turn>model
<|channel>thought
<channel|>
```

That prompt is 137 tokens with the checkpoint tokenizer. Prompt size scales with
the option list: the same state with 26 options renders to 513 tokens.

Reproduce the render with `transformers`:

```python
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("shisa-ai/shisa-de-1")
payload = {
    "evidence": state,
    "criterion": "Is this message spam?",
    "options": [
        {"letter": "A", "description": "Unsolicited bulk or scam message"},
        {"letter": "B", "description": "Ordinary message"},
    ],
}
prompt = tok.apply_chat_template(
    [
        {"role": "system", "content": SYSTEM_LINE},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

`tokenize=False` plus `enable_thinking=False` matches the harness that produced
the published JevBench numbers. `enable_thinking=True` renders a different
prompt and is not part of this readout.

## 4. The answer slots

Each option letter must be exactly one token, and the ids are checkpoint
specific, so resolve them from the served tokenizer rather than hardcoding them.
For `shisa-ai/shisa-de-1`:

| Letter | Token id | Letter | Token id | Letter | Token id | Letter | Token id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `A` | 236776 | `H` | 236814 | `O` | 236806 | `V` | 236847 |
| `B` | 236799 | `I` | 236777 | `P` | 236791 | `W` | 236824 |
| `C` | 236780 | `J` | 236863 | `Q` | 236935 | `X` | 236917 |
| `D` | 236796 | `K` | 236855 | `R` | 236794 | `Y` | 236874 |
| `E` | 236788 | `L` | 236798 | `S` | 236773 | `Z` | 236953 |
| `F` | 236811 | `M` | 236792 | `T` | 236774 | | |
| `G` | 236823 | `N` | 236797 | `U` | 236836 | | |

Check the single-token requirement before sending anything:

```python
ids = tok.encode("A", add_special_tokens=False)
assert len(ids) == 1 and tok.decode(ids) == "A"
```

**Boundary check.** Appending a letter to the prompt must add exactly that
letter's token:

```python
prompt_ids = tok.encode(prompt, add_special_tokens=False)
for letter in letters:
    combined = tok.encode(prompt + letter, add_special_tokens=False)
    assert combined == prompt_ids + [slot_ids[letter]]
```

If that fails, the client's tokenizer is not the served checkpoint's tokenizer,
and every letter logprob would be read from the wrong position. Fail loudly
instead. `Readout.check_boundary` runs this on every question, and
`shisa-de doctor` runs it before you depend on an endpoint.

## 5. Reading the distribution

At the answer position, `top_logprobs[0]` is keyed by decoded token text. Take
the option letters, drop everything else, and softmax over what remains.

Worked example: the 2-option spam question above, one request, 137 prompt
tokens.

| Token | Logprob | In the option list |
| --- | --- | --- |
| `A` | -0.0007664603181183338 | yes |
| `B` | -8.25076675415039 | yes |
| `1` | -8.62576675415039 | no, discarded |
| `S` | -10.56326675415039 | no, discarded |

Softmax over `A` and `B` alone:

| Letter | Option | Probability |
| --- | --- | --- |
| `A` | Unsolicited bulk or scam message | 0.999739 |
| `B` | Ordinary message | 0.000261 |

The answer is `A`. The client returns `spam`, the option key behind `A`, with
`0.999739` as the raw probability.

Two more measured cases, each one request:

| Question | Letters and logprobs | Probabilities | Answer |
| --- | --- | --- | --- |
| noul: "Does this message ask the reader to click a link?" | `A` -0.0014327033422887325, `B` -7.5014328956604 | `A` 0.999447, `B` 0.000553 | yes, 0.999447 |
| score: "How risky is acting on this message?" with levels `Safe`, `Suspicious`, `Dangerous` | `C` -0.0861106812953949, `B` -2.7111105918884277, `A` -7.2111105918884277 | `A` 0.000750, `B` 0.067496, `C` 0.931754 | `C`, weighted position 1.931000 |

A noul is a two-option choice: `A` is yes and `B` is no. A score is a choice
over ordered levels; the returned score is the probability-weighted position,
`sum(index * probability)`.

## 6. The fallback

A letter outside the returned top 20 has no logprob in the first response. The
readout fills the gap with one extra request per missing letter: send the same
prompt with the letter appended, and ask for that token's logprob with
`prompt_logprobs`.

```json
{
  "model": "shisa-ai/shisa-de-1",
  "prompt": "<the same prompt>Y",
  "max_tokens": 1,
  "temperature": 0,
  "prompt_logprobs": 0
}
```

```json
{
  "choices": [
    {
      "prompt_logprobs": [
        null,
        {"236874": {"logprob": -8.790605545043945, "rank": 26, "decoded_token": "Y"}}
      ],
      "text": ""
    }
  ],
  "usage": {"prompt_tokens": 514, "completion_tokens": 1, "total_tokens": 515}
}
```

The last entry of `prompt_logprobs` is the token at the answer boundary, keyed
by token id, with its logprob and its rank in the full distribution. `rank`
separates a letter that nearly won from one that is weak but far ahead of the
rest of the vocabulary: rank 2 means the letter nearly won, rank 26 out of
262,144 tokens means it did not.

Measured cost, one 26-option question with the spam state:

| Measure | Value |
| --- | --- |
| Options | 26 |
| Letters outside the top 20 | 6: `Q`, `T`, `U`, `V`, `W`, `Y` |
| Requests | 7 |
| Prompt tokens across requests | 3597 (513 for the first request, 514 for each of the six fallbacks) |
| Ranks of the recovered letters | `Q` 21, `T` 22, `W` 23, `U` 25, `V` 25, `Y` 26 |
| Answer | `D` at 0.398416 |

The full 26-letter distribution after the fallbacks: `D` 0.398416, `B`
0.241652, `A` 0.129347, `C` 0.100735, `E` 0.032704, and the remaining 21 letters
below 0.023 each.

Raising `logprobs` above 20 avoids some fallback requests at the cost of a larger
response body, but it does not remove the fallback: at 26 options, 6 of the 26
letters fell outside the top 20. The fallback path above works for every option
count.

## 7. From letters to answers

**noul.** `A` is yes, `B` is no. The answer value is `P(yes) = p_A`.

**choice.** The answer is the argmax letter's option key. `probabilities` is
keyed by option key, and `confidence` uses the statistic the hosted System One
API returns:

```text
confidence = (K * p_max - 1) / (K - 1)      # K = number of options, clipped to 0..1
```

For two options that is `2 * p_max - 1`; for the 2-option spam example, `0.999478`.
TypeSafe documents this formula and states that callers are not locked into it,
which is why the raw `probabilities` are returned beside it.

**score.** `legend` maps each level index back to its description, and
`probabilities` maps each level index to its probability. `score` is the
weighted position:

```text
score = sum(index * p_index)      # 0-based, so levels [Safe, Suspicious, Dangerous]
                                  # with C at 0.931754 gives 0.000750*0 + 0.067496*1 + 0.931754*2 = 1.931000
```

`classify` returns the level description at the argmax instead of the weighted
position, so an ordered scale reads as one of its levels. `decide` returns the
weighted position, matching the System One answer shape.

## 8. Calibration

Temperature scaling is applied after the softmax, per question type.

| Question type | Temperature | Effect |
| --- | --- | --- |
| noul | 1.69 | `p^(1/T) / (p^(1/T) + (1-p)^(1/T))` |
| choice, score | 1.90 | `p_i^(1/T)` renormalized over the options |

These were fitted by minimizing negative log-likelihood on dev splits of the
committed suites, on the local merged bf16 serving shape. On the test split
(n=7446) they moved expected calibration error from 0.054 to 0.042 for noul and
from 0.138 to 0.046 for choice. The argmax never changes; only the reported
probability does.

`calibrated=True` is the default for `classify` and `decide`, and every answer
records `calibrated` and `temperature`, so a threshold fitted against raw scores
is never mixed with tempered ones silently. Pass `calibrated=False` for raw
logprob-derived probabilities.

The fit belongs to a serving shape, not to the checkpoint alone. The temperatures
above were fitted on a local merged server; the hosted endpoint returns slightly
different logprobs (the same card example that scores 0.9973 in the model card
scores 0.999351 here). Re-fit before using these numbers to drive a gate.

## 9. Costs and limits

| Limit | Value | Source |
| --- | --- | --- |
| Options per question | 26 (`A` to `Z`) | The readout reads one letter per answer |
| Questions per request | 1 | One answer position per prompt |
| Prompt tokens, 2 options | 137 | Measured, spam example |
| Prompt tokens, 26 options | 513 | Measured, spam example |
| Requests per question | 1, plus 1 per letter outside the top-k | Measured: 7 requests for 26 options |
| Round trip, one question | 81 to 128 ms, median 100 ms | Measured from a workstation against the hosted endpoint |
| Completion tokens | 1 | `max_tokens: 1` |

Because one question costs one request, a 20-label multi-label head costs 20
requests. `DecisionModel` runs questions concurrently over a thread pool
(`max_workers`, default 8) and reports `usage["requests"]` and
`usage["input_tokens"]` per call.

## 10. Reference implementation

The whole readout, with no dependencies beyond `httpx` and `transformers`:

```python
import json, math, httpx
from transformers import AutoTokenizer

MODEL = "shisa-ai/shisa-de-1"
SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed "
          "option. Respond with only its uppercase letter, with no explanation or reasoning.")
tok = AutoTokenizer.from_pretrained(MODEL)

def ask(state, criterion, options, api_key, base_url="https://api.shisa.ai/openai"):
    payload = {"evidence": state, "criterion": criterion,
               "options": [{"letter": chr(65 + i), "description": d} for i, d in enumerate(options)]}
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def post(body):
        response = httpx.post(f"{base_url}/v1/completions", json=body, timeout=120,
                              headers={"Authorization": f"Bearer {api_key}"})
        response.raise_for_status()
        return response.json()

    letters = [chr(65 + i) for i in range(len(options))]
    data = post({"model": MODEL, "prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": 20})
    top = data["choices"][0]["logprobs"]["top_logprobs"][0]
    logprobs = {letter: top[letter] for letter in letters if letter in top}
    for letter in letters:
        if letter in logprobs:
            continue
        fallback = post({"model": MODEL, "prompt": prompt + letter, "max_tokens": 1,
                         "temperature": 0, "prompt_logprobs": 0})
        entries = fallback["choices"][0]["prompt_logprobs"][-1]
        logprobs[letter] = next(iter(entries.values()))["logprob"]

    peak = max(logprobs.values())
    weights = {letter: math.exp(value - peak) for letter, value in logprobs.items()}
    total = sum(weights.values())
    probabilities = {letter: weight / total for letter, weight in weights.items()}
    best = max(probabilities, key=probabilities.get)
    return options[ord(best) - 65], probabilities
```

## 11. Verification

```bash
shisa-de doctor                        # endpoint, served model id, boundary check
shisa-de explain --labels spam,ham     # every step of one readout, printed
python -m pytest tests/                # offline: rendering, slots, fallback accounting
SHISA_DE_LIVE=1 python -m pytest tests/test_live.py    # against a live endpoint
```

`shisa-de explain` prints the rendered prompt, the resolved slot token ids, the
top logprobs with the answer slots marked, the normalized distribution, and the
equivalent `curl`. It is the fastest way to confirm that a new endpoint behaves
like the one described here.

A deployment matches this page when: `/v1/models` lists the model id, the
boundary check passes, one question returns one letter, and the reported
`system_fingerprint` is recorded alongside any numbers you keep. Logprobs from a
different serving shape are close but not identical, so thresholds and
calibration carry over only after a re-fit.

### Reproducibility

A fixed prompt on the hosted endpoint returns a fixed distribution, with rare
exceptions. Measured on 2026-09-25 against `vllm-0.26.0-tp2-c1aff9a1`:

| Repeat | Question | Result |
| --- | --- | --- |
| 8 sequential calls, separate connections | noul | `A` logprob -0.0010584949 every time (raw `P(yes)` 0.9996646499) |
| 6 concurrent calls, identical prompts | choice | probabilities 0.9963790844 to 0.9963790854 (spread 1.0e-09) |
| 1 of 13 calls across processes | noul | raw `P(yes)` 0.9997384 instead of 0.9996646, tempered 0.99247 instead of 0.99128 |

The outlier is larger than batching noise and did not reproduce. Treat a
threshold that falls between 0.9913 and 0.9925 on this question as undecided
rather than as a stable verdict, and re-measure on the endpoint you deploy
against.
