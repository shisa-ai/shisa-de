# The DE-2 readout

DE-2 answers the same typed questions as DE-1, through the same scaffold, with a
different read. This page is the contract for that read. It states what changes
from [the DE-1 readout](READOUT.md) and assumes that page for everything it does
not restate: the system line, the user JSON, the empty-description rule, the
softmax, and how letters become typed answers.

- Readout version: `de2-codebook-v1`
- Maximum options per prompt: 256 for a text choice; 26 for a noul, a score, or
  any question with an image
- Requests per question: 1 up to 128 options, 2 up to 256, plus 2 when the
  question thinks
- Answer position: the first token after `<channel|>`

Measured numbers on this page, unless a line says otherwise, come from the LoRA
adapter `de2-v4-lr5e5-e3-s7` on `google/gemma-4-26B-A4B-it` (snapshot
`4d7ae4984b7d`), served by vLLM 0.30.0 on one GPU in bf16 with the Triton
attention backend, fingerprint `vllm-0.30.0-709530de`, on 2026-10-06. No DE-2
checkpoint is hosted or released yet, so these describe a candidate on a local
server. Re-measure on the endpoint you deploy against.

## What changes from DE-1

| | DE-1 (`de1-letter-slots-v3`) | DE-2 (`de2-codebook-v1`) |
| --- | --- | --- |
| Scaffold | System line, user JSON, empty thought block | The same, byte for byte |
| User turn | Written once | Written twice ([section 2](#2-the-repeated-read)) |
| Answer codes | `A` to `Z` | `A` to `Z`, then 230 two-letter codes ([section 1](#1-the-codebook)) |
| Reading the codes | Top 20 logprobs, one fallback request per missing letter | Every code requested by token id ([section 3](#3-reading-the-codes)) |
| Wide text choices | 27 to 676 through chunked finalist selection | 27 to 256 in one prompt; above 256 rejected |
| Thinking | Never | When the first read is unsure ([section 4](#4-the-thinking-read)) |
| Calibration | Bundled temperatures for `shisa-ai/shisa-de-1` | None fitted ([section 7](#7-calibration)) |
| Images | One direct read on the chat endpoint | The same ([section 6](#6-images)) |

Up to 26 options the two contracts render the same one-pass prompt: the codes
are the letters and the payload key is still `letter`. The DE-2 training
scaffold and this client's render matched string for string on a 26-option
question (checked 2026-10-05).

## The policy in one page

| # | Step | Detail |
| --- | --- | --- |
| 1 | Render the repeated prompt | The scaffold with the user JSON written twice |
| 2 | Read the codes | One request per 128 candidates, naming each code's token id |
| 3 | Normalize | Softmax over the option codes only |
| 4 | Gate | Stop here unless the top probability is below 0.7 and there are at most 26 options |
| 5 | Think | Render with thinking on, generate up to 1,024 tokens, stop at `<channel|>` |
| 6 | Read again | The same codes, after the thought and a `<channel|>` |
| 7 | Answer | The highest-probability code of the last read, mapped back to its option |

`DecisionModel(policy=...)` selects how far down this table a question goes:

| Policy | Steps | `Answer.strategy` |
| --- | --- | --- |
| `repeat-think` (default) | 1 to 7 | `repeat2`, or `repeat2-think` when the question thought |
| `repeat` | 1 to 3 | `repeat2` |
| `direct` | The one-pass scaffold, then 2 and 3 | `direct` |

The research repository adopted `repeat-think` on 2026-10-04. On its
9,888-question Decision Index sample the policy scored 72.94% against 70.23% for
one pass (+2.71 points, standard error 0.30), with 7.0% of questions thinking.
Those are the research notes' numbers, measured in-process on vLLM 0.30.0 with
gates chosen on the same questions; this repository has not reproduced them.
Source: `research-jev-universal-classifiers`, `DE-2.md`, "Inference policy".

## 1. The codebook

Option `i` is shown code `i` of a fixed list of 256: `A` to `Z`, then two-letter
uppercase codes in lexical order, `AA`, `AB`, ... `IW`. The list is data, not a
rule: it is the pairs that pass a single-token answer-boundary audit on the
Gemma tokenizer, so `GZ` is absent because it fails there. It ships as
`shisa_de/data/codebook-de2.json` and is the list DE-2 was trained with (source:
research repository, `evals/reports/de2-token-code-audit.json`,
`candidate_codebook_256`).

The payload is unchanged apart from the codes:

```json
{"evidence": "...", "criterion": "...", "options": [
  {"letter": "A", "description": "..."},
  {"letter": "Z", "description": "..."},
  {"letter": "AA", "description": "..."}
]}
```

Each code must be one token, and the ids are resolved from the served tokenizer,
never read from the data file. For the tokenizer above:

| Code | Token id | Code | Token id |
| --- | --- | --- | --- |
| `A` | 236776 | `AA` | 8686 |
| `B` | 236799 | `AB` | 3066 |
| `Z` | 236953 | `IW` | 99543 |

`A` and `AA` are different tokens at the one scored position, so they do not
collide. A code that is not one token raises `ReadoutError`; its first letter is
never scored in its place.

**Boundary check.** Every DE-2 prompt ends on `<channel|>` (token 101), and
`<channel|>` is a control token, so a code appended to it tokenizes the same
whatever precedes it. The check therefore has two halves:

```python
prompt_ids = tok.encode(prompt, add_special_tokens=False)
assert prompt_ids[-1] == close_id                       # every read
assert tok.encode("<channel|>" + code, add_special_tokens=False) == [close_id, code_id]   # once per code
```

The DE-1 form re-tokenizes the whole prompt once per letter, which a 256-option
question cannot afford. `Readout.check_code_boundary` runs this form on every
read, and `shisa-de doctor` runs it over all 256 codes.

## 2. The repeated read

The user message is the JSON object, a fixed separator, and the same JSON object
again. The separator, byte for byte, is two newlines, the sentence, one newline:

```text
\n\nRead the same input again before answering:\n
```

The complete rendered prompt for the 2-option spam question of the DE-1 page:

```text
<bos><|turn>system
Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.<turn|>
<|turn>user
{"evidence": {"sms": "WINNER!! You have won a $1000 gift card. Claim it now: bit.ly/xyz", "sender": "+1-555-0199"}, "criterion": "Is this message spam?", "options": [{"letter": "A", "description": "Unsolicited bulk or scam message"}, {"letter": "B", "description": "Ordinary message"}]}

Read the same input again before answering:
{"evidence": {"sms": "WINNER!! You have won a $1000 gift card. Claim it now: bit.ly/xyz", "sender": "+1-555-0199"}, "criterion": "Is this message spam?", "options": [{"letter": "A", "description": "Unsolicited bulk or scam message"}, {"letter": "B", "description": "Ordinary message"}]}<turn|>
<|turn>model
<|channel>thought
<channel|>
```

That prompt is 238 tokens; the one-pass prompt is 137. The system line and the
`enable_thinking=False` render are DE-1's. Do not serve DE-2 with a chat
template that repeats the input itself and also use this client: the turn would
be written four times.

## 3. Reading the codes

```http
POST {base_url}/v1/completions
Content-Type: application/json

{
  "model": "de2-v4-lr5e5-e3-s7",
  "prompt": "<the rendered prompt>",
  "max_tokens": 1,
  "temperature": 0,
  "logprobs": 20,
  "logprob_token_ids": [236776, 236799],
  "return_tokens_as_token_ids": true
}
```

- `logprob_token_ids` names every candidate, so each code's logprob is returned
  whatever its rank. vLLM requires `logprobs` to be set beside it.
- `return_tokens_as_token_ids` keys the returned map by `token_id:<id>`, so a
  code is matched by id rather than by decoded text.
- vLLM accepts at most 128 ids. A question with more options sends the same
  prompt once per block of 128, in codebook order, and one softmax runs over the
  union. The logprobs are over the whole vocabulary, so blocks combine exactly.

Response, trimmed:

```json
{
  "choices": [{"text": "A", "logprobs": {"top_logprobs": [
    {"token_id:236776": -2.9802276912960224e-06, "token_id:236799": -13.12500286102295}
  ]}}],
  "usage": {"prompt_tokens": 238, "completion_tokens": 1}
}
```

Softmax over the two codes gives `A` 0.999998 and `B` 0.000002.

**Prompt-token check.** `usage.prompt_tokens` must equal the number of tokens
the client's tokenizer produces for the prompt. A different count means the
client and the server disagree about the tokenizer, and the read raises
`ReadoutError`. This is the DE-2 guard against the mismatch the DE-1 read
catches by matching decoded text.

**Servers without `logprob_token_ids`.** A server that ignores the field returns
its ordinary top 20; one that rejects it with HTTP 400 or 422 is asked again
without it, and the client stops sending it for the rest of the session. Either
way the codes missing from the top-k are filled by the DE-1 fallback: one
request per code, `prompt + code`, `prompt_logprobs: 0`. The answer is the same;
the cost is not. `Answer.missing_from_top` lists the codes that needed it.

The logprobs returned for named ids equal the top-k logprobs for the same
tokens: on a 26-option question the largest difference across the 20 letters
both requests returned was 0.0.

Measured cost on the spam state, policy `repeat`:

| Options | Requests | Prompt tokens |
| --- | --- | --- |
| 2 | 1 | 238 |
| 26 | 1 | 990 |
| 77 | 1 | 2,622 |
| 256 | 2 | 17,324 (8,662 per request) |

Two more reads of the spam state, each one request:

| Question | Codes and logprobs | Probabilities | Answer |
| --- | --- | --- | --- |
| noul: "Does this message ask the reader to click a link?" | `A` -3.576e-06, `B` -13.000004 | `A` 0.999998, `B` 0.000002 | yes, 0.999998 |
| score: "How risky is acting on this message?" with levels `Safe`, `Suspicious`, `Dangerous` | `C` -0.047850, `B` -3.297850, `A` -8.922850 | `A` 0.000135, `B` 0.037322, `C` 0.962544 | `C`, weighted position 1.962409 |

## 4. The thinking read

A question thinks when both hold, on the raw distribution of the repeated read:

- its top probability is below the gate, 0.7;
- it has at most 26 options.

`DecisionModel(think_gate=..., think_budget=...)` change the gate and the
budget. The option cap is fixed. The research notes report no gain from thinking
above 26 options or from budgets above 1,024.

**The thought.** Render the question once, not twice, with thinking on and this
system line in place of the direct one:

```text
Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Reason through the question step by step before you answer. When your reasoning is complete, respond with only its uppercase letter.
```

With thinking on, the template writes `<|think|>` at the top of the system turn
and ends the prompt at `<|turn>model\n`; the model opens and writes the thought
itself.

```json
{
  "model": "de2-v4-lr5e5-e3-s7",
  "prompt": "<the thinking prompt>",
  "max_tokens": 1024,
  "temperature": 0,
  "stop_token_ids": [101],
  "return_token_ids": true
}
```

`choices[0].token_ids` holds the generated tokens. Keep those before the first
`<channel|>` (101); if there is none, the budget cut the thought and all of them
are kept. A server that returns no `token_ids` raises `ReadoutError`: the
thought has to be continued token for token, and re-tokenizing its text is not
that.

**The read after it.** Send the thinking prompt's tokens, the thought's tokens,
and one `<channel|>`, as a token-id prompt, with the code request of section 3:

```json
{"model": "...", "prompt": [2, 105, "...", 101], "max_tokens": 1, "temperature": 0,
 "logprobs": 20, "logprob_token_ids": [236776, 236799, 236780, 236796],
 "return_tokens_as_token_ids": true}
```

The client always writes the closing `<channel|>` itself, so a thought the
budget cut is closed the same way as one the model finished. That distribution
is the answer, whatever the first read said.

Measured, on "What is 17 * 23 - 111?" with options 270, 280, 290, 391:

| Read | `A` | `B` | `C` | `D` |
| --- | --- | --- | --- | --- |
| Repeated read | 0.1761 | 0.6543 | 0.1654 | 0.0041 |
| After a 260-token thought, closed by the model | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

Three requests, 779 prompt tokens in total, about 2.0 seconds against roughly
0.04 seconds for a question that does not think. The same question returned a
first-read `B` of 0.6402 on one of three runs: logprobs move a little between
batches, so a question near the gate can think on one call and not the next.

## 5. Answers and accounting

Typed answers are built exactly as on the DE-1 page. A wide DE-2 choice is an
ordinary choice: `probabilities` is a distribution over every option,
`confidence` is defined, and `to_wire()` has no extension fields. Nothing is
conditional on finalists, because nothing was eliminated.

| Field | Meaning |
| --- | --- |
| `Answer.strategy` | `repeat2`, `repeat2-think` or `direct` |
| `Answer.stages`, `logical_reads` | Code reads the answer took: 1, or 2 when it thought |
| `Answer.requests` | HTTP requests: reads, blocks past 128, the thought, any fallbacks |
| `Answer.thought_tokens`, `thought_closed` | Set when the question thought |
| `result.raw[head]["components"]` | Each read's logprobs and probabilities, in order |
| `result.raw[head]["thought"]` | The thought's text, with `debug=True` |
| `result.usage["thought_tokens"]` | Thought tokens across the call |
| `result.usage["output_tokens"]` | Logical reads plus thought tokens; an estimate, not server billing |
| `result.meta` | `family`, `readout_version`, `policy`, `think_gate`, `think_budget`, `think_option_cap` |

## 6. Images

An image question is one direct read on `/v1/chat/completions`, exactly as on
the DE-1 page: no repeat, no thought, at most 26 options, no fallback, and
`strategy` is `direct`. `readout_version` is still `de2-codebook-v1`.

Measured: 32x32 solid red, green and blue PNGs were each classified correctly
among `red`, `green`, `blue` at probability 1.0000, one request and 362 input
tokens each.

**The system turn is not rendered as the text scaffold renders it.** vLLM hands
Gemma 4's chat template every message of an image request as content parts, and
the template writes a space after a system message that arrives that way. The
image prompt therefore carries one token the text prompt does not, after the
system line. DE-2 was trained on the string rendering, without images. Each
image read records what it saw in `result.raw[head]["system_render"]`
(`string`, `differs` or `unverified`), and `shisa-de doctor --probe` reports it
for the endpoint.

Serving with `--chat-template-content-format string` removes the space from
text chat requests and does not help here: on this vLLM every image request then
fails with HTTP 500, "Failed to apply prompt replacement". So the client records
the difference and answers. `Readout.read_image(..., require_string_system=True)`
makes it an error, for a server whose template has been fixed.

## 7. Calibration

No temperature has been fitted for `de2-codebook-v1`. The bundled DE-2 record
holds 1.0 for both question types, every DE-2 answer has `calibrated=False`, and
`calibrated=True` returns raw probabilities rather than raising.

One fit has been attempted and is recorded, not applied, under `withheld` in
`shisa_de/data/calibration-de2.json`. It was made on the one-pass scaffold
rather than the repeated read, in-sample on 576 questions from 192 Japanese
rows where the checkpoint is at ceiling, through Transformers rather than a
server. Its noul optimum, 1.36, moved in-sample expected calibration error from
0.0233 to 0.0263, and its choice optimum sat on the grid floor.

To calibrate DE-2, fit on this readout, on the serving shape, with a held-out
split, write a record whose `readout_version` is `de2-codebook-v1`, and pass it
as `DecisionModel(calibration=...)` or `--calibration path.json`. A record fitted
against a DE-1 readout does not apply; [the DE-1 page](READOUT.md#8-calibration)
has the rule. An answer read after a thought is never tempered, whatever record
is supplied: no fit covers that read.

## 8. Serving

What the client needs from a vLLM server, and what it does not:

| Need | Why |
| --- | --- |
| `/v1/completions` with `logprobs`, `logprob_token_ids`, `return_tokens_as_token_ids` | The code read. The default `--max-logprobs 20` is enough at every width |
| `return_token_ids` and `stop_token_ids` | The thinking read |
| Token-id prompts | The read after a thought |
| `prompt_logprobs` | Only for the fallback on a server without `logprob_token_ids` |
| `/v1/chat/completions` with images, `top_logprobs` and `prompt_logprobs` | Image questions only |

The research notes report that vLLM 0.30.0's default attention kernel diverges
from Transformers on Gemma 4 and that `--attention-backend TRITON_ATTN` matches
it (72.10% against 72.12% on a 4,552-question sample, where the default scored
69.35%). The client cannot see which backend a server runs. The measurements on
this page used Triton.

The server used here:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-26B-A4B-it --served-model-name gemma-4-26b-a4b-it \
  --enable-lora --max-lora-rank 16 --lora-modules de2-v4-lr5e5-e3-s7=/path/to/adapter \
  --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN
```

A 256-option question on the repeated read was 8,662 tokens here, so size
`--max-model-len` for the widest question you send.

## 9. Wide choices against DE-1's overflow

DE-1 reaches past 26 options by splitting a choice into chunks and asking a
final question among the chunk winners. DE-2 does not: it was trained on the
codebook, and one read returns a full distribution.

On 200 rows of the research repository's Banking77 evaluation set (77 options,
seed-7 shuffle, criterion "Which banking intent best describes this customer
message?"), the same DE-2 weights:

| Read | Accuracy | Requests per question | Prompt tokens per question |
| --- | --- | --- | --- |
| DE-1's `finalist-top1` rule | 0.800 | 25.5 | 12,622 |
| `direct` | 0.835 | 1 | 1,360 |
| `repeat` | 0.840 | 1 | 2,685 |
| `repeat-think` | 0.845 | 1 | 2,685 |

The first row was measured on 2026-10-05 at fingerprint `vllm-0.30.0-feefc094`
and the rest on 2026-10-06. No question thought, because all have 77 options, so
the last two rows are the same prompts: their one-row difference is serving
noise, and the differences among the three DE-2 rows are not resolved (`repeat`
minus `direct` is 0.005, standard error 0.017). The cost difference against the
finalist rule is the finding; 200 rows do not rank the reads.

## 10. Reference implementation

The repeated read, without the thinking step:

```python
import json, math, httpx
from transformers import AutoTokenizer

MODEL = "de2-v4-lr5e5-e3-s7"
SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed "
          "option. Respond with only its uppercase letter, with no explanation or reasoning.")
REPEAT = "\n\nRead the same input again before answering:\n"
tok = AutoTokenizer.from_pretrained("google/gemma-4-26B-A4B-it")
CODES = json.load(open("shisa_de/data/codebook-de2.json"))["codes"]

def ask(state, criterion, options, base_url="http://127.0.0.1:8021"):
    codes = CODES[:len(options)]
    user = json.dumps({"evidence": state, "criterion": criterion,
                       "options": [{"letter": c, "description": d} for c, d in zip(codes, options)]},
                      ensure_ascii=False)
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": REPEAT.join([user, user])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    ids = [tok.encode(code, add_special_tokens=False) for code in codes]
    assert all(len(i) == 1 for i in ids) and prompt_ids[-1] == tok.encode("<channel|>", add_special_tokens=False)[0]
    ids = [i[0] for i in ids]

    logprobs = {}
    for start in range(0, len(ids), 128):
        data = httpx.post(f"{base_url}/v1/completions", timeout=120, json={
            "model": MODEL, "prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": 20,
            "logprob_token_ids": ids[start:start + 128], "return_tokens_as_token_ids": True}).json()
        assert data["usage"]["prompt_tokens"] == len(prompt_ids)
        top = data["choices"][0]["logprobs"]["top_logprobs"][0]
        logprobs.update({code: top[f"token_id:{i}"] for code, i in zip(codes[start:], ids[start:start + 128])})

    peak = max(logprobs.values())
    weights = {code: math.exp(value - peak) for code, value in logprobs.items()}
    total = sum(weights.values())
    probabilities = {code: weight / total for code, weight in weights.items()}
    best = max(probabilities, key=probabilities.get)
    return options[codes.index(best)], probabilities
```

## 11. Verification

```bash
shisa-de doctor --model de2-v4-lr5e5-e3-s7 --tokenizer google/gemma-4-26B-A4B-it --probe
shisa-de explain --model de2-v4-lr5e5-e3-s7 --tokenizer google/gemma-4-26B-A4B-it --labels spam,ham
SHISA_DE_LIVE=1 SHISA_DE_ENDPOINT=http://127.0.0.1:8021 SHISA_DE_MODEL=de2-v4-lr5e5-e3-s7 \
  SHISA_DE_TOKENIZER=google/gemma-4-26B-A4B-it python -m pytest tests/test_live.py -v
```

`doctor` lists the served models, resolves all 256 codes and checks their
boundary, without asking the model anything. `--probe` adds one question through the policy, which fails the report
if the server cannot serve the read, and one text-only chat request that reports
how the server renders the system turn. `explain` prints the first read of the
policy: the repeated prompt, the slots, the request, and whether the gate would
send the question on to a thought.

A deployment matches this page when `/v1/models` lists the model id, the
boundary check passes over the codebook, the probe read passes, and the
`system_fingerprint` is recorded beside any numbers you keep.

Not established here: the policy's accuracy on anything but the research
sample, latency under load, behaviour on a hosted DE-2 endpoint, and whether the
image path's system-turn space changes image answers.
