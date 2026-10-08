# Decision Index 0.3 with shisa-de

How `shisa-ai/shisa-de-2` was run on the public
[Decision Index](https://github.com/apolinario/decision-index) 0.3 suite through
this client, and how to run it again. The kit owns the suite, the scoring and
the index; this repository owns how the model reads each question. The 0.2.1
protocol notes that used to be on this page are in the git history.

Results, with every request's answer: <https://huggingface.co/datasets/shisa-ai/decision-index-0.3-shisa-de-2>.

## Results

Measured on 2026-10-08 and 2026-10-09 against `vllm-0.30.0-50f64a6f` (vLLM 0.30.0, merged BF16
weights, `max_model_len` 262,144), one RTX PRO 6000 Blackwell per shard, two
shards, kit commit `62d2f51`, client v0.4.0. All three runs answered all 140,620
requests (140,178 scoreable) with no errors and no unsupported requests.

| Run | Policy | Public index | Raw index | Median | Mean | 80th percentile |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `shisa-de-2-adaptive` | `repeat-think` (default) | 59.01 | 68.68 | 47 ms | 770 ms | 332 ms |
| `shisa-de-2-repeat` | `repeat` | 57.57 | 67.65 | 45 ms | 169 ms | 144 ms |
| `shisa-de-2-single` | `direct` | 54.64 | 65.62 | 35 ms | 102 ms | 79 ms |

Latency is per suite request, measured by the client with one request in flight
per server. A request can carry several questions, each read separately. It is
not the maintainers' latency measurement, which they take themselves.

The public index is 20% of the board's Full score. The maintainers run the
private 80%.

## The three policies

The engine takes `--option policy=`, which is the client's `policy=`:

| Policy | What is sent per question |
| --- | --- |
| `repeat-think` | The user turn written twice. If that read's top probability is under 0.7 and the question has at most 26 options, the model thinks for up to 1,024 tokens and is read again. |
| `repeat` | The user turn written twice. No thinking. |
| `direct` | The user turn written once. No thinking. |

All three are the `de2-codebook-v1` contract in [READOUT-DE2.md](READOUT-DE2.md).
The engine asks for raw probabilities (`calibrated=False`), so the calibration
record does not affect the answers.

## What the engine does

`scripts/decision_index_engine.py` wraps `DecisionModel.decide`:

- A suite `choice` becomes a `Choice`, with its options in suite order. Up to
  256 options are read in one prompt; more is reported as unsupported.
- A suite `noul` becomes a `Noul` with the standard Yes/No options. Its optional
  criteria descriptions are not rendered.
- Every prompt is counted against the server's context window before any
  request is sent. A request that does not fit is reported as unsupported.
  Nothing is truncated and no shorter read is substituted.
- Only a context-limit error from the server counts as unsupported. Any other
  HTTP error stays an error, and the kit retries it on resume.
- `environment.json` records the policy, the readout version, the serving
  manifest, and a SHA-256 of the engine and every client source file.

## Run it

You need a GPU that holds the model with its 262,144-token context (the runs
here used 96 GB cards), vLLM 0.30.0, and access to
[`shisa-ai/shisa-de-2`](https://huggingface.co/shisa-ai/shisa-de-2).

### 1. Install

```bash
git clone https://github.com/shisa-ai/shisa-de
git clone https://github.com/apolinario/decision-index
git -C decision-index checkout 62d2f51

cd shisa-de
python -m venv .venv
.venv/bin/pip install -e . -e "../decision-index[rebuild]"
```

The engine and the run script are in `scripts/`, which is not part of the PyPI
package, so run from a checkout.

### 2. Build the suite

The kit does not redistribute the suite; it rebuilds it from pinned sources.
These are the kit's own commands (about 7 GB of downloads; accept the
[HLE](https://huggingface.co/datasets/cais/hle) terms first). The suite used
here was built earlier and verified against the 0.3 hashes, not rebuilt for
these runs.

```bash
export HF_HUB_DISABLE_XET=1
.venv/bin/python -m decision_index suite rebuild --work work
.venv/bin/python -m decision_index suite import \
    --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
    --gsm8k-rows work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz
```

That leaves the suite in `suite-0.3/`.

### 3. Serve the model

No special options. vLLM takes the 262,144-token context from the model.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve shisa-ai/shisa-de-2 --host 127.0.0.1 --port 8026
```

For a second shard, start another server on another GPU and port.

### 4. Run and score

```bash
.venv/bin/python scripts/decision_index_run.py \
    --suite-dir suite-0.3 \
    --endpoint http://127.0.0.1:8026 \
    --policy repeat-think \
    --out runs/shisa-de-2-adaptive
```

- Pass `--endpoint` once per server. The suite is split round-robin across them,
  one sequential process each, and merged before scoring.
- `--policy repeat` and `--policy direct` give the other two runs.
- `--limit 20` stops each shard after 20 requests, to check the setup.
- Running the same command again resumes; finished requests are kept.

The run directory ends with `results.jsonl`, `environment.json`, `scores.json`,
`index.json` and `benchmark-summary.json`. `scores.json` holds the public index
and must say `"complete": true`.

Wall time on two servers was about 15 hours for `repeat-think`, 3.3 hours for
`repeat` and 2 hours for `direct`.

To score an existing results file again:

```bash
.venv/bin/python -m decision_index score --edition 0.3 --suite-dir suite-0.3 \
    --results runs/shisa-de-2-adaptive/results.jsonl --engine shisa-de-2-adaptive
```

## Submitting

The kit's README has the procedure. In short:

1. Upload the run directory to a Hub dataset so that `runs/<name>/scores.json`
   exists.
2. Open a pull request on the kit adding a line to `submissions/README.md` with
   the model, the results link, the engine and commit, the hardware and the
   exact settings.
3. The maintainers re-score the results file, then run the model themselves on
   the private parts and measure latency on one RTX PRO 6000. They need the
   weights and this engine; a model whose median, mean or 80th-percentile
   latency is over 1,000 ms per request is not added to the board.

The kit's README says a separate Reasoning board is planned for models that
choose to think before answering. `repeat-think` is such a policy; `repeat` and
`direct` are not.
