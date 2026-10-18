# Changelog

## 0.6.0

Support Shisa DE-2.1 and make it the default model. DE-2.1 is read through the
DE-2 contract unchanged: the readout versions, the prompts, the requests and the
policy are the same, and `de2-codebook-v1` still identifies the read. What is new
is a calibration record per released checkpoint.

- **Breaking:** `DEFAULT_MODEL` (and `--model`'s default) is now
  `shisa-ai/shisa-de-2.1`. Pass `model="shisa-ai/shisa-de-2"` to keep reading
  DE-2, which keeps its own record.
- Bundle `shisa_de/data/calibration-de2.1.json`, temperatures per read for
  `shisa-ai/shisa-de-2.1`: 1.98 / 1.53 read once, 2.11 / 1.64 read twice, 5.23
  after thinking (noul / choice), against DE-2's 2.69 / 1.81, 2.76 / 1.89 and
  6.8. Measured on 2026-10-19 on the release candidate's merged weights before
  upload (local vLLM 0.30.0, merged BF16, Triton attention, one RTX PRO 6000),
  19,597 questions from 59 suites. DE-2.1 is less overconfident than DE-2, and
  DE-2's temperatures over-correct it; see `docs/READOUT-DE2.md` section 7.
- Choose the bundled record by checkpoint: `CHECKPOINT_CALIBRATION_FILES` maps
  `shisa-ai/shisa-de-1`, `shisa-ai/shisa-de-2` and `shisa-ai/shisa-de-2.1` to
  their records, matched on the model id or, for an aliasing server, the
  tokenizer name. Any other checkpoint gets its family's record, reported and
  not applied, as before. `calibration_for(family, model=, tokenizer=)` takes
  the served model, and `--calibration` also accepts a checkpoint id.
- Validation: `python -m pytest tests/` (255 passed, 14 skipped). The live
  suite was not run against DE-2.1.

## 0.5.1

Add an opt-in state normalization for quote delimiters. The readout versions,
the prompts the client sends, the requests, and the bundled calibration are
unchanged, and the normalization is off by default.

- Add `strip_state_quotes=False` to `DecisionModel` and
  `--strip-state-quotes` to the CLI. When set, delimiter quote characters are
  removed from the edges of state strings before the prompt is rendered, so
  `"text"`, `"text`, `text"`, `""text` and `text""` all embed as `text` while
  interior quoting such as `He said "hi"` is kept. Measured through this client
  against `shisa-ai/shisa-de-2` on Shisa API on 2026-10-15, three reads per
  variant: the bare text `What are my balances currently?` and the quoted
  `"What are my balances currently?"` were classified differently in every
  read, and a stray trailing quote moved the answer further than a stray
  leading one. Brackets, braces and parentheses did not move it.
- Export `strip_quote_delimiters` and `QUOTE_DELIMITERS` for callers that render
  prompts themselves, and add the same option to the DE-2 trainer's renderer in
  the training repository.
- Record the setting in `result.meta` and in the request configuration, so
  answers read with it are not compared with answers read without it.

## 0.5.0

Make DE-2 the default model on Shisa API and standardize the service name in
package metadata and user documentation.

- Default `DecisionModel` and CLI usage to `shisa-ai/shisa-de-2` at
  `https://api.shisa.ai/openai`; explicit DE-1 and endpoint overrides remain
  supported.
- Add the Decision Index runner and document the measured direct, repeat, and
  repeat-think runs.
- Update offline fixtures to declare the DE-1 or DE-2 contract they exercise.

The DE-1 and DE-2 readout versions are unchanged; this release changes the
hosted default and release documentation, not either answer-producing contract.

## 0.4.0

Read settings by part, per model and per call, and a first DE-2 calibration.
Both readout versions are unchanged: every setting selects a read 0.3.0 already
had, and DE-2's raw distributions and chosen options are the same.

- Calibrate `shisa-ai/shisa-de-2` by default. The DE-2 record now holds
  temperatures per read (`direct`, `repeat2`, `think`), fitted on 2026-10-08, and
  the client applies each only to the read it was fitted on, including answers
  read after a thought. **DE-2 probabilities and confidences on that model are
  lower than in 0.3.0**; pass `calibrated=False` for the raw ones. Other DE-2
  checkpoints stay raw, with a warning.
- Calibration records may carry a `reads` table;
  `Calibration.temperature_for(type, read)` reads it. Records without one behave
  as before.
- Add `scripts/calibrate_de2_collect.py` and `scripts/calibrate_de2_fit.py`, the
  collection and fit behind the record.

- Add `reads` (`"single"` / `"double"`), `reasoning`, `reasoning_prob`,
  `reasoning_len` and `compound` to `DecisionModel`. `policy`, `think_gate`,
  `think_budget` and `overflow` remain and mean the same thing; passing both
  names with different values is an error.
- Accept `policy`, `reads`, `reasoning`, `reasoning_prob`, `reasoning_len` and
  `compound` on `classify` and `decide` to override the model's defaults for one
  call. `result.meta` records what the call ran under.
- Reject combinations neither contract defines: a double read or reasoning on
  DE-1, `compound=True` on DE-2, reasoning after a single read, and
  `reasoning=True` with `probability=True`.
- Add `--reads`, `--reasoning` / `--no-reasoning`, `--reasoning-prob`,
  `--reasoning-len` and `--compound` / `--no-compound` to the CLI. `doctor
  --json` reports `reads`, `reasoning` and `compound`.
- Expose `ReadOptions`, `resolve_options` and `READS`.
- `DecisionModel.policy`, `think_gate`, `think_budget` and `overflow` are now
  read-only views of `DecisionModel.options`.

## 0.3.0

DE-2 support. DE-1 answers are unchanged: the DE-1 readout is still
`de1-letter-slots-v3`, with the same requests, prompts and scoring.

- Read DE-2 through its own contract, `de2-codebook-v1`, documented in
  `docs/READOUT-DE2.md`. Every question is rendered with the user turn written
  twice; when that read's top probability is below 0.7 and the question has at
  most 26 options, the model thinks for up to 1,024 tokens and is read again
  after the thought. `policy="repeat"` and `policy="direct"` stop earlier, and
  `think_gate` / `think_budget` move the gate and the budget.
- Read DE-2 text choices of up to 256 options in one prompt, from a pinned
  codebook of `A`–`Z` and two-letter codes, with a full distribution and a
  confidence. DE-1 keeps `finalist-top1` overflow.
- Request every DE-2 code's logprob by token id (`logprob_token_ids`), so a
  question costs one request up to 128 options and two up to 256, with no
  per-letter fallback. A server without that field is read from its top-k with
  the fallback.
- Pick the contract from the model's family: `family=` / `--family`, then a
  `de-1`/`de-2` slug in the model id, then one in the tokenizer source,
  otherwise DE-2 with a warning. `resolve_family`, `model_family` and
  `family_is_explicit` expose the reading.
- Apply a bundled calibration record only to the checkpoint it was fitted on.
  Another checkpoint of the same family gets raw probabilities and a warning; a
  record passed as `calibration=` is applied as given.
  `Calibration.applicability` returns the level and the reasons.
- Keep the DE-1 record valid under `de1-letter-slots-v3`: it was fitted against
  v1, and `CALIBRATION_COMPATIBLE` records that v2 and v3 left the direct text
  read unchanged.
- Ship the DE-2 calibration record unfitted. DE-2 answers are raw; the one fit
  attempted so far is kept in the record as provenance and not applied.
- `doctor` reports the family and its source, the readout version and policy,
  and how far the calibration matches; it fails on a record from another family
  or an incompatible readout. `doctor --probe` also sends one question and
  reports how the server renders the system turn on the chat path.
- Record `system_render` on every image read. vLLM renders a space after the
  system line on image requests, on hosted DE-1 as well; the client records it
  and answers, and `Readout.read_image(require_string_system=True)` makes it an
  error.
- `result.meta["calibrated"]` now says what the answers carry rather than what
  was requested, and `meta` gains `family` and `policy`. `usage` gains
  `thought_tokens`. The calibration identity is
  `model|readout_version|serving_shape[temperatures]`.
- `probability=True` (added in 0.2.1) skips the thinking read on DE-2 and
  accepts its wide choices, which one read covers; DE-1 behaviour is unchanged.
  `meta["policy"]` records the policy that ran.
- Add `--family`, `--policy` and `--calibration <family|path>` to the CLI, and
  point the live suite at any served model with `SHISA_DE_MODEL` and
  `SHISA_DE_TOKENIZER`.
## 0.2.1

- Add `probability=True` to `decide`, `system_one`, and `classify` to require a
  single logical read per question and reject choice overflow before requests.
- Record probability intent in decision metadata. Calibration, direct request
  bodies, return shapes, and the readout version are unchanged. Missing-letter
  recovery remains supported.

## 0.2.0

- Add `finalist-top1` overflow for 27–676-option text choices: balanced chunks,
  then one final choice among chunk winners. Use `overflow="error"` for strict
  rejection. Small choices, image reads and ordered-score limits are unchanged.
- Mark overflow scores as conditional on finalists, with no calibrated
  confidence; expose strategy, finalists, stage count and logical read usage.
- Expose text `max_logprobs` on `DecisionModel`, retaining the default of 20.
- Identify the expanded answer contract as `de1-letter-slots-v3`. No small-choice
  prompt or scoring change; bundled calibration is not applied to overflow.

## 0.1.2

- Serialize lazy tokenizer imports and loads across client instances to prevent
  concurrent first-call initialization races.
- Keep initialized tokenizer access and normal requests outside the lock, and
  allow initialization to be retried after a failure.

## 0.1.1

- Use `pip install shisa-de` in the README.
- Add this changelog.
- No API or scoring changes.

## 0.1.0

- Initial PyPI release.
- Classify text and images through Shisa Platform or a local DE-1 server.
- Ask typed questions with `Noul`, `Choice`, and `Score`.
- Add `doctor`, `ask`, and `explain` CLI commands.
