# Changelog

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
