# Changelog

## 0.3.0

- Select the calibration record from the served model id: an id carrying a DE-1
  slug (`shisa-de-1`, `de-1`, `de1-…`) uses the DE-1 record, anything else is
  assumed to be DE-2. `model_family` and `family_is_explicit` expose that
  reading, and a non-DE-1 id with no slug is reported as an assumption rather
  than silently treated as DE-1.
- Ship `data/calibration-de2.json`. It is a **placeholder**: fitted in-sample on
  the 192-row Japanese calibration partition, on the in-process Transformers
  readout rather than through vLLM. Its `choice` temperature was withheld as
  degenerate and ships as `1.0`, so `calibrated=True` is a no-op on DE-2 choice
  questions; its `noul` temperature is `1.36`. Refit before relying on it.
- Carry provenance in the calibration identity: `id` is now
  `model|readout_version|serving_shape[temperatures]`, so an answer records the
  checkpoint its temperatures were fitted on, not only the numbers.
- `doctor` now checks that the record and the served model agree on family and
  readout version, prints the fitted model, readout and serving shape, and fails
  when they disagree. Previously `ok` depended only on the id appearing in
  `/v1/models`, so a DE-1 record applied to a DE-2 endpoint passed.
- Add `--calibration <family|path>` to `doctor`, `ask` and `explain`, and
  `calibration_for`, `load_calibration_file`, `resolve_calibration` and
  `calibration_from_dict` to the API.

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
