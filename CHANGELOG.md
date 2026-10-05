# Changelog

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
