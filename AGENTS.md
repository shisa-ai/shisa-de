# shisa-de — Agent Guide

Client library for Shisa DE-1 and DE-2 decision models. This file exists to prevent the
four mistakes that would make the library quietly wrong: changing a readout
without bumping its version, writing a number into a doc that nobody
measured, applying one checkpoint's calibration to another, and reading a
checkpoint through the other family's contract.

## Summary

- `shisa_de` sends typed questions to a served DE-1 or DE-2 and reads the
  answers back. It never loads weights. The only text it generates is DE-2's
  bounded thought, which is read past, not returned as an answer.
- Each family has its own contract and its own page. `docs/READOUT.md` is
  DE-1 (`de1-letter-slots-v3`); `docs/READOUT-DE2.md` is DE-2
  (`de2-codebook-v1`) and states only what differs. The code implements those
  pages; the pages do not describe the code. If they disagree, the page is
  right and the code is a bug — unless the page itself was wrong, in which case
  both change together.
- The family picks the contract: readout version, policy, option limit and
  calibration record. It is resolved in `shisa_de/family.py` from a declared
  `family=`, then the model id, then the tokenizer source, and otherwise assumed
  to be DE-2 with a warning. An assumption is always reported as one.
- Source of truth for readout identities: `shisa_de/readout.py`
  (`READOUT_VERSIONS`, one current version per family, and
  `CALIBRATION_COMPATIBLE`) and the two readout pages.
- A calibration record is applied on its own only to the checkpoint it was
  fitted on, under a readout its fit still describes. The DE-2 record holds one
  fit per read (`direct`, `repeat2`, `think`), each applied only to that read.
- Every answer records its provenance (`family`, `readout_version`, `policy`,
  `strategy`, `calibration`, `temperature`, `calibrated`, `requests`) so a
  caller can tell which read produced it.

## Project Overview

Two layers, two contracts:

- `shisa_de/readout.py` — the raw reads. DE-1: render a prompt, send one
  request, read the option letters, fall back for letters outside the top-k,
  softmax. DE-2: the same scaffold, codes from a 256-entry codebook requested by
  token id, an optional repeated user turn, and a bounded thought.
- `shisa_de/policy.py` — sequences the DE-2 reads: twice, then think when unsure.
- `shisa_de/client.py` — the friendly layer. `classify` for label sets,
  `decide` for typed questions. DE-1 text choice overflow uses
  `shisa_de/overflow.py` above the unchanged per-prompt readout; DE-2 reads wide
  choices natively and never overflows.

What must not change casually:

- The system line, the user JSON keys (`evidence`, `criterion`, `options`), the
  `enable_thinking=False` render, and the DE-1 request body (`max_tokens: 1`,
  `temperature: 0`, `logprobs: 20`). These are the measured scaffold, and DE-2
  shares all but the request body.
- For DE-2: the codebook and its order, the repeat separator, the thinking
  system line, the gate (0.7), the option cap (26), the budget (1,024), and the
  rule that the client writes the closing `<channel|>` after a thought.
- Changing any of the above changes answers, so bump that family's readout
  version and update its page in the same commit.
- The empty-description rule: a label with no description renders as the label.
  Blank descriptions give the model indistinguishable options.
- The boundary checks, and DE-2's prompt-token check. They are the only things
  standing between a tokenizer mismatch and silently reading the wrong
  distribution.

## Key Files

| File | Purpose |
| --- | --- |
| `docs/READOUT.md` | The DE-1 contract: request, prompt, slots, fallback, arithmetic, calibration rules, families, measured costs |
| `docs/READOUT-DE2.md` | The DE-2 contract: codebook, repeated read, code request, thinking read, serving, measured costs |
| `README.md` | The API surface users see first; links to both readout pages |
| `shisa_de/readout.py` | Rendering, slot and code resolution, boundary checks, requests, fallback, thought generation, softmax, readout versions |
| `shisa_de/policy.py` | The DE-2 policy: repeated read, gate, thought, read after it; `ReadOptions` and the per-model and per-call settings that select a read |
| `shisa_de/client.py` | `DecisionModel.classify`, `DecisionModel.decide`, `health`, `Decision`, `Answer` |
| `shisa_de/overflow.py` | Balanced chunks, top-one finalists, conditional score maps; DE-1 text choices only |
| `shisa_de/questions.py` | `Noul`, `Choice`, `Score` and their wire shapes |
| `shisa_de/calibration.py` | Temperature scaling, the confidence statistic, record selection, and `Calibration.applicability` |
| `shisa_de/family.py` | Resolves DE-1/DE-2 from a declaration, the model id or the tokenizer, and says which |
| `shisa_de/data/calibration.json` | Fitted DE-1 temperatures with their provenance |
| `shisa_de/data/calibration-de2.json` | The DE-2 record: temperatures per read with the fit's provenance, and one withheld earlier fit |
| `scripts/calibrate_de2_collect.py`, `calibrate_de2_fit.py` | Collect raw reads from a served DE-2 and fit the per-read temperatures |
| `shisa_de/data/codebook-de2.json` | The 256 DE-2 answer codes, in order, with the audit they came from |
| `shisa_de/cli.py` | `shisa-de doctor`, `ask`, `explain` |
| `tests/test_offline.py` | Stubbed tokenizers and mock transports, including a stub DE-2 server; no network |
| `tests/test_live.py` | Opt-in live checks for either family, including the model card's 161-token example |

## Workflow Expectations

### Before Starting

- Read `docs/READOUT.md`, and `docs/READOUT-DE2.md` for DE-2, for anything that
  touches rendering, slots, requests, policy, or scoring.
- `git status -sb`; the repo is small and commits are frequent.
- Check whether the endpoint you are testing against is the hosted one or a
  local vLLM, and record which one produced any number you keep.

### During Work

- One logical unit per commit. A readout change and its doc update are one unit.
- New behavior gets a test in `tests/test_offline.py` against the stub, and a
  live test only when the behavior depends on the served model.
- Keep the package importable without network access. Tokenizer loading stays
  lazy inside `Readout.ensure_tokenizer`.

### Before Claiming Done

- `python -m pytest tests/` passes.
- A readout version identifies one family's answer-producing contract, not the
  client build or the contents of `readout.py`. Bump it only when that family's
  rendering, requests, slot handling, policy, or scoring semantics change, and
  update its page in the same commit. Do not bump it for implementation-only
  fixes such as locks, lazy initialization, refactoring, or error handling that
  leave that contract unchanged. Track those changes with the package version.
  Editing `readout.py` alone is never a reason to bump a readout version.
- When you bump a version, decide its `CALIBRATION_COMPATIBLE` entry in the same
  commit. List the predecessors only if the direct text read is answer-identical
  to theirs; otherwise leave them out, which stops every record fitted against
  them from being applied until it is re-fitted. Never edit a calibration
  record's `readout_version` to make it match: it records what the fit was made
  against.
- Any number added to a doc was measured, and the doc says when and against
  which serving fingerprint.

## Verification

| Scope | Commands |
| --- | --- |
| Offline suite | `python -m pytest tests/` |
| Live suite, hosted DE-1 | `SHISA_DE_LIVE=1 python -m pytest tests/test_live.py -v` |
| Live suite, served DE-2 | add `SHISA_DE_ENDPOINT`, `SHISA_DE_MODEL`, `SHISA_DE_TOKENIZER` |
| Endpoint check | `shisa-de doctor`, and `shisa-de doctor --probe` to spend two requests on the reads |
| Readout walkthrough | `shisa-de explain --labels spam,ham` |
| Packaging | `python -m build` (optional), `pip install -e .` |

The live suite spends real requests and needs `SHISA_API_KEY` for the hosted
endpoint. It is skipped unless `SHISA_DE_LIVE=1`. Tests that assert one family's
contract skip on the other, so run it against both before changing shared code.

## Evidence Rules

- Measured numbers carry a date and the serving fingerprint
  (`vllm-0.26.0-tp2-c1aff9a1` for the hosted endpoint on 2026-09-25;
  `vllm-0.30.0-709530de` for the local DE-2 server on 2026-10-06). Logprobs
  move between serving shapes; a number without its fingerprint is not
  reproducible.
- The research repository's numbers are its own, not this repo's. Cite them
  with their source and say they were not reproduced here, as
  `docs/READOUT-DE2.md` does for the policy's accuracy.
- No DE-2 checkpoint is hosted or released. DE-2 numbers here come from a
  candidate adapter on a local server; say so wherever one appears.
- The model card's published numbers are the card's, not this repo's. When a
  card value and a local measurement disagree, report both and say which is
  which (the card's spam example scores 0.9973 there and 0.999351 on the hosted
  endpoint).
- Do not present a temperature-scaled probability as a raw one, or the reverse.
  `calibrated` is on every answer for that reason.

## Git Discipline

Conventional commit prefixes, imperative mood, subject at 72 characters or
fewer, body bullets for the change and the validation.

- Commit immediately when a logical unit is complete and verified.
- Never `git add .`, `git add -A`, or `git commit -a`. Stage exact paths.
- Leave unrelated work alone. Do not `git restore`, `git checkout --`,
  `git reset --hard`, or `git clean -fd` unless asked.
- A local commit does not authorize a push, tag, or release. Ask first.
- No AI attribution or co-author footers.

## Coordination Hygiene

- `docs/READOUT.md` and `README.md` are the high-conflict files; two agents
  editing them at once will conflict. Coordinate before editing either.
- Readout versions live in one place, `shisa_de/readout.py`: one per family in
  `READOUT_VERSIONS`. Do not define a copy elsewhere.

## Blockers

- If the served checkpoint's tokenizer is unavailable, the readout cannot run
  and the boundary check cannot pass. Stop and report; do not guess token ids.
- If a live result contradicts a readout page, treat the page as the spec and
  the result as a finding. Report the discrepancy with the fingerprint rather
  than editing the page to match one run.
