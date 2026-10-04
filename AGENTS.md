# shisa-de — Agent Guide

Client library for Shisa DE-1 and DE-2 decision models. This file exists to prevent the
three mistakes that would make the library quietly wrong: changing the readout
without bumping its version, writing a number into a doc that nobody
measured, and applying one checkpoint's calibration to another.

## Summary

- `shisa_de` sends typed questions to a served DE-1 or DE-2 and reads the
  answers back. It never loads weights and never generates prose.
- DE-1 and DE-2 share the letter-slot readout but not a calibration. The record
  is chosen from the served model id (`shisa_de/family.py`): an explicit DE-1
  slug is DE-1, anything else is assumed to be DE-2 and reported as an
  assumption. A record and a model that disagree on family or readout version
  fail `doctor`.
- `docs/READOUT.md` is the contract. The code implements that page; the page
  does not describe the code. If they disagree, the page is right and the code
  is a bug — unless the page itself was wrong, in which case both change
  together.
- Source of truth for the readout identity: `shisa_de/readout.py`
  (`READOUT_VERSION`) and `docs/READOUT.md`.
- Every answer records its provenance (`readout_version`, `calibration`,
  `temperature`, `calibrated`, `requests`) so a caller can tell which readout
  produced it.

## Project Overview

Two layers, one contract:

- `shisa_de/readout.py` — the raw readout. Render a prompt, send one request,
  read the option letters, fall back for letters outside the top-k, softmax.
- `shisa_de/client.py` — the friendly layer. `classify` for label sets,
  `decide` for typed questions. Text choice overflow uses `shisa_de/overflow.py`
  above the unchanged per-prompt readout.

What must not change casually:

- The system line, the user JSON keys (`evidence`, `criterion`, `options`), the
  `enable_thinking=False` render, and the request body (`max_tokens: 1`,
  `temperature: 0`, `logprobs: 20`). These are the measured scaffold. Changing
  any of them changes answers, so bump `READOUT_VERSION` and update
  `docs/READOUT.md` in the same commit.
- The empty-description rule: a label with no description renders as the label.
  Blank descriptions give the model indistinguishable options.
- The boundary check. It is the only thing standing between a tokenizer
  mismatch and silently reading the wrong distribution.

## Key Files

| File | Purpose |
| --- | --- |
| `docs/READOUT.md` | The readout contract: request, prompt, slots, fallback, arithmetic, measured costs |
| `README.md` | The API surface users see first; links to the readout page |
| `shisa_de/readout.py` | Rendering, slot resolution, boundary check, requests, fallback, softmax |
| `shisa_de/client.py` | `DecisionModel.classify`, `DecisionModel.decide`, `Decision`, `Answer` |
| `shisa_de/overflow.py` | Balanced chunks, top-one finalists, conditional score maps; text choices only |
| `shisa_de/questions.py` | `Noul`, `Choice`, `Score` and their wire shapes |
| `shisa_de/calibration.py` | Temperature scaling, the confidence statistic, and record selection by family |
| `shisa_de/family.py` | Reads DE-1/DE-2 from a served model id, and says whether that was declared or assumed |
| `shisa_de/data/calibration.json` | Fitted DE-1 temperatures with their provenance |
| `shisa_de/data/calibration-de2.json` | Fitted DE-2 temperatures; currently a placeholder, and says so |
| `shisa_de/cli.py` | `shisa-de doctor`, `ask`, `explain` |
| `tests/test_offline.py` | Stubbed tokenizer and mock transport; no network |
| `tests/test_live.py` | Opt-in live checks, including the model card's 161-token example |

## Workflow Expectations

### Before Starting

- Read `docs/READOUT.md` for anything that touches rendering, slots, requests,
  or scoring.
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
- `READOUT_VERSION` identifies the answer-producing contract, not the client
  build or the contents of `readout.py`. Bump it only when rendering, requests,
  slot handling, or scoring semantics change, and update `docs/READOUT.md` in
  the same commit. Do not bump it for implementation-only fixes such as locks,
  lazy initialization, refactoring, or error handling that leave that contract
  unchanged. Track those changes with the package version. Editing
  `readout.py` alone is never a reason to bump the readout version.
- Any number added to a doc was measured, and the doc says when and against
  which serving fingerprint.

## Verification

| Scope | Commands |
| --- | --- |
| Offline suite | `python -m pytest tests/` |
| Live suite | `SHISA_DE_LIVE=1 python -m pytest tests/test_live.py -v` |
| Endpoint check | `shisa-de doctor` |
| Readout walkthrough | `shisa-de explain --labels spam,ham` |
| Packaging | `python -m build` (optional), `pip install -e .` |

The live suite spends real requests and needs `SHISA_API_KEY` for the hosted
endpoint. It is skipped unless `SHISA_DE_LIVE=1`.

## Evidence Rules

- Measured numbers carry a date and the serving fingerprint
  (`vllm-0.26.0-tp2-c1aff9a1` for the hosted endpoint on 2026-09-25). Logprobs
  move between serving shapes; a number without its fingerprint is not
  reproducible.
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
- `READOUT_VERSION` lives in one place. Do not define a second copy.

## Blockers

- If the served checkpoint's tokenizer is unavailable, the readout cannot run
  and the boundary check cannot pass. Stop and report; do not guess token ids.
- If a live result contradicts `docs/READOUT.md`, treat the doc as the spec and
  the result as a finding. Report the discrepancy with the fingerprint rather
  than editing the doc to match one run.
