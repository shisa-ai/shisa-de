# Submitting DE-2 to Decision Index 0.2.1

This page records the Decision Index 0.2.1 protocol as the upstream
reproduction kit defines it, checks the kit's README against its implementation,
and lists the eligibility questions a public Shisa DE-2 submission has to
answer. Reviewed on 2026-10-06. This is a protocol reference, not a model score
or latency report.

Everything labelled **upstream** is a claim from the kit at commit
`87d4650b42b377c0291a89c1f1a879f9b31082bf` ("Version 0.2.1", 2026-09-27),
either its README or its code. Source links for every claim are collected in
[Upstream sources](#upstream-sources). **Local** marks something I read or
computed in this workspace.

- Kit: <https://github.com/apolinario/decision-index>, commit
  `87d4650b42b377c0291a89c1f1a879f9b31082bf`.
- Local suite copy: `/data/decision-index-de2-suite/suite-0.2/`.
- Model: `shisa-ai/shisa-de-2-rc-v7s13` release candidate (private), LoRA adapter
  on `google/gemma-4-26B-A4B-it`.
- Client: this repository, DE-2 contract `de2-codebook-v1`
  ([docs/READOUT-DE2.md](READOUT-DE2.md)).

## What the kit owns, and what it does not

**Upstream** is the reproduction kit: the frozen suite, the row and result
formats, the reference engines, the scorers, the index math, and the submission
procedure. The live board is Decision Index 0.2.1 (2026-09-27) and is the
default edition in the kit.

**Upstream** does not host or run the model. Every entrant on the board was run
with its author's own inference code. The kit ships two reference engines
(`http`, `transformers`) and a `random` sanity check; a model with its own
harness is wrapped in an `Engine` subclass and passed as
`--engine module:Class`. The scoring side does not depend on the engine.

**Local** therefore: the model's answers for a Shisa DE-2 submission come from
this repository's client (or a thin adapter over it), rendered with the DE-2
scaffold in [docs/READOUT-DE2.md](READOUT-DE2.md). The kit scores those answers;
it does not define how the model reads a question.

## Editions

| | 0.1 | 0.2 | 0.2.1 |
| --- | --- | --- | --- |
| Label | Decision Index 0.1 | Decision Index 0.2 | Decision Index 0.2.1 |
| Suite directory | `suite/` | `suite-0.2/` | `suite-0.2/` |
| Headline score | `balanced_raw` | `balanced_skill` | `balanced_skill` |
| Requests (before exclusions) | 132,422 | 121,057 | 120,340 |
| Scoreable | 131,980 | 120,615 | 119,898 |
| Added requests | 0 | 30,419 | 30,419 |
| Catalog benchmarks | 37 | 44 | 44 |
| Benchmarks in the index | 19 of 25 | 40 | 38 |

**Upstream**: 0.2.1 rescores 0.2 on the same row files, with the same hashes and
the same `suite-0.2/`. A complete 0.2 run is also a complete 0.2.1 run, and
`score --edition 0.2.1` rescores it. **Local** check: `editions.py` gives 0.2 and
0.2.1 identical `rows_gz_sha256`, `rows_sha256`, and `added_sha256`, so
`editions.compatible` returns true for the pair and `pipeline.py` takes the
0.2.1 rescoring path.

What 0.2.1 changes from 0.2 (all **upstream**, from the README "What changed in
0.2.1" list and `decision_index/data/index-0.2.1.json`):

- Area weights: Arts & Human Taste is fixed at 10%; the other four areas share
  90% in proportion to the square root of their benchmark count. The stated
  shares are Tools & Automation 18.3%, Retrieval & Classification 20.0%,
  Knowledge & Reasoning 25.8%, Language Understanding 25.8%.
- Gold ★ benchmarks weigh 1.2 inside their area, the rest 1.0.
- SGD and RouterBench leave the index (SGD pending a fixed builder, RouterBench
  because its prompt reveals the best route). Both stay on the board as
  shown-not-counted.
- ACOS is scored by F1 per review instead of case-exact accuracy, and stays in
  the index.
- ToolRet and BRIGHT score only queries with at least one relevant candidate
  among the 32: 685 of 1,000 and 220 of 550, with chance recomputed on the same
  queries.
- RAGTruth chance is F1 of always answering "hallucinated", 0.5177, replacing
  0.4113 in 0.2.
- Home appliances drops 24 duplicate test rows and 48 rows identical to
  published dev rows: 88 of 160 rows stay.

0.2 differs from 0.2.1 in three scoring places (README, "0.2 differs in three
places"): RouterBench and SGD count, every area is the plain mean of its
benchmarks, and the five areas weigh the same. 0.1 is not chance-corrected and
its headline is `balanced_raw`.

**Local** arithmetic check of the 0.2.1 area weights: with Knowledge 10
benchmarks, Language 10, Retrieval 6, Tools 5, and Arts fixed at 0.1, the square
root sizing in `index02.area_weights` gives Knowledge 25.85%, Language 25.85%,
Retrieval 20.02%, Tools 18.28%, which round to the README's 25.8 / 25.8 / 20.0 /
18.3. The README table and the implementation agree.

## Frozen suite: hashes and counts

**Upstream** row files for 0.2 and 0.2.1 (from `decision_index/editions.py`;
the same values are in `hub/0.2.1/manifest.json`):

| File | SHA-256 |
| --- | --- |
| `selected-rows.jsonl.gz`, compressed | `25aac5e890a54a3172c7a0c184b4cc8b9a43f10b6ee89bbad8da923be423c656` |
| `selected-rows.jsonl.gz`, uncompressed | `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5` |
| `added-rows.jsonl.gz`, uncompressed | `7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89` |
| `excluded-questions.json` | `331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81` |
| `release-v2/acos-subset.json` | `3b20eea1613ae3e1644339e6f89a4287b307a5750a86239bb8bca6f8238b8e45` |
| `release-v2/retrieval-subsets.json` | `9228a9492ea40c499d8bb024417f566d0c0ced62c71ed4c1bf908e87ed50d8df` |
| `release-v2.1/toolret-subset.json` | `c301d5503955c14d99485e284d434bb1533f51229c34da1edc19dd8388116f01` |
| `release-v2.1/bright-subset.json` | `ed3a05220e09be2b96c02514345c61894f7a0ffc73e6884a100392d23692cfb8` |
| `release-v2.1/home-appliances-subset.json` | `d2d0df922a47a9757bef17f382a4ed4c5c3bfff953f941c29404e93a490bce2c` |

**Upstream** counts: `selected-rows.jsonl.gz` has 124,971 lines;
`added-rows.jsonl.gz` has 30,419; the exclusions file has 442 request ids. The
uncompressed hash is authoritative because a rebuilt gzip differs from the lab's
in its header, and `Suite.verify` falls back to the uncompressed hash when the
compressed one does not match.

**Local** verification against the local suite copy, read-only, no rebuild and
no download:

- `selected-rows.jsonl.gz` compressed hash is
  `3becb6a58ea12f30bb5053b48465d8fedc04a10f8b1695469847cab723383062`, which does
  not equal the pinned compressed hash; its uncompressed hash is
  `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5`, which
  matches. The differing compressed hash is accepted by the kit; the exact
  cause of the gzip byte difference was not investigated.
- `added-rows.jsonl.gz` uncompressed hash is
  `7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89`, matching.
- `excluded-questions.json` matches, 442 rows.
- Line counts match: 124,971 and 30,419.
- `manifest.json` reports `release-v2.1`, `requests` 120,340, `scoreable_requests`
  119,898, `added_requests` 30,419.
- Every subset file's SHA-256 matches the value pinned in `editions.py`.

So the local suite copy is the upstream 0.2.1 suite by the kit's own check.

### Noul question counts

**Upstream** `docs/format.md` defines the `noul` question as a yes/no probability
(`{"type": "noul", ...}`) answered as `{"type": "noul", "noul": p_yes}`, and
notes that every question in the base rows is a `choice`; only the seven
benchmarks added in 0.2 introduce `noul`.

**Local** count over the edition-filtered suite rows on 2026-10-06 (before
applying the 442 scoring exclusions):
| | Count | Notes |
| --- | ---: | --- |
| `choice` questions | 318,698 | After edition subsets |
| `noul` questions | 14,700 | |
| `noul` with non-empty `criteria` | 4,700 | RAGTruth 2,700; PhishNChips `is_phishing` 2,000 |
| `noul` with empty `criteria` | 10,000 | PhishNChips five `sig_*` questions, 2,000 rows each |
| `noul` actually scored | 2,700 | RAGTruth only |

PhishNChips (catalog 56) is scored on its `verdict` **choice** field only
(`FIELDS = {56: ("verdict",)}` in `scoring/added.py`); its `is_phishing` noul is
sent to the engine but not scored, and its five `sig_*` nouls are listed in
`scoring.unscored_fields`. So of the 4,700 noul questions that carry criteria,
2,700 are scored.

## Run rules

**Upstream** encodes these in the runner and engines (README "Rules"):

- **No truncation.** An engine that cannot fit a request raises `Unsupported`;
  the row is recorded `unsupported` and counts as wrong. Nothing is cut to fit.
- **No option filtering.** Every option in `criteria` gets a probability, or
  `validate` rejects the response.
- **No prompt tuning.** The reference engines use one fixed rendering for every
  benchmark.
- **Unanswered = wrong.** Unsupported, errored, abstained, and pending requests
  score zero before chance correction. The per-benchmark report shows the native
  metric on answered cases separately.
- **Linked cases stay whole.** Multi-request cases (ToolRet and BRIGHT chunks,
  RouterBench tracks, ACOS reviews) count only when every linked request
  succeeded.
- **Exclusions apply to everyone**, at scoring time; the frozen files are
  untouched.

**Local** confirmation against the implementation:

- `engines/base.py` `validate` requires the answer's question keys to equal the
  asked keys, the type to match, a `choice` answer's key to be one of
  `criteria`, and `choice` probabilities to cover exactly the criteria keys,
  be finite, lie in `[0, 1]`, and sum to 1 within 0.01. For `noul` it requires
  only that `noul` is finite and in `[0, 1]`.
- `engines/transformers_engine.py` raises `Unsupported` when prompt tokens plus
  the longest option exceed the context limit, and never truncates.
- `runner.py` records `unsupported` as a final status and retries only `error`
  rows on resume (`if rid in previous and previous[rid] != "error": continue`).
  Five errors before the first success stop the run; an out-of-memory or
  device-side assert also stops it.
- `scoring/report.py` scores a benchmark only on complete case groups
  (`all(... == "ok")`), which is the linked-case rule.

## Exclusions

**Upstream** `hub/excluded-questions.json` drops 442 request ids at scoring time
for every engine, so no model is graded on a case another model was structurally
unable to attempt. The reasons and counts:

| Reason | Count | Rule summary |
| --- | ---: | --- |
| `beyond_nearly_every_context_window` | 380 | Rows longer than almost every entrant can accept; 14 of 27 measured sweeps answer none, 4 answer all. |
| `sibling_chunk_of_an_excluded_group` | 35 | Surviving chunks of a partly excluded ToolRet query are removed with it. |
| `duplicate_options` | 17 | Two or more options are byte-identical. |
| `gold_duplicated` | 6 | The correct answer is one of the identical strings. |
| `whitespace_duplicate_options` | 3 | e.g. MuSR "pantry" and "pantry ". |
| `gold_whitespace_duplicated` | 1 | The gold is the space-variant. |

**Upstream** deliberately keeps CRUXEval case/whitespace collisions and the
MMLU/ARC/GPQA notation collisions, which are semantic. Its stated criterion: a
row is excluded for reach only when nearly the whole field cannot attempt it; a
limit one model has and others do not is that model's result, not a suite
defect.

## Scoring

**Upstream** writes three files at the end of `score` or `pipeline`:

- `benchmark-summary.json`: the native metric of every benchmark on its complete
  supported case groups, with answered/unsupported/error counts.
- `index.json`: per benchmark `raw`, `skill`, `coverage`, chance level; the five
  areas; `index` and `raw_index`.
- `scores.json`: both combined, plus edition, suite hashes, counts, latency, and
  the completion flag. This is the file a submission points at.

**Upstream** index math (`scoring/index02.py`, panel in
`data/index-0.2.1.json`):

1. Coverage first: `raw = native score × answered / requests`. Unanswered,
   unsupported, errored, and abstained requests count as wrong. Benchmarks
   carried from the 0.1 panel keep track-level scoring, where an unanswered
   group scores zero inside the metric. ACOS is scored by per-review F1,
   averaged over reviews.
2. Chance correction: `skill = clip((raw − chance) / (1 − chance), 0, 1)`.
   Chance is per track for GSM8K, per query for ToolRet and BRIGHT, F1 of always
   answering "hallucinated" for RAGTruth (0.5177), per-review F1 of answering yes
   to every ACOS pair (0.031), all-fields-right by chance for BFCL, SATA-Bench,
   and Home appliances, F1 of random guessing for the other F1 benchmarks, and
   the mean of 1/options otherwise. ForecastBench enters against its baseline:
   `clip((0.25 − Brier) / 0.25) × coverage`.
3. Areas and index: inside an area, gold benchmarks weigh 1.2 and the rest 1.0.
   Arts & Human Taste weighs 10%; the other four share 90% in proportion to the
   square root of their benchmark count. `index = 100 × weighted mean of the five
   areas`. `breadth_skill` uses the same weights in a geometric mean. `raw_index`
   substitutes `raw` for `skill`. MMLU, ARC-Easy, ARC-Challenge, SimpleBench,
   RouterBench, and SGD are scored and shown but not counted. Scores within 0.25
   index points of the next are tied.

**Local** confirmation: `index02.area_weights` implements the sqrt sizing with a
`fixed` map (`{"arts": 0.1}`); `bench_weights` reads `gold` with a default of 1.0;
`loss_value` implements the ForecastBench baseline rule; `added_value` applies
`score × answered / requests` and the chance correction. The 38 index benchmarks
are the 10 + 10 + 6 + 5 + 7 benchmark ids in `index-0.2.1.json`.

## Complete-run rule and artifacts

**Upstream** `docs/format.md`: a run is complete when every scoreable request of
the edition has a result. For 0.2.1 that is 119,898 + 30,419 = 150,317 result
rows. `pipeline.py` sets `"complete": completed >= expected` with
`expected = scoreable + added_requests`, and `completed` counts rows present in
`results.jsonl` under the edition's read-time subsets.

**Local** consequence: a "complete" run can contain `unsupported`, `error`, and
`abstained` rows. Those rows count as wrong but do not block completion, because
the check counts a result row's presence, not its status. The submission rule
that matters is the one in the README: runs must be complete and untouched, and
any declared capacity limits are fine as long as nothing was truncated.

**Upstream** expected artifacts in a run directory, and what `--upload` sends:
`results.jsonl.gz`, `benchmark-summary.json`, `index.json`, `scores.json`,
`environment.json`, `status.json`.

**Local** trap: the runner writes a `status.json` event `"complete"` when it
exits its loop, including after a `--limit N` early break. `--limit` therefore
produces a run whose `status.json` says complete but whose `scores.json`
`"complete"` flag is false. Use `scores.json`, not `status.json`, to judge
completeness.

## Submission procedure

**Upstream** README, "Submitting a model to the leaderboard":

1. Run the full suite (`pipeline` or `hf-job`; a complete 0.2 run is also a
   complete 0.2.1 run) and upload the run directory to a Hub dataset
   (`--upload <you>/<repo>`; `hf-job` does this).
2. Open a pull request adding a line to `submissions/README.md` (create it if
   needed) with the model name, the results dataset link (which must contain
   `runs/<name>/scores.json`), the engine/commit used, and the hardware. Runs
   must be complete and untouched; the results file is re-scored on review.
3. Mention any declared capacity limits. They appear in `environment.json` and
   in the unsupported counts and are acceptable as long as nothing was truncated.

**Local** note: this repository has no `submissions/` directory; the pull request
targets the upstream repository, not `shisa-de`. Creating `submissions/README.md`
is an upstream change, outside this repository's scope.

### Latency threshold

**Upstream**: the maintainers measure latency themselves, single-process on one
RTX PRO 6000, using the fastest path the model's code supports, on a private
held-out sample. The same sample validates submitted runs by comparing answers.
Models whose median latency on that measurement is over 1,000 ms per request are
not added to the board: at that speed they are no longer Jev-like. The latency
in a run's own `scores.json` is a guide, not that measurement.

**Local** consequence: the threshold is applied by the maintainers on their
hardware and their sample, so a local latency number neither guarantees nor
disqualifies. `environment.json` records the `latency` definition each engine
declares; the upstream serial runner's `total_wall_ms` includes prompt
construction and the complete engine call. The separate research batched
runner writes shared chunk times; those are not per-request latency.

## Privacy and raw responses

**Upstream** `results.jsonl` includes `payload` (`state` and `questions`) and
`raw_output` for `ok` rows unless the run was started with `--compact`, which
drops both. `--compact` leaves `status`, `response`, and the timing fields.

**Local** risk: `state` is the suite text. Several upstream sources restrict
redistribution, and the README says to treat the suite as evaluation data under
those terms, not to train on it, and not to republish it. `--upload` defaults to
a private dataset (`upload_run(..., private=True)`; `hf-job` needs `--public` to
override), and the README tells you to keep the suite dataset private. A public
results dataset built without `--compact` would republish the state text and any
model output in `raw_output`. For a DE-2 run, `raw_output` may also contain the
bounded thought, which can quote the state.

Practical handling for a submission:

- Run with `--compact` so `results.jsonl` carries no `payload` and no
  `raw_output`.
- Keep the results dataset private until review, and prefer publishing only the
  files a submission needs (`scores.json` and the summary), or a private dataset
  plus a public scores link.
- The maintainers re-score the results file and compare answers against their
  held-out sample; `--compact` keeps `response`, which is what those checks need.

**Ambiguity**: the README's step 1 says to upload the run directory without
saying whether `--compact` is acceptable for submission. The re-scoring and
answer-comparison steps need only `response`, so `--compact` should be
sufficient, but upstream does not state this explicitly. Confirm before
publishing a public dataset.

## Proposed local run (not yet executed)

The intended setup, recorded so the eligibility notes have context. This is a
plan, not a measured result:

- Upstream serial runner (`decision_index run`/`pipeline`) with `--compact`.
- Answers produced by this repository's DE-2 `repeat-think` policy over a local
  merged BF16 vLLM server, context 32,768, overflow refused rather than
  truncated.
- Scoring with `decision_index score --edition 0.2.1` against the local suite
  copy, which passes the upstream 0.2.1 uncompressed hashes.

Two upstream-vs-research differences to resolve before this plan is final:

- **Overflow behaviour.** The upstream rule is to refuse a request that does not
  fit (`Unsupported`, counted wrong). The prior research runner
  (`scripts/run_decision_index_policy.py`) instead falls back: a read-twice
  prompt that would exceed the context window falls back to the single read, and
  a question whose thinking prompt would exceed it keeps its read-twice answer.
  The proposed run refuses, which matches the upstream rule. The fallback is a
  deviation that must be avoided or disclosed.
- **Latency fields.** The research runner's latency fields are shared chunk wall
  times, not per-request latency. If the local `scores.json` latency is reported,
  say so.

## Eligibility concerns

### Noul rendered as generic Yes/No

**Upstream** defines `noul` as a yes/no probability with optional `criteria`
([docs/format.md](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/docs/format.md)).
The two reference engines treat the criteria differently:

- `http` forwards `{"model", "state", "questions"}` unchanged, so the served
  model receives the criteria.
- `transformers` renders a `noul` with the fixed pair
  `{"false": "False", "true": "True"}` and never reads the row's criteria
  (`render_prompt`, line 11), answering with `probs["true"]`.

**Local** facts:

- This client's `Noul.options()` returns `[("yes", "Yes"), ("no", "No")]` and
  ignores `criteria` even when present; the readout renders options from
  `Question.options()`, so a noul's criteria never reach the prompt. `Noul.to_wire`
  does include criteria if given, but rendering does not use them.
- The prior research engine's `question_options` maps a `noul` to
  `[('yes', 'Yes'), ('no', 'No')]`, identically, and its `answer` requires the
  binary order `['yes', 'no']` with `noul = probabilities[0]`.

Assessment:

- Dropping noul criteria is **consistent with the upstream reference
  `transformers` engine**, which also ignores them and uses a fixed pair. It is
  not a new deviation introduced by this client, and the label names differ only
  as Yes/No versus True/False over the same boolean.
- It is a single fixed rendering applied to every noul question across all
  benchmarks, so it does not meet the "no prompt tuning" prohibition as written,
  which targets per-benchmark adaptation.
- The protocol does not state that an engine must render noul criteria, and the
  two reference engines disagree on whether they are forwarded. That is an
  ambiguity, not a clear rule.
- Scored impact is limited: only RAGTruth's 2,700 nouls are scored, and their
  `instructions` already state the proposition the criteria restate. The 10,000
  PhishNChips signal nouls carry empty criteria, so nothing is dropped for them.
  PhishNChips' scored field is a `choice` question (`verdict`).

Do not claim eligibility is guaranteed on this point. Disclose the generic
Yes/No rendering in the submission, note that it matches the upstream
`transformers` reference engine and the model's own scaffold, and let the
maintainers decide. Changing the noul rendering would require a readout-version
bump because it changes prompts; it is not necessary merely to reproduce the
existing model contract.

### Correction to the RC card's training-pool claim

On 2026-10-06, the model owner clarified that the multiple-review training-pool
requirement stated in the RC card at snapshot
`cf090e1ddbdf8a2370057fa74154728e1a0bddf0` was invented by the submitting agent,
not a project requirement. The owner is correcting the card. This document's
initial revision repeated that statement as a submission concern; that was an
error. Do not treat it as an admission rule or an eligibility blocker.

The checkpoint was private when downloaded. Public submission still requires
arranging model access for maintainers and owner approval to publish; those are
separate from the retracted review claim.

### Prior policy tuning on benchmark samples

**Local** facts from the research repository
(`/home/lhl/research-jev-universal-classifiers`, `DE-2.md`):

- The policy gate and option cap (think below 0.7, at most 26 options) were
  chosen on Decision Index questions, and the comparison table is computed on
  the same questions. `DE-2.md` says "Gates were chosen on the questions they
  are scored on" and "Computed from the saved answers of the three arms above,
  DE-2, 1,024-token budget" over the 9,888-question Decision Index sample. The
  adopted variant is reported at 72.94% on that sample, against 70.23% for the
  one-pass scaffold. The same notes say the policy "holds on weights it was not
  tuned on", meaning other checkpoints, not other question sets.
- The calibration scale 0.3 (temperature 10/3) was fitted on 3,094 binary
  validation questions. `DE-2.md` states the intent directly: "Fit the scale on
  our own held-out data, not on index questions."
- The repository contains overlap-audit scripts
  (`scripts/audit_decision_index_overlap.py`,
  `scripts/audit_decision_index_intent_overlap.py`) whose docstrings say the
  audit checks exact full-state overlap against DE-2 train pools and validation
  rows and "cannot establish sample-level training exposure or rule out
  paraphrases."

Risk: the adopted policy was selected on a Decision Index sample, so the policy
choice is in-sample with respect to that sample. If the 9,888-question sample is
inside the 0.2.1 suite, then part of the reported index comes from a policy
fitted to suite questions, which is the kind of benchmark-sample tuning the
submission rules discourage. The research notes call it "the index sample" and
report per-benchmark gains and losses from it, which suggests it is a subset of
the suite, but I did not find the sample file (`/data/screen-v2/items` is not
present) and neither the RC card nor `DE-2.md` states that it is disjoint from
the 0.2.1 scoring set. Treat the overlap as unverified. Run the overlap audit
against `/data/decision-index-de2-suite/suite-0.2/` and read the result before
claiming the index is out-of-sample; do not claim eligibility is guaranteed
either way.

### Other risks

- The RC card's local index numbers (53.84 / 56.86 / 58.13) are local runs, not
  board entries, and were produced with vLLM 0.30.0 on an H20-3e, not the
  board's RTX PRO 6000. They do not speak to the latency threshold.
- The RC card requires the Triton attention backend on vLLM 0.30.0, and
  `chat_template_content_format="string"`; serving with different settings can
  lower accuracy. The board's latency measurement uses the model's fastest path,
  so a run that changes these settings may score differently from the measured
  path.
- The research notes' thinking-only latency is not the latency of the gated
  `repeat-think` policy. Measure the submitted policy end to end, including
  conditional thoughts. A faster single-read policy would be a different
  submission configuration and must not supply the latency claimed for
  `repeat-think` scores. Maintainers determine the accepted implementation
  and independently check its answers and latency.

## Unresolved methodology questions

1. **Noul criteria.** Must an engine render a `noul` question's `criteria`, or is
   the fixed generic pair the intended rendering? The reference engines disagree
   (`http` forwards, `transformers` drops). Affects 4,700 questions, 2,700 of
   them scored.
2. **Yes/No versus True/False.** Both are generic; the answer is `p_yes` either
   way. Whether the board treats the label spelling as part of the model's
   scaffold or as fixed by the protocol is not stated.
3. **Benchmark-sample overlap.** Whether the 9,888-question Decision Index
   sample used to choose the 0.7/26 policy, and the 3,094 binary validation
   questions used to fit scale 0.3, are disjoint from the 0.2.1 scoring set. The
   notes state the policy gates were chosen on the questions they are scored on;
   the calibration was fitted on validation questions. Disjointness is not
   established by the documents read.
4. **`--compact` for submission.** Whether a compacted `results.jsonl` (no
   `payload`, no `raw_output`) is accepted for the review re-scoring and the
   held-out answer comparison. The checks appear to need only `response`, but
   the README does not say so.
5. **Model access.** If the checkpoint remains private, arrange maintainer
   access for independent answer and latency verification.
6. **Overflow fallback.** Whether the DE-2 policy's capacity fallback (read
   twice → single read, or keep the read-twice answer) is acceptable to the
   board, or whether every over-limit prompt must be refused as `Unsupported`.
   The proposed run refuses; the research runner falls back.
7. **Interactive panel.** Upstream keeps the six interactive environments
   (MiniWoB++, ScienceWorld, Boxoban, RTFM, Hanabi, Codenames) unrun and out of
   the index. No action is needed for a static submission, but a board entry has
   no results for them.

## Upstream sources

All links are pinned to commit
`87d4650b42b377c0291a89c1f1a879f9b31082bf`.

| Claim | Source |
| --- | --- |
| Editions, what changed in 0.2.1 and 0.2, rules, submission, latency, licence notes | [README.md](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/README.md) |
| Row and result formats, `noul`, complete-run rule | [docs/format.md](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/docs/format.md) |
| Sources, sampling, licences, 0.2.1 subsets | [docs/suite.md](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/docs/suite.md) |
| Engine contract, `Unsupported`, reference engines, board harnesses | [docs/engines.md](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/docs/engines.md) |
| Edition hashes and counts | [decision_index/editions.py](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/decision_index/editions.py) |
| Runner statuses, resume, error/device stops | [decision_index/runner.py](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/decision_index/runner.py) |
| Index math, area weights, chance correction, ForecastBench | [decision_index/scoring/index02.py](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/decision_index/scoring/index02.py) |
| 0.2.1 panel, gold weights, not-in-index list, chance levels | [decision_index/data/index-0.2.1.json](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/decision_index/data/index-0.2.1.json) |
| Linked-case completeness, native metrics | [decision_index/scoring/report.py](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/decision_index/scoring/report.py) |
| Added-benchmark scoring, PhishNChips scored field, RAGTruth F1 | [decision_index/scoring/added.py](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/decision_index/scoring/added.py) |
| `noul` rendered as fixed False/True, no truncation | [decision_index/engines/transformers_engine.py](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/decision_index/engines/transformers_engine.py) |
| `http` engine forwards questions unchanged | [decision_index/engines/http.py](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/decision_index/engines/http.py) |
| Response validation | [decision_index/engines/base.py](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/decision_index/engines/base.py) |
| Complete flag, upload defaults, artifacts | [decision_index/pipeline.py](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/decision_index/pipeline.py) |
| Exclusion reasons and counts | [hub/excluded-questions.json](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/hub/excluded-questions.json) |
| 0.2.1 manifest, subset hashes | [hub/0.2.1/manifest.json](https://github.com/apolinario/decision-index/blob/87d4650b42b377c0291a89c1f1a879f9b31082bf/hub/0.2.1/manifest.json) |
