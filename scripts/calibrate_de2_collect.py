"""Collect raw DE-2 distributions for a calibration fit, through the shisa_de client.

Phase 1 reads every sampled question once and twice (no thinking). Phase 2 runs
the thinking read on the questions the gate would send there (double-read top
probability < 0.7, at most 26 options). Rows are raw option-softmax
probabilities in option order, with the gold option's index.

    python scripts/calibrate_de2_collect.py OUT_DIR [CASES_PER_SUITE] [MAX_THOUGHTS]
    python scripts/calibrate_de2_fit.py OUT_DIR

The suites are `cases.jsonl` files in the research repository's layout
(`SHISA_DE_SUITES`, default `~/research-jev-universal-classifiers/evals/suites`).
The endpoint and model come from `SHISA_DE_ENDPOINT` and `SHISA_DE_MODEL`. Each
phase stops taking new work after 10 minutes; cases are shuffled first so the
cut falls on every suite alike. The 2026-10-08 record was collected with
`OUT_DIR 100000 100000` against a server started with `--max-num-seqs 64`.
"""
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from shisa_de import Choice, DecisionModel, Noul, Score, codes_for

SUITES = Path(os.environ.get("SHISA_DE_SUITES") or Path.home() / "research-jev-universal-classifiers/evals/suites")
OUT = Path(sys.argv[1])
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 300
THINK_CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
GATE, OPTION_CAP = 0.7, 26
WORKERS, PHASE_SECONDS = 64, 600


def build(q):
    if q["type"] == "noul":
        return Noul(str(q["instructions"]), criteria=q.get("options") or None)
    if q["type"] == "choice":
        return Choice(str(q["instructions"]), dict(q["options"]))
    return Score(str(q["instructions"]), list(q["levels"]))


def gold_index(q, gold):
    if q["type"] == "noul":
        return 0 if gold in (True, 1) else 1
    if q["type"] == "choice":
        return list(q["options"]).index(gold)
    return int(gold)


cases = []
for path in sorted(SUITES.glob("*/*/cases.jsonl")):
    if path.parent.name.endswith("-reversed"):
        continue
    rows = [json.loads(line) for line in path.open()]
    random.Random(13).shuffle(rows)
    kept = 0
    for case in rows:
        questions = {k: q for k, q in case["questions"].items() if case["gold"].get(k) is not None}
        if not questions or kept >= CAP:
            continue
        kept += 1
        cases.append((path.parent.name, case, questions))
random.Random(13).shuffle(cases)  # so a time cap cuts every suite alike
print("cases", len(cases), "questions", sum(len(q) for _, _, q in cases), flush=True)

de = DecisionModel.from_endpoint(os.environ.get("SHISA_DE_ENDPOINT") or "http://127.0.0.1:8027/v1",
                                 model=os.environ.get("SHISA_DE_MODEL") or "shisa-ai/shisa-de-2",
                                 api_key="", local_files_only=True, timeout=900, max_workers=1)


def probs(component, n):
    return [component["probabilities"][code] for code in codes_for(n)]


def phase1(item):
    if time.perf_counter() - started > PHASE_SECONDS:
        return []
    suite, case, questions = item
    built = {k: build(q) for k, q in questions.items()}
    try:
        single = de.decide(case["state"], built, reads="single", calibrated=False)
        double = de.decide(case["state"], built, reasoning=False, calibrated=False)
    except Exception as exc:  # noqa: BLE001
        return [{"suite": suite, "id": case["id"], "error": repr(exc)[:300]}]
    out = []
    for qid, q in questions.items():
        n = len(built[qid].options())
        out.append({
            "suite": suite, "id": case["id"], "qid": qid, "type": q["type"], "n": n,
            "language": case.get("language"), "gold": gold_index(q, case["gold"][qid]),
            "half": "fit" if int(hashlib.sha256(str(case.get("group") or case["id"]).encode()).hexdigest(), 16) % 2 else "test",
            "single": probs(single.raw[qid]["components"][0], n),
            "double": probs(double.raw[qid]["components"][0], n),
        })
    return out


started = time.perf_counter()
rows, errors = [], []
with ThreadPoolExecutor(WORKERS) as pool, (OUT / "reads.jsonl").open("w") as f:
    for done, batch in enumerate(pool.map(phase1, cases), 1):
        for row in batch:
            (errors if "error" in row else rows).append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if done % 500 == 0:
            print("phase1", done, round(time.perf_counter() - started), "s", flush=True)
print("phase1 done", len(rows), "rows", len(errors), "errors", round(time.perf_counter() - started), "s", flush=True)

gated = [r for r in rows if max(r["double"]) < GATE and r["n"] <= OPTION_CAP]
print("gated", len(gated), "of", len(rows), flush=True)
random.Random(13).shuffle(gated)
gated = gated[:THINK_CAP]
lookup = {(s, c["id"]): (c, q) for s, c, q in cases}


def phase2(row):
    if time.perf_counter() - started > PHASE_SECONDS:
        return None
    case, questions = lookup[(row["suite"], row["id"])]
    try:
        result = de.decide(case["state"], {row["qid"]: build(questions[row["qid"]])}, calibrated=False)
    except Exception as exc:  # noqa: BLE001
        return {**{k: row[k] for k in ("suite", "id", "qid")}, "error": repr(exc)[:300]}
    answer, raw = result.answers[row["qid"]], result.raw[row["qid"]]
    out = {k: row[k] for k in ("suite", "id", "qid", "type", "n", "gold", "half", "double")}
    out.update(strategy=answer.strategy, thought_tokens=answer.thought_tokens, thought_closed=answer.thought_closed,
               wall_ms=result.usage["wall_ms"])
    if answer.strategy == "repeat2-think":
        out["thought"] = probs(raw["components"][1], row["n"])
        out["double_again"] = probs(raw["components"][0], row["n"])
    return out


started = time.perf_counter()
with ThreadPoolExecutor(WORKERS) as pool, (OUT / "thoughts.jsonl").open("w") as f:
    for done, row in enumerate(pool.map(phase2, gated), 1):
        if row is None:
            continue
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        if done % 100 == 0:
            print("phase2", done, round(time.perf_counter() - started), "s", flush=True)
print("phase2 done", round(time.perf_counter() - started), "s", flush=True)
(OUT / "DONE").write_text("ok\n")
