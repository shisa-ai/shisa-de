"""Fit temperatures per read and question type from calib_collect.py output.

Each suite carries equal total weight within a cell. Temperatures are fitted on
the `fit` half by weighted NLL over a grid (0.5..32.0 step 0.01, the research
repository's protocol) and reported on the `test` half. Thoughts are too few to
halve by type, so they are also reported with 5-fold cross-validation.
"""
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUN = Path(sys.argv[1])
GATE, OPTION_CAP = 0.7, 26
GRID = [round(0.5 + 0.01 * i, 2) for i in range(3151)]


def temper(p, t):
    logs = [math.log(max(x, 1e-12)) / t for x in p]
    top = max(logs)
    w = [math.exp(x - top) for x in logs]
    total = sum(w)
    return [x / total for x in w]


def weights(rows):
    count = Counter(r["suite"] for r in rows)
    return [1.0 / (count[r["suite"]] * len(count)) for r in rows]


def nll(rows, w, key, t):
    return -sum(wi * math.log(max(temper(r[key], t)[r["gold"]], 1e-12)) for r, wi in zip(rows, w))


def fit(rows, key):
    w = weights(rows)
    # Coarse then fine: the loss is unimodal in T.
    coarse = min(GRID[::25], key=lambda t: nll(rows, w, key, t))
    near = [t for t in GRID if abs(t - coarse) <= 0.25]
    return min(near, key=lambda t: nll(rows, w, key, t))


def metrics(rows, key, t=1.0, bins=15):
    w = weights(rows)
    acc = conf = loss = 0.0
    bucket = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r, wi in zip(rows, w):
        p = temper(r[key], t) if t != 1.0 else r[key]
        top = max(p)
        ok = p.index(top) == r["gold"]
        acc += wi * ok
        conf += wi * top
        loss -= wi * math.log(max(p[r["gold"]], 1e-12))
        b = bucket[min(bins - 1, int(top * bins))]
        b[0] += wi
        b[1] += wi * ok
        b[2] += wi * top
    ece = sum(abs(b[1] - b[2]) for b in bucket.values())
    return {"n": len(rows), "suites": len({r["suite"] for r in rows}), "acc": round(acc, 4),
            "conf": round(conf, 4), "ece": round(ece, 4), "nll": round(loss, 4)}


def kind(r):
    return "noul" if r["type"] == "noul" else "choice"


reads = [r for r in map(json.loads, (RUN / "reads.jsonl").open()) if "error" not in r]
thoughts = [r for r in map(json.loads, (RUN / "thoughts.jsonl").open()) if "thought" in r]
errors = sum(1 for line in (RUN / "reads.jsonl").open() if '"error"' in line)
kept = [r for r in reads if max(r["double"]) >= GATE or r["n"] > OPTION_CAP]
report = {"rows": {"reads": len(reads), "read_errors": errors, "thoughts": len(thoughts),
                   "gated": len(reads) - len(kept), "suites": len({r["suite"] for r in reads})}, "cells": {}}

for name, rows, key in (("single", reads, "single"), ("double", reads, "double"),
                        ("double_kept", kept, "double"), ("thought", thoughts, "thought")):
    for k in ("noul", "choice"):
        cell = [r for r in rows if kind(r) == k]
        train, test = [r for r in cell if r["half"] == "fit"], [r for r in cell if r["half"] == "test"]
        if len(train) < 30 or len(test) < 30:
            report["cells"][f"{name}/{k}"] = {"skipped": f"fit {len(train)}, test {len(test)}"}
            continue
        t = fit(train, key)
        report["cells"][f"{name}/{k}"] = {"T": t, "scale": round(1 / t, 3), "fit_n": len(train),
                                          "test_raw": metrics(test, key), "test_fitted": metrics(test, key, t)}

# Thoughts: one temperature across types, 5-fold cross-validated.
if len(thoughts) >= 50:
    fold = lambda r: int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 5  # noqa: E731
    held = []
    for f in range(5):
        t = fit([r for r in thoughts if fold(r) != f], "thought")
        held += [{**r, "cv": temper(r["thought"], t)} for r in thoughts if fold(r) == f]
    report["thought_all"] = {"T_all_rows": fit(thoughts, "thought"), "raw": metrics(thoughts, "thought"),
                             "cross_validated": metrics(held, "cv"),
                             "same_rows_double_read": metrics(thoughts, "double"),
                             "types": dict(Counter(kind(r) for r in thoughts)),
                             "hit_budget": sum(r["thought_tokens"] >= 1024 for r in thoughts),
                             "closed": sum(bool(r["thought_closed"]) for r in thoughts)}

# By option count, to see whether one choice temperature holds across widths.
for name, lo, hi in (("2", 2, 2), ("3-8", 3, 8), ("9-26", 9, 26), ("27-256", 27, 256)):
    cell = [r for r in reads if kind(r) == "choice" and lo <= r["n"] <= hi]
    if len(cell) >= 60:
        report.setdefault("single_choice_by_width", {})[name] = {"T": fit(cell, "single"), **metrics(cell, "single")}
        report.setdefault("double_choice_by_width", {})[name] = {"T": fit(cell, "double"), **metrics(cell, "double")}

(RUN / "fit.json").write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps(report, indent=1))
