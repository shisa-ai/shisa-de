"""Run the Decision Index public suite through the shisa-de client and score it.

Needs the Decision Index kit importable (``pip install -e`` its checkout), a
built suite directory, and one local vLLM server per ``--endpoint``. The suite
is split round-robin across the endpoints, one sequential process each, then
the shard results are merged and scored. Re-running the same command resumes.

    python scripts/decision_index_run.py --suite-dir suite-0.3 \
        --endpoint http://127.0.0.1:8026 --policy repeat-think \
        --out runs/shisa-de-2
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from decision_index.runner import iter_rows
from decision_index.suite.io import Suite

import shisa_de

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "scripts.decision_index_engine:ShisaDE2Engine"


def split(suite_dir, edition, shards, out):
    """Write one row file per shard, round-robin in suite order."""
    suite = Suite(Path(suite_dir), edition)
    suite.verify(strict=True)
    paths = [out / f"shard{i}-rows.jsonl.gz" for i in range(shards)]
    if all(p.exists() for p in paths):
        return paths
    files = [gzip.open(p, "wt", encoding="utf-8", compresslevel=3) for p in paths]
    for count, row in enumerate(iter_rows(suite.row_paths, suite.in_edition)):
        files[count % shards].write(json.dumps(row, ensure_ascii=False) + "\n")
    for f in files:
        f.close()
    return paths


def manifest(endpoint, model, shard, shards, note):
    """Record what the server at this endpoint says about itself."""
    with httpx.Client(base_url=endpoint, timeout=60) as http:
        served = next(m for m in http.get("/v1/models").json()["data"] if m["id"] == model)
        probe = http.post("/v1/completions", json={"model": model, "prompt": "0", "max_tokens": 1}).json()
        version = http.get("/version")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    return {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint, "model": model,
        "max_model_len": served["max_model_len"],
        "system_fingerprint": probe.get("system_fingerprint"),
        "vllm": version.json().get("version") if version.status_code == 200 else None,
        "shisa_de": f"{shisa_de.__version__} ({commit or 'no git checkout'})",
        "shard": f"{shard} of {shards}, round-robin over the suite",
        "note": note,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--edition", default="0.3")
    ap.add_argument("--endpoint", action="append", required=True,
                    help="local vLLM base URL without /v1; repeat for one shard per server")
    ap.add_argument("--model", default="shisa-ai/shisa-de-2")
    ap.add_argument("--tokenizer", help="defaults to --model")
    ap.add_argument("--policy", default="repeat-think", choices=("direct", "repeat", "repeat-think"))
    ap.add_argument("--out", required=True, help="run directory; scores.json lands here")
    ap.add_argument("--name", help="engine name written to scores.json; defaults to the run directory's name")
    ap.add_argument("--limit", type=int, help="stop each shard after this many requests, for a trial run")
    ap.add_argument("--note", default="", help="free text for the serving manifest, such as the GPU and server command")
    a = ap.parse_args()

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    shards = len(a.endpoint)
    rows = split(a.suite_dir, a.edition, shards, out)

    runs = []
    for i, endpoint in enumerate(a.endpoint):
        record = manifest(endpoint, a.model, i, shards, a.note)
        path = out / f"serving-manifest-{i}.json"
        path.write_text(json.dumps(record, indent=1) + "\n")
        command = [
            sys.executable, "-m", "decision_index", "run", "--engine", ENGINE, "--model", a.model,
            "--option", f"tokenizer={a.tokenizer or a.model}", "--option", f"base_url={endpoint}",
            "--option", f"max_tokens={record['max_model_len']}", "--option", f"serving_manifest={path}",
            "--option", f"policy={a.policy}",
            "--rows", str(rows[i]), "--compact", "--out", str(out / f"shard{i}"),
        ]
        if a.limit:
            command += ["--limit", str(a.limit)]
        log = (out / f"shard{i}.log").open("a")
        runs.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
    if any(run.wait() for run in runs):
        raise SystemExit(f"a shard failed; see {out}/shard*.log and re-run to resume")
    for i in range(shards):
        status = json.loads((out / f"shard{i}/status.json").read_text())
        if status["event"] != "complete" or a.limit:
            raise SystemExit(f"shard {i} stopped at {status['completed']} requests; re-run without --limit to finish")

    with (out / "results.jsonl").open("wb") as merged:
        for i in range(shards):
            with (out / f"shard{i}/results.jsonl").open("rb") as part:
                shutil.copyfileobj(part, merged)
    for i in range(shards):
        name = "environment.json" if i == 0 else f"environment-shard{i}.json"
        shutil.copyfile(out / f"shard{i}/environment.json", out / name)
    subprocess.run([
        sys.executable, "-m", "decision_index", "score", "--edition", a.edition,
        "--suite-dir", a.suite_dir, "--results", str(out / "results.jsonl"), "--engine", a.name or out.name,
    ], check=True)


if __name__ == "__main__":
    main()
