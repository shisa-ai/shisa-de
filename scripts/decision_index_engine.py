"""Decision Index engine using the unchanged Shisa DE-2 client contract.

Requires the separately installed, pinned Decision Index reproduction kit.
Run with ``--engine scripts.decision_index_engine:ShisaDE2Engine``. Use the
upstream runner's ``--compact`` flag to keep benchmark text out of results.
``--option policy=repeat`` reads twice and never thinks, ``policy=direct`` reads
once; the default is the adaptive ``repeat-think``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
from decision_index.engines import Engine, Unsupported
from shisa_de import Choice, DecisionModel, Noul
from shisa_de.readout import MAX_CODES, READOUT_VERSIONS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ShisaDE2Engine(Engine):
    name = "shisa-de2-client"
    latency = (
        "Sequential suite request wall time including client rendering, all question "
        "reads and conditional thoughts, and local HTTP; excludes loading and warmup. "
        "This is not the maintainers' independently measured leaderboard latency."
    )

    def __init__(self, *, model, tokenizer, base_url, max_tokens=32768,
                 serving_manifest, policy="repeat-think", **options):
        if options:
            raise ValueError(f"Unknown engine options: {sorted(options)}")
        super().__init__()
        if not base_url.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("This benchmark engine requires an explicit local endpoint")
        if policy not in ("direct", "repeat", "repeat-think"):
            raise ValueError("policy must be 'direct', 'repeat' or 'repeat-think'")
        self.repeat = 1 if policy == "direct" else 2
        thinks = policy == "repeat-think"
        self.max_tokens = int(max_tokens)
        if self.max_tokens < 2:
            raise ValueError("max_tokens must be at least 2")
        self.model = DecisionModel(
            base_url=base_url, model=model, tokenizer=tokenizer,
            local_files_only=True, family="de2", policy=policy,
            think_gate=0.7, think_budget=1024, max_workers=1, timeout=600,
        )
        root = Path(__file__).resolve().parents[1]
        self.provenance = {
            "model": model, "tokenizer": tokenizer,
            "readout_version": READOUT_VERSIONS["de2"],
            "policy": policy, "think_gate": 0.7 if thinks else None,
            "think_budget": 1024 if thinks else None,
            "think_option_cap": 26 if thinks else None, "calibration": "none", "max_tokens": self.max_tokens,
            "max_options": MAX_CODES, "max_workers": 1,
            "context_overflow": "unsupported; no truncation or single-read fallback",
            "noul_options": "Yes/No; optional criteria descriptions are not rendered",
            "serving": json.loads(Path(serving_manifest).read_text()),
            "source_sha256": {
                str(p.relative_to(root)): sha256(p)
                for p in [Path(__file__), *sorted((root / "shisa_de").glob("*.py")),
                          *sorted((root / "shisa_de/data").glob("*.json"))]
            },
        }

    def __call__(self, state, questions):
        typed = {}
        for key, q in questions.items():
            if q["type"] == "choice":
                if not 2 <= len(q["criteria"]) <= MAX_CODES:
                    raise Unsupported(f"Choice capacity is 2..{MAX_CODES} options")
                typed[key] = Choice(q["instructions"], q["criteria"])
            elif q["type"] == "noul":
                typed[key] = Noul(q["instructions"], q.get("criteria"))
            else:
                raise Unsupported(f"Unsupported question type: {q['type']}")
        # Validate every prompt as it will be sent before any request. Never
        # shorten the state, option list, or number of repetitions to fit.
        readout = self.model.readout
        tokenizer = readout.ensure_tokenizer()
        for question in typed.values():
            prompt = readout.render(state, question, repeat=self.repeat, max_options=MAX_CODES)
            count = len(tokenizer.encode(prompt, add_special_tokens=False))
            if count + 1 > self.max_tokens:
                raise Unsupported(f"Context window: {count} prompt tokens plus answer exceeds {self.max_tokens}")
        try:
            decision = self.model.decide(state, typed, calibrated=False)
        except httpx.HTTPStatusError as exc:
            # Only a server-declared context limit is a capacity refusal. Other
            # HTTP errors remain errors and are retried by the upstream runner.
            if exc.response.status_code in (400, 413, 422) and any(
                text in exc.response.text.lower()
                for text in ("maximum context length", "max_model_len", "maximum model length")
            ):
                raise Unsupported("Context window exceeded during the conditional thought/read") from exc
            raise
        return decision.to_wire(), {"meta": decision.meta, "usage": decision.usage}

    def runtime(self):
        return {"endpoint": self.model.base_url, "client_only": True}

    def close(self):
        self.model.close()
