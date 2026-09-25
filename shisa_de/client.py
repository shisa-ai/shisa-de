"""The friendly layer: labels in, answers out.

Two entry points, both on `DecisionModel`:

- `classify(state, {"intent": [...]})` mirrors the label-set style of encoder
  classifiers: name a head, pass the labels, get a dict back.
- `decide(state, {"route": Choice(...), "urgent": Noul(...)})` is the full
  System One call, with the three question types and their typed answers.

Both send one question per request, because that is DE-1's contract. Heads and
labels run concurrently over a small thread pool, and every answer records what
it cost.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import httpx

from .calibration import Calibration, confidence as wire_confidence, load_calibration, temper_binary, temper_distribution
from .questions import Choice, Noul, Question, QuestionError, Score
from .readout import LETTERS, READOUT_VERSION, LetterRead, Readout, ReadoutError

DEFAULT_ENDPOINT = "https://api.shisa.ai/openai"
DEFAULT_MODEL = "shisa-ai/shisa-de-1"
ENDPOINT_ENV = "SHISA_DE_ENDPOINT"
API_KEY_ENVS = ("SHISA_DE_API_KEY", "SHISA_API_KEY")


@dataclass
class Answer:
    """One typed answer, with the detail behind it."""

    type: str
    calibrated: bool = False
    temperature: float = 1.0
    requests: int = 1
    missing_from_top: list[str] = field(default_factory=list)
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    choice: str | None = None
    noul: float | None = None
    score: float | None = None
    level: str | None = None
    legend: dict[str, str] | None = None

    @property
    def value(self) -> Any:
        """The primary value: the option, the probability of yes, or the position."""
        if self.type == "noul":
            return self.noul
        if self.type == "score":
            return self.score
        return self.choice

    @property
    def label(self) -> Any:
        """The answer as a label: the option key, or the level the answer sits on.

        `classify` returns this, so an ordered scale reads as one of its levels
        rather than as an index or a weighted position.
        """
        if self.type == "score" and self.level is not None:
            return self.level
        return self.value

    def to_wire(self) -> dict[str, Any]:
        """The TypeSafe answer shape."""
        if self.type == "noul":
            return {"type": "noul", "noul": self.noul}
        if self.type == "score":
            return {
                "type": "score",
                "score": self.score,
                "legend": self.legend,
                "probabilities": self.probabilities,
                "confidence": self.confidence,
            }
        return {
            "type": "choice",
            "choice": self.choice,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
        }

    def to_dict(self) -> dict[str, Any]:
        """Everything, including the fields TypeSafe does not return."""
        out: dict[str, Any] = {"type": self.type, "value": self.value, "confidence": self.confidence,
                               "probabilities": self.probabilities, "calibrated": self.calibrated,
                               "temperature": self.temperature, "requests": self.requests}
        if self.missing_from_top:
            out["missing_from_top"] = self.missing_from_top
        return out


class Decision(dict):
    """A dict of head to value, with the typed answers and provenance beside it.

    ``result["route"]`` is the quick path; ``result.answers["route"]`` is the
    same answer with its distribution, confidence, and request accounting.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.answers: dict[str, Answer] = {}
        self.raw: dict[str, dict[str, Any]] = {}
        self.usage: dict[str, Any] = {}
        self.meta: dict[str, Any] = {}

    # -- TypeSafe-style accessors -----------------------------------------

    @property
    def choices(self) -> dict[str, Answer]:
        return {key: answer for key, answer in self.answers.items() if answer.type == "choice"}

    @property
    def nouls(self) -> dict[str, Answer]:
        return {key: answer for key, answer in self.answers.items() if answer.type == "noul"}

    @property
    def scores(self) -> dict[str, Answer]:
        return {key: answer for key, answer in self.answers.items() if answer.type == "score"}

    def to_wire(self) -> dict[str, Any]:
        """The TypeSafe response body, for callers that expect that shape."""
        return {
            "model": self.meta.get("model"),
            "answers": {key: answer.to_wire() for key, answer in self.answers.items()},
            "usage": {
                "input_tokens": self.usage.get("input_tokens", 0),
                "output_tokens": self.usage.get("output_tokens", 0),
            },
        }


def _question_from_head(head: str, spec: Any) -> tuple[Question, dict[str, Any]]:
    """Turn a head's label spec into a question, plus options for the answer."""
    options: dict[str, Any] = {}

    if isinstance(spec, Question):
        return spec, options

    if isinstance(spec, (list, tuple, set)):
        labels = list(spec)
        options = {"labels": labels}
        return Choice(instructions=_default_question(head), criteria={label: None for label in labels}), options

    if isinstance(spec, str):
        raise QuestionError(
            f"head {head!r}: a bare string is not a label set; pass a list of labels or a question object"
        )

    if not isinstance(spec, Mapping):
        raise QuestionError(f"head {head!r}: expected a label list, a mapping, or a question object")

    spec = dict(spec)
    known_keys = {"type", "labels", "levels", "prompt", "instructions", "multi_label", "cls_threshold", "threshold"}
    if not known_keys & set(spec):
        # A bare mapping of label to description is read as the label set itself.
        if spec and all(isinstance(value, (str, type(None))) for value in spec.values()):
            options = {"labels": list(spec)}
            return Choice(instructions=_default_question(head), criteria=dict(spec)), options

    labels = spec.get("labels")
    levels = spec.get("levels")
    prompt = spec.get("prompt") or spec.get("instructions")
    multi_label = bool(spec.get("multi_label"))
    threshold = spec.get("cls_threshold", spec.get("threshold"))

    if levels is not None:
        if not isinstance(levels, Sequence) or isinstance(levels, str):
            raise QuestionError(f"head {head!r}: levels must be a list of level descriptions")
        options = {"levels": list(levels)}
        return Score(instructions=prompt or _default_question(head), criteria=list(levels)), options

    if labels is None:
        raise QuestionError(f"head {head!r}: expected 'labels' or 'levels'")

    if isinstance(labels, Mapping):
        described = dict(labels)
        options = {"labels": list(described)}
        return Choice(instructions=prompt or _default_question(head), criteria=described), options

    if isinstance(labels, (list, tuple, set)):
        label_list = list(labels)
        options = {"labels": label_list, "multi_label": multi_label}
        if multi_label:
            template = prompt or _multi_label_question(head)
            options["cls_threshold"] = 0.5 if threshold is None else float(threshold)
            return _MultiLabel(instructions=template, labels=label_list), options
        return Choice(instructions=prompt or _default_question(head), criteria={label: None for label in label_list}), options

    raise QuestionError(f"head {head!r}: labels must be a list or a mapping of label to description")


class _MultiLabel(Question):
    """A label set where several labels may apply: one yes/no question per label."""

    def __init__(self, instructions: str, labels: Sequence[str]) -> None:
        self.template = instructions
        self.labels = list(labels)
        self.instructions = instructions
        self.type = "multi_label"

    def to_wire(self) -> dict[str, Any]:
        raise QuestionError("a multi-label head is sent as one noul question per label")

    def options(self) -> list[tuple[str, Any]]:
        raise QuestionError("a multi-label head has no single option list")

    def noul_for(self, label: str) -> Noul:
        if "{label}" in self.template:
            instructions = self.template.format(label=label)
        else:
            instructions = f"{self.template} {label}"
        return Noul(instructions=instructions)


def _default_question(head: str) -> str:
    return f"What is the {head.replace('_', ' ')}?"


def _multi_label_question(head: str) -> str:
    return "Is {label} one of the " + head.replace("_", " ") + "?"


class DecisionModel:
    """A client for a served DE-1 decision model.

    The model is served, never loaded: this class holds an HTTP client and a
    tokenizer, and the weights stay on the GPU server.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        tokenizer: str | None = None,
        tokenizer_revision: str | None = None,
        local_files_only: bool = False,
        timeout: float = 120.0,
        max_workers: int = 8,
        calibration: Calibration | None = None,
        transport: httpx.BaseTransport | None = None,
        readout: Readout | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get(ENDPOINT_ENV) or DEFAULT_ENDPOINT).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.calibration = calibration if calibration is not None else load_calibration()
        self.max_workers = max(1, int(max_workers))
        if readout is not None:
            self.readout = readout
        else:
            self.readout = Readout(
                base_url=self.base_url,
                model=self.model,
                tokenizer=tokenizer,
                tokenizer_revision=tokenizer_revision,
                local_files_only=local_files_only,
                timeout=timeout,
                api_key=api_key or _api_key_from_env(),
                transport=transport,
            )

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_endpoint(cls, base_url: str, model: str = DEFAULT_MODEL, **kwargs: Any) -> "DecisionModel":
        """Point at a specific endpoint, such as a local vLLM on port 8021."""
        return cls(base_url=base_url, model=model, **kwargs)

    @classmethod
    def from_pretrained(cls, model: str = DEFAULT_MODEL, base_url: str | None = None, **kwargs: Any) -> "DecisionModel":
        """Pick a checkpoint by name and talk to it where it is served.

        No weights are downloaded. The endpoint comes from `base_url`, then
        `SHISA_DE_ENDPOINT`, then the hosted default, and the key comes from
        `SHISA_DE_API_KEY` or `SHISA_API_KEY`.
        """
        return cls(base_url=base_url, model=model, **kwargs)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self.readout.close()

    def __enter__(self) -> "DecisionModel":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def health(self) -> dict[str, Any]:
        """Check the endpoint, the served model id, and the answer boundary."""
        report: dict[str, Any] = {
            "base_url": self.base_url,
            "model": self.model,
            "readout_version": READOUT_VERSION,
            "calibration": self.calibration.id,
            "ok": False,
        }
        try:
            ids = self.readout.list_models()
            report["models_status"] = 200
            report["served_models"] = ids
            report["model_listed"] = self.model in ids
        except (httpx.HTTPError, ReadoutError) as exc:
            report["models_status"] = f"error: {exc}"
        try:
            slots = self.readout.slots(2)
            prompt = self.readout.render({"health": "check"}, Noul(instructions="Is this a health check?"))
            self.readout.check_boundary(prompt, slots)
            report["boundary_check"] = "passed"
            report["slots"] = {slot.letter: slot.token_id for slot in slots}
            report["ok"] = True
        except (ReadoutError, QuestionError) as exc:
            report["boundary_check"] = f"failed: {exc}"
        return report

    # -- the friendly call -------------------------------------------------

    def classify(
        self,
        state: Any,
        labels: Mapping[str, Any],
        *,
        include_confidence: bool = False,
        include_probabilities: bool = False,
        calibrated: bool = True,
        debug: bool = False,
    ) -> Decision:
        """Classify a state against named label sets.

        ``labels`` maps a head name to its label set: a list of labels, a
        mapping of label to description, ``{"levels": [...]}`` for an ordered
        scale, ``{"labels": [...], "multi_label": True}`` for several labels at
        once, or a question object for full control.

        Returns a `Decision`: ``result["intent"]`` is the chosen label, and
        ``result.answers["intent"]`` carries the distribution behind it.
        """
        questions: dict[str, Any] = {}
        parsed: dict[str, Any] = {}
        for head, spec in labels.items():
            question, options = _question_from_head(head, spec)
            questions[head] = question
            parsed[head] = options
        decision = self._run(state, questions, calibrated=calibrated, debug=debug)
        for head, question in questions.items():
            if isinstance(question, _MultiLabel):
                threshold = parsed[head]["cls_threshold"]
                entries = []
                for label in question.labels:
                    answer = decision.answers.pop(f"{head}:{label}")
                    entry: dict[str, Any] = {"label": label, "confidence": answer.confidence,
                                             "probabilities": answer.probabilities}
                    if answer.noul is not None and answer.noul >= threshold:
                        entries.append(entry)
                    decision.raw[f"{head}:{label}"] = {"noul": answer.noul, "requests": answer.requests}
                if include_probabilities:
                    decision[head] = entries
                elif include_confidence:
                    decision[head] = [{"label": entry["label"], "confidence": entry["confidence"]} for entry in entries]
                else:
                    decision[head] = [entry["label"] for entry in entries]
                continue
            answer = decision.answers[head]
            if include_probabilities:
                decision[head] = {"label": answer.label, "confidence": answer.confidence,
                                  "probabilities": answer.probabilities}
            elif include_confidence:
                decision[head] = {"label": answer.label, "confidence": answer.confidence}
            else:
                decision[head] = answer.label
        return decision

    # -- the full call -----------------------------------------------------

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        calibrated: bool = True,
        debug: bool = False,
    ) -> Decision:
        """Ask typed questions and get typed answers.

        ``questions`` maps a question id to a `Noul`, `Choice`, or `Score`, or
        to the label-set shorthand `classify` accepts. The dict view returns
        each answer's primary value; ``result.answers`` returns the full answer.
        """
        parsed = {head: _question_from_head(head, spec)[0] for head, spec in questions.items()}
        decision = self._run(state, parsed, calibrated=calibrated, debug=debug)
        for head, answer in decision.answers.items():
            decision[head] = answer.value
        return decision

    system_one = decide

    # -- the machinery -----------------------------------------------------

    def _run(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        calibrated: bool,
        debug: bool,
    ) -> Decision:
        started = time.perf_counter()
        decision = Decision()
        work: list[tuple[str, Question]] = []
        for head, question in questions.items():
            if isinstance(question, _MultiLabel):
                work.extend((f"{head}:{label}", question.noul_for(label)) for label in question.labels)
            else:
                work.append((head, question))

        def run(item: tuple[str, Question]) -> tuple[str, Question, LetterRead, list[tuple[str, Any]]]:
            head, question = item
            read, options = self.readout.evaluate(state, question, debug=debug)
            return head, question, read, options

        if self.max_workers > 1 and len(work) > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(work))) as pool:
                results = list(pool.map(run, work))
        else:
            results = [run(item) for item in work]

        input_tokens = 0
        requests = 0
        for head, question, read, options in results:
            answer = _answer_from_read(question, read, options, self.calibration, calibrated)
            decision.answers[head] = answer
            decision.raw[head] = {
                "logprobs": read.logprobs,
                "probabilities": read.probabilities,
                "requests": read.requests,
                "missing_from_top": read.missing_from_top,
                "ranks": read.ranks,
                "prompt_tokens": read.prompt_tokens,
                "sampled": read.sampled,
                "options": [key for key, _ in options],
            }
            if debug:
                decision.raw[head]["prompt"] = read.prompt
                decision.raw[head]["top_logprobs"] = read.top_logprobs
            input_tokens += read.prompt_tokens
            requests += read.requests
        decision.usage = {
            "input_tokens": input_tokens,
            "output_tokens": len(work),
            "requests": requests,
            "wall_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        decision.meta = {
            "model": self.model,
            "base_url": self.base_url,
            "readout_version": READOUT_VERSION,
            "calibration": self.calibration.id,
            "calibrated": calibrated,
        }
        return decision


def _answer_from_read(
    question: Question,
    read: LetterRead,
    options: Sequence[tuple[str, Any]],
    calibration: Calibration,
    calibrated: bool,
) -> Answer:
    """Turn one letter read into a typed answer, tempering if asked."""
    temperature = calibration.temperature_for(question.type) if calibrated else 1.0
    applied = calibrated and temperature != 1.0
    letters = [LETTERS[index] for index in range(len(options))]
    raw = {letter: read.probabilities[letter] for letter in letters}

    common: dict[str, Any] = {
        "calibrated": applied,
        "temperature": temperature,
        "requests": read.requests,
        "missing_from_top": list(read.missing_from_top),
    }

    if question.type == "noul":
        yes = temper_binary(raw["A"], temperature) if applied else raw["A"]
        probabilities = {"yes": yes, "no": 1.0 - yes}
        return Answer(type="noul", noul=yes, probabilities=probabilities,
                      confidence=wire_confidence(probabilities), **common)

    probs = temper_distribution(raw, temperature) if applied else dict(raw)
    best = max(probs, key=probs.get)
    index = letters.index(best)

    if question.type == "score":
        weights = [probs[letter] for letter in letters]
        return Answer(
            type="score",
            score=float(sum(position * weight for position, weight in enumerate(weights))),
            level=str(options[index][1]),
            choice=options[index][0],
            legend={str(position): str(level) for position, (_, level) in enumerate(options)},
            probabilities={str(position): probs[letter] for position, letter in enumerate(letters)},
            confidence=wire_confidence(probs),
            **common,
        )

    return Answer(
        type="choice",
        choice=options[index][0],
        probabilities={options[letters.index(letter)][0]: probs[letter] for letter in letters},
        confidence=wire_confidence(probs),
        **common,
    )


def _api_key_from_env() -> str | None:
    for name in API_KEY_ENVS:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


__all__ = ["Answer", "Decision", "DecisionModel", "DEFAULT_ENDPOINT", "DEFAULT_MODEL"]
