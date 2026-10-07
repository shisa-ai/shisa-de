"""The friendly layer: labels in, answers out.

Two entry points, both on `DecisionModel`:

- `classify(state, {"intent": [...]})` mirrors the label-set style of encoder
  classifiers: name a head, pass the labels, get a dict back.
- `decide(state, {"route": Choice(...), "urgent": Noul(...)})` is the full
  System One call, with the three question types and their typed answers.

Questions run concurrently over a bounded head pool. The served model's family
picks the contract: DE-1 answers from one read, and its text choices above 26
options use balanced chunks and a final choice among chunk winners; DE-2 reads
the question twice, thinks when that read is unsure, and reads text choices up
to 256 options in one prompt. Every answer records its strategy and cost.
"""

from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

from .calibration import (
    Calibration,
    calibration_for,
    confidence as wire_confidence,
    temper_binary,
    temper_distribution,
)
from .family import resolve_family
from .images import prepare_image
from .questions import MAX_OPTIONS, Choice, Noul, Question, QuestionError, Score
from .overflow import OVERFLOW_STRATEGY, OverflowRead, read_overflow, validate_overflow
from .policy import THINK_OPTION_CAP, PolicyRead, ReadOptions, read_policy, resolve_options
from .readout import MAX_CODES, READOUT_VERSIONS, LetterRead, Readout, ReadoutError, codes_for

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
    confidence: float | None = 0.0
    strategy: str = "direct"
    score_semantics: str = "option-softmax"
    stages: int = 1
    logical_reads: int = 1
    finalists: list[str] | None = None
    thought_tokens: int = 0
    thought_closed: bool | None = None
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
        out = {
            "type": "choice",
            "choice": self.choice,
            "probabilities": self.probabilities,
        }
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.score_semantics != "option-softmax":
            out.update(strategy=self.strategy, score_semantics=self.score_semantics,
                       calibrated=False)
        return out

    def to_dict(self) -> dict[str, Any]:
        """Everything, including the fields TypeSafe does not return."""
        out: dict[str, Any] = {"type": self.type, "value": self.value, "confidence": self.confidence,
                               "probabilities": self.probabilities, "calibrated": self.calibrated,
                               "temperature": self.temperature, "requests": self.requests}
        if self.missing_from_top:
            out["missing_from_top"] = self.missing_from_top
        out.update(strategy=self.strategy, score_semantics=self.score_semantics,
                   stages=self.stages, logical_reads=self.logical_reads)
        if self.finalists is not None:
            out["finalists"] = list(self.finalists)
        if self.thought_closed is not None:
            out.update(thought_tokens=self.thought_tokens, thought_closed=self.thought_closed)
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
    """A client for a served DE-1 or DE-2 decision model.

    The model is served, never loaded: this class holds an HTTP client and a
    tokenizer, and the weights stay on the GPU server.

    ``family`` declares which contract the served checkpoint is read through
    (``"de1"`` or ``"de2"``). Left unset it is taken from the model id, then the
    tokenizer source, and otherwise assumed to be DE-2 with a warning.
    ``policy`` selects the DE-2 read: ``"repeat-think"`` (the default),
    ``"repeat"`` or ``"direct"``; DE-1 is always ``"direct"``.

    The same read can be set a part at a time, here as the model's default or
    on one `classify` / `decide` call:

    - ``reads``: ``"single"`` or ``"double"``, how often the user turn is written.
    - ``reasoning``: think when the read is unsure, then read again.
    - ``reasoning_prob``: the top probability below which it thinks (0.7).
      ``think_gate`` is the same setting.
    - ``reasoning_len``: the most tokens a thought may run to (1,024).
      ``think_budget`` is the same setting.
    - ``compound``: read a DE-1 text choice above 26 options in two rounds.
      ``overflow="finalist-top1"`` / ``"error"`` is the same setting.

    DE-2 defaults to a double read with reasoning; DE-1 to a single read with
    compound on. A combination a family's contract does not define (a double
    read or reasoning on DE-1, compound on DE-2, reasoning after a single read)
    raises `ValueError`.
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
        image_top_logprobs: int = 20,
        max_logprobs: int = 20,
        overflow: str | None = None,
        calibration: Calibration | None = None,
        family: str | None = None,
        policy: str | None = None,
        think_gate: float | None = None,
        think_budget: int | None = None,
        reads: str | None = None,
        reasoning: bool | None = None,
        reasoning_prob: float | None = None,
        reasoning_len: int | None = None,
        compound: bool | None = None,
        transport: httpx.BaseTransport | None = None,
        readout: Readout | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get(ENDPOINT_ENV) or DEFAULT_ENDPOINT).rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model = model or DEFAULT_MODEL
        # The family picks the contract: the readout version, the policy, the
        # option limit and the calibration record. It is resolved once, here, and
        # its source is kept so an assumption is never presented as a detection.
        tokenizer_source = tokenizer or getattr(readout, "tokenizer_source", None)
        self.family, self.family_source = resolve_family(self.model, tokenizer_source, family)
        self.family_explicit = self.family_source != "assumed"
        if not self.family_explicit:
            warnings.warn(
                f"{self.model!r} names neither DE-1 nor DE-2, so it is read as {self.family!r}; "
                "pass family='de1' or family='de2' to say which contract it was trained for",
                stacklevel=2,
            )
        self.readout_version = READOUT_VERSIONS[self.family]
        # The model's defaults. A call may override any of them for itself.
        self.options = resolve_options(
            self.family, policy=policy or None, reads=reads, reasoning=reasoning,
            reasoning_prob=reasoning_prob, reasoning_len=reasoning_len, compound=compound,
            think_gate=think_gate, think_budget=think_budget, overflow=overflow,
        )
        # A record chosen for the caller is applied only to the checkpoint it was
        # fitted on. A record the caller passes is their assertion and is applied
        # as given; `health` still reports how far it matches.
        supplied = calibration is not None
        self.calibration = calibration if supplied else calibration_for(self.family)
        self.calibration_level, self.calibration_reasons = self.calibration.applicability(
            self.model, self.readout_version, family=self.family, tokenizer=tokenizer_source)
        self.calibration_applied = self.calibration.fitted and (
            supplied or self.calibration_level == "checkpoint")
        if self.calibration.fitted and not self.calibration_applied:
            warnings.warn(
                f"the bundled {self.family} calibration is not applied to {self.model!r}: "
                f"{'; '.join(self.calibration_reasons)}. Answers are raw probabilities; pass "
                "calibration= to apply a record you vouch for",
                stacklevel=2,
            )
        self.max_workers = max(1, int(max_workers))
        if isinstance(max_logprobs, bool) or not isinstance(max_logprobs, int) or max_logprobs < 1:
            raise ValueError("max_logprobs must be a positive integer")
        if isinstance(image_top_logprobs, bool) or not isinstance(image_top_logprobs, int) or image_top_logprobs < 1:
            raise ValueError("image_top_logprobs must be a positive integer")
        self.image_top_logprobs = image_top_logprobs
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
                max_logprobs=max_logprobs,
                api_key=api_key if api_key is not None else _api_key_from_env(),
                transport=transport,
            )

    # -- the model's default read ------------------------------------------

    @property
    def policy(self) -> str:
        return self.options.policy

    @property
    def reads(self) -> str:
        return self.options.reads

    @property
    def reasoning(self) -> bool:
        return self.options.reasoning

    @property
    def think_gate(self) -> float:
        return self.options.think_gate

    @property
    def think_budget(self) -> int:
        return self.options.think_budget

    reasoning_prob = think_gate
    reasoning_len = think_budget

    @property
    def overflow(self) -> str:
        return self.options.overflow

    @property
    def compound(self) -> bool:
        return self.options.compound

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

    def health(self, probe: bool = False) -> dict[str, Any]:
        """Check the endpoint, the served model id, the answer boundary, and the calibration.

        A passing boundary check says the answer codes were read from the right
        position. It says nothing about whether the temperatures applied to them
        belong to this checkpoint, which is a separate failure and a silent one,
        so the calibration's provenance is checked here too and a mismatch fails
        the report.

        ``probe=True`` also spends requests: one question through the family's
        own read, which fails the report if the server cannot serve it, and one
        text-only chat request that reports how the server renders the system
        turn, which is what the image path depends on.
        """
        report: dict[str, Any] = {
            "base_url": self.base_url,
            "model": self.model,
            "model_family": self.family,
            "family_source": self.family_source,
            "family_explicit": self.family_explicit,
            "readout_version": self.readout_version,
            "policy": self.policy,
            "reads": self.reads,
            "reasoning": self.reasoning,
            "compound": self.compound,
            "calibration": self.calibration.id,
            "calibration_model": self.calibration.model,
            "calibration_readout_version": self.calibration.readout_version,
            "calibration_serving_shape": self.calibration.serving_shape,
            "calibration_level": self.calibration_level,
            "calibration_applied": self.calibration_applied,
            "ok": False,
        }
        if not self.family_explicit:
            report["family_note"] = (
                f"{self.model!r} carries no DE-1 or DE-2 slug; {self.family!r} was assumed, "
                "not detected"
            )
        matched = self.calibration_level != "mismatch"
        report["calibration_match"] = matched
        if not matched:
            report["calibration_mismatch"] = list(self.calibration_reasons)
        elif self.calibration_reasons:
            report["calibration_note"] = list(self.calibration_reasons)
        try:
            ids = self.readout.list_models()
            report["models_status"] = 200
            report["served_models"] = ids
            report["model_listed"] = self.model in ids
        except (httpx.HTTPError, ReadoutError) as exc:
            report["models_status"] = f"error: {exc}"
        question = Noul(instructions="Is this a health check?")
        try:
            if self.family == "de2":
                # Every code the contract can use, not only the two a noul needs.
                slots = self.readout.code_slots(MAX_CODES)
                prompt = self.readout.render({"health": "check"}, question,
                                             repeat=1 if self.policy == "direct" else 2)
                tokenizer = self.readout.ensure_tokenizer()
                self.readout.check_code_boundary(tokenizer.encode(prompt, add_special_tokens=False), slots)
                report["slots"] = {"count": len(slots), **{slot.letter: slot.token_id for slot in slots[:2]}}
            else:
                slots = self.readout.slots(2)
                prompt = self.readout.render({"health": "check"}, question)
                self.readout.check_boundary(prompt, slots)
                report["slots"] = {slot.letter: slot.token_id for slot in slots}
            report["boundary_check"] = "passed"
            report["ok"] = bool(report.get("model_listed", False)) and matched
        except (ReadoutError, QuestionError) as exc:
            report["boundary_check"] = f"failed: {exc}"
        if probe:
            try:
                result = self.decide({"health": "check"}, {"probe": question}, calibrated=False)
                answer = result.answers["probe"]
                report["read_probe"] = f"passed ({answer.strategy}, {answer.requests} requests)"
            except (ReadoutError, QuestionError) as exc:
                report["read_probe"] = f"failed: {exc}"
                report["ok"] = False
            report["chat_system_render"] = self.readout.probe_chat_render({"health": "check"}, question)
        return report

    # -- the friendly call -------------------------------------------------

    def classify(
        self,
        state: Any,
        labels: Mapping[str, Any],
        *,
        include_confidence: bool = False,
        include_probabilities: bool = False,
        probability: bool = False,
        image: str | Path | None = None,
        calibrated: bool | None = None,
        debug: bool = False,
        policy: str | None = None,
        reads: str | None = None,
        reasoning: bool | None = None,
        reasoning_prob: float | None = None,
        reasoning_len: int | None = None,
        compound: bool | None = None,
    ) -> Decision:
        """Classify a state against named label sets.

        ``labels`` maps a head name to its label set: a list of labels, a
        mapping of label to description, ``{"levels": [...]}`` for an ordered
        scale, ``{"labels": [...], "multi_label": True}`` for several labels at
        once, or a question object for full control.

        ``image`` attaches a local file, HTTP(S) URL, or image data URL.
        Calibration defaults to enabled for direct text and disabled for images,
        and is applied only where a fitted record covers the served checkpoint.
        DE-1 wide text choices use uncalibrated finalist scores; explicit
        ``calibrated=True`` is rejected for those heads. DE-2 reads text choices
        up to 256 options in one prompt.

        ``probability=True`` requires one logical read per question (per label
        for multi-label heads). On DE-1 it rejects choice overflow; on DE-2 it
        skips the thinking read. It does not change calibration or the return
        shape; ``include_probabilities`` controls the dict view only.

        ``policy``, ``reads``, ``reasoning``, ``reasoning_prob``,
        ``reasoning_len`` and ``compound`` override the model's defaults for
        this call only; see `DecisionModel`.

        Returns a `Decision`: ``result["intent"]`` is the chosen label, and
        ``result.answers["intent"]`` carries the distribution behind it.
        """
        questions: dict[str, Any] = {}
        parsed: dict[str, Any] = {}
        for head, spec in labels.items():
            question, options = _question_from_head(head, spec)
            questions[head] = question
            parsed[head] = options
        settings = resolve_options(self.family, self.options, policy=policy, reads=reads, reasoning=reasoning,
                                   reasoning_prob=reasoning_prob, reasoning_len=reasoning_len, compound=compound)
        decision = self._run(state, questions, image=image, calibrated=calibrated, probability=probability,
                             debug=debug, settings=settings, asked_reasoning=bool(reasoning))
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
            if answer.score_semantics != "option-softmax" and (include_probabilities or include_confidence):
                decision[head].update(strategy=answer.strategy, score_semantics=answer.score_semantics,
                                      calibrated=False)
        return decision

    # -- the full call -----------------------------------------------------

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        probability: bool = False,
        image: str | Path | None = None,
        calibrated: bool | None = None,
        debug: bool = False,
        policy: str | None = None,
        reads: str | None = None,
        reasoning: bool | None = None,
        reasoning_prob: float | None = None,
        reasoning_len: int | None = None,
        compound: bool | None = None,
    ) -> Decision:
        """Ask typed questions and get typed answers.

        ``questions`` maps a question id to a `Noul`, `Choice`, or `Score`, or
        to the label-set shorthand `classify` accepts. The dict view returns
        each answer's primary value; ``result.answers`` returns the full answer.
        ``image`` attaches a local file, HTTP(S) URL, or image data URL.
        Calibration defaults to enabled for direct text and disabled for images.
        DE-1 wide text choices return uncalibrated finalist scores and reject
        explicit ``calibrated=True``.
        ``probability=True`` requires one logical read per question: DE-1
        rejects choice overflow, and DE-2 skips the thinking read. It leaves
        calibration and the return shape unchanged. Letter-recovery requests
        are still allowed for the same answer position.

        ``policy``, ``reads``, ``reasoning``, ``reasoning_prob``,
        ``reasoning_len`` and ``compound`` override the model's defaults for
        this call only; see `DecisionModel`.
        """
        parsed = {head: _question_from_head(head, spec)[0] for head, spec in questions.items()}
        settings = resolve_options(self.family, self.options, policy=policy, reads=reads, reasoning=reasoning,
                                   reasoning_prob=reasoning_prob, reasoning_len=reasoning_len, compound=compound)
        decision = self._run(state, parsed, image=image, calibrated=calibrated, probability=probability,
                             debug=debug, settings=settings, asked_reasoning=bool(reasoning))
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
        image: str | Path | None,
        calibrated: bool | None,
        probability: bool,
        debug: bool,
        settings: ReadOptions,
        asked_reasoning: bool = False,
    ) -> Decision:
        if not isinstance(probability, bool):
            raise ValueError("probability must be a boolean")
        if probability and asked_reasoning:
            raise ValueError("probability=True takes one logical read per question; it cannot be "
                             "combined with reasoning=True")
        started = time.perf_counter()
        image_url = prepare_image(image) if image is not None else None
        requested_calibration = calibrated
        if calibrated is None:
            calibrated = image_url is None
        decision = Decision()
        work: list[tuple[str, Question]] = []
        for head, question in questions.items():
            if isinstance(question, _MultiLabel):
                work.extend((f"{head}:{label}", question.noul_for(label)) for label in question.labels)
            else:
                work.append((head, question))

        # A probability call takes one logical read per question, so on DE-2 it
        # stops at the repeated read: an answer read after a thought is a second.
        policy = "repeat" if probability and settings.policy == "repeat-think" else settings.policy

        # Validate every head before issuing any requests.
        overflow_heads = set()
        for head, question in work:
            wide = isinstance(question, Choice) and len(question.options()) > MAX_OPTIONS
            if self.family == "de2":
                # DE-2 reads a text choice of up to MAX_CODES options in one prompt,
                # so nothing overflows; the image path keeps the letter limit.
                if wide and image_url is not None:
                    raise QuestionError(f"head {head!r}: image choices above {MAX_OPTIONS} options are not supported")
                question.validate(MAX_CODES if wide else MAX_OPTIONS)
            elif wide:
                if probability:
                    raise QuestionError(
                        f"head {head!r}: probability=True requires a single read; "
                        f"reduce the choice to at most {MAX_OPTIONS} options"
                    )
                if not settings.compound:
                    raise QuestionError(f"head {head!r}: overflow='error' rejects choices above {MAX_OPTIONS} options")
                if image_url is not None:
                    raise QuestionError("image choice overflow is not supported")
                if requested_calibration is True:
                    raise QuestionError("overflow scores are uncalibrated; omit calibrated or pass calibrated=False")
                validate_overflow(question)
                overflow_heads.add(head)
            else:
                question.validate()

        def run(item: tuple[str, Question]) -> tuple[str, Question, Any, list[tuple[str, Any]]]:
            head, question = item
            if head in overflow_heads:
                return head, question, read_overflow(self.readout, state, question, debug=debug), question.options()
            if image_url is not None:
                read = self.readout.read_image(
                    state, question, image_url, top_logprobs=self.image_top_logprobs, debug=debug,
                )
                options = question.options()
            elif self.family == "de2":
                read = read_policy(self.readout, state, question, policy=policy,
                                   think_gate=settings.think_gate, think_budget=settings.think_budget, debug=debug)
                options = question.options()
            else:
                read, options = self.readout.evaluate(state, question, debug=debug)
            return head, question, read, options

        if self.max_workers > 1 and len(work) > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(work))) as pool:
                results = list(pool.map(run, work))
        else:
            results = [run(item) for item in work]

        input_tokens = 0
        requests = 0
        logical_reads = 0
        thought_tokens = 0
        for head, question, read, options in results:
            if isinstance(read, PolicyRead):
                # Each read is tempered by its own fit. A record without per-read
                # temperatures was fitted on a read before any thought, so an
                # answer read after one stays raw under it.
                covered = bool(self.calibration.reads) or read.thought is None
                answer = _answer_from_read(question, read.read, options, self.calibration,
                                           calibrated and self.calibration_applied and covered,
                                           read_name=read.components[-1][0])
                answer.strategy = read.strategy
                answer.stages = answer.logical_reads = read.logical_reads
                answer.requests = read.requests
                if read.thought is not None:
                    answer.thought_tokens = read.thought_tokens
                    answer.thought_closed = read.thought.closed
                decision.answers[head] = answer
                decision.raw[head] = {**read.raw(debug), "options": [key for key, _ in options]}
                input_tokens += read.prompt_tokens
                requests += read.requests
                logical_reads += read.logical_reads
                thought_tokens += read.thought_tokens
                continue
            if isinstance(read, OverflowRead):
                answer = Answer(type="choice", probabilities=read.probabilities,
                                choice=max(read.probabilities, key=read.probabilities.get),
                                confidence=None, calibrated=False, temperature=1.0,
                                requests=read.requests, missing_from_top=read.missing_from_top,
                                strategy=OVERFLOW_STRATEGY, score_semantics="conditional-on-finalists",
                                stages=2, logical_reads=read.logical_reads, finalists=read.finalists)
                decision.answers[head] = answer
                decision.raw[head] = read.raw(debug)
                input_tokens += read.prompt_tokens
                requests += read.requests
                logical_reads += read.logical_reads
                continue
            logical_reads += 1
            answer = _answer_from_read(question, read, options, self.calibration,
                                       calibrated and self.calibration_applied,
                                       read_name="image" if image_url is not None else "direct")
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
                if read.messages is not None:
                    decision.raw[head]["messages"] = read.messages
            if read.system_render is not None:
                decision.raw[head]["system_render"] = read.system_render
            input_tokens += read.prompt_tokens
            requests += read.requests
        decision.usage = {
            "input_tokens": input_tokens,
            "output_tokens": logical_reads + thought_tokens,
            "logical_reads": logical_reads,
            "thought_tokens": thought_tokens,
            "requests": requests,
            "wall_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        # What the answers carry, not what was asked for: a requested calibration
        # that no fitted record covers leaves every answer raw.
        flags = {answer.calibrated for answer in decision.answers.values()}
        decision.meta = {
            "model": self.model,
            "base_url": self.base_url,
            "family": self.family,
            "readout_version": self.readout_version,
            "policy": policy,
            "probability": probability,
            "input_type": "image" if image_url is not None else "text",
            "calibration": self.calibration.id,
            "calibrated": next(iter(flags)) if len(flags) == 1 else (None if flags else calibrated),
            "calibration_by_head": {head: answer.calibrated for head, answer in decision.answers.items()},
            "strategy_by_head": {head: answer.strategy for head, answer in decision.answers.items()},
        }
        if self.family == "de2":
            decision.meta.update(think_gate=settings.think_gate, think_budget=settings.think_budget,
                                 think_option_cap=THINK_OPTION_CAP)
        return decision


def _answer_from_read(
    question: Question,
    read: LetterRead,
    options: Sequence[tuple[str, Any]],
    calibration: Calibration,
    calibrated: bool,
    read_name: str | None = None,
) -> Answer:
    """Turn one letter read into a typed answer, tempering if asked."""
    temperature = calibration.temperature_for(question.type, read_name) if calibrated else 1.0
    applied = calibrated and temperature != 1.0
    letters = codes_for(len(options))
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
