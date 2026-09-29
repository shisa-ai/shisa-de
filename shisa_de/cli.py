"""Command line: `shisa-de doctor`, `shisa-de ask`, `shisa-de explain`.

`doctor` checks an endpoint before you depend on it. `ask` is the one-line
classification call. `explain` prints every step of the readout for one
question, which is the fastest way to see what the model actually receives.
"""

from __future__ import annotations

import argparse
import json
import shlex
from typing import Any

from .client import DEFAULT_ENDPOINT, DEFAULT_MODEL, DecisionModel, _api_key_from_env
from .readout import LETTERS, READOUT_VERSION
from .questions import Choice


def _state_from(text: str | None) -> Any:
    if text is None:
        return {
            "ticket": "Order 4812 was marked delivered on Monday. The customer says the parcel "
                      "never arrived and tracking has not updated since Friday.",
            "account": "customer since 2021, no prior claims",
        }
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return text
    return text


def _labels_from(spec: str | None) -> dict[str, Any]:
    if not spec:
        return {"intent": ["order_status", "refund_request", "cancel_order", "speak_to_human"]}
    labels = [part.strip() for part in spec.split(",") if part.strip()]
    return {"intent": labels}


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=None, help=f"endpoint (default: {DEFAULT_ENDPOINT})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="served model id")
    parser.add_argument("--tokenizer", default=None, help="tokenizer source (defaults to --model)")
    parser.add_argument("--timeout", type=float, default=120.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shisa-de", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check an endpoint, the served model, and the answer boundary")
    _common(doctor)
    doctor.add_argument("--json", action="store_true", help="print the report as JSON")

    ask = sub.add_parser("ask", help="classify a state against a label set")
    _common(ask)
    ask.add_argument("--state", default=None, help="JSON state, or plain text; a built-in example by default")
    ask.add_argument("--image", help="local image path, HTTP(S) URL, or image data URL")
    ask.add_argument("--image-top-logprobs", type=int, default=20, help="image logprob limit (must be allowed by the server)")
    ask.add_argument("--labels", default=None, help="comma separated labels")
    ask.add_argument("--prompt", default=None, help="the question to ask (default: derived from the head name)")
    ask.add_argument("--confidence", action="store_true", help="return the label with its confidence")
    ask.add_argument("--probabilities", action="store_true", help="return the full distribution")

    explain = sub.add_parser("explain", help="print every step of the readout for one question")
    _common(explain)
    explain.add_argument("--state", default=None, help="JSON state, or plain text; a built-in example by default")
    explain.add_argument("--labels", default=None, help="comma separated labels")
    explain.add_argument("--prompt", default=None, help="the question to ask")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = DecisionModel(
        base_url=args.base_url, model=args.model, tokenizer=args.tokenizer, timeout=args.timeout,
        image_top_logprobs=getattr(args, "image_top_logprobs", 20),
    )
    with model:
        if args.command == "doctor":
            report = model.health()
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print(f"endpoint        {report['base_url']}")
                print(f"model           {report['model']}")
                print(f"readout         {report['readout_version']}")
                print(f"calibration     {report['calibration']}")
                print(f"models endpoint {report.get('models_status')}")
                if report.get("served_models") is not None:
                    listed = "yes" if report.get("model_listed") else "no"
                    print(f"model listed    {listed} (of {len(report['served_models'])} served ids)")
                print(f"boundary check  {report.get('boundary_check')}")
                print(f"ok              {report['ok']}")
            return 0 if report["ok"] else 1

        if args.command == "ask":
            labels = _labels_from(args.labels)
            if args.prompt:
                labels = {head: {"labels": values, "prompt": args.prompt} for head, values in labels.items()}
            result = model.classify(
                {} if args.image and args.state is None else _state_from(args.state),
                labels,
                image=args.image,
                include_confidence=args.confidence,
                include_probabilities=args.probabilities,
            )
            print(json.dumps(dict(result), indent=2, ensure_ascii=False))
            return 0

        if args.command == "explain":
            return _explain(model, args)
    return 2


def _explain(model: DecisionModel, args: argparse.Namespace) -> int:
    state = _state_from(args.state)
    labels = _labels_from(args.labels)
    head, label_list = next(iter(labels.items()))
    question = Choice(
        instructions=args.prompt or f"What is the {head.replace('_', ' ')}?",
        criteria={label: None for label in label_list},
    )
    readout = model.readout
    prompt = readout.render(state, question)
    options = question.options()
    slots = readout.slots(len(options))
    readout.check_boundary(prompt, slots)

    print("1. The prompt the model receives")
    print("-" * 72)
    print(prompt)
    print()
    print("2. The answer slots")
    print("-" * 72)
    for slot in slots:
        print(f"   {slot.letter}  token {slot.token_id:>6}")
    print()
    read = readout.read(prompt, len(options), debug=True)
    print("3. One request: max_tokens 1, temperature 0, logprobs 20")
    print("-" * 72)
    print(f"   POST {model.base_url}/v1/completions")
    print(f"   prompt tokens {read.prompt_tokens}, requests {read.requests}, sampled {read.sampled!r}")
    for token, logprob in sorted(read.top_logprobs.items(), key=lambda kv: -kv[1]):
        marker = "  <- answer slot" if token in [slot.letter for slot in slots] else ""
        print(f"   {token!r:>14}  {logprob:9.4f}{marker}")
    if read.missing_from_top:
        print(f"   fallback requests for letters outside the top 20: {', '.join(read.missing_from_top)}")
    print()
    print("4. The distribution over the option letters")
    print("-" * 72)
    for (key, description), slot in zip(options, slots):
        print(f"   {read.probabilities[slot.letter]:7.4f}  {slot.letter}  {key}  ({description})")
    best = read.answer()
    print(f"   answer: {best} ({read.probabilities[best]:.4f})")
    print()
    print("5. The same request without Python")
    print("-" * 72)
    body = {"model": model.model, "prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": 20}
    print(f"curl -s {shlex.quote(model.base_url + '/v1/completions')} \\")
    print("  -H 'Content-Type: application/json' \\")
    if _api_key_from_env():
        print('  -H "Authorization: Bearer ${SHISA_DE_API_KEY:-$SHISA_API_KEY}" \\')
    print(f"  -d {shlex.quote(json.dumps(body))}")
    print()
    print(f"readout {READOUT_VERSION}; {len(LETTERS)} letters available, {len(options)} used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
