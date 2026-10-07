"""Command line: `shisa-de doctor`, `shisa-de ask`, `shisa-de explain`.

`doctor` checks an endpoint before you depend on it. `ask` is the one-line
classification call. `explain` prints every step of the readout for one
question, which is the fastest way to see what the model actually receives.

All three read DE-1 or DE-2, chosen from `--family`, the model id or the
tokenizer source. For DE-2, `explain` shows the first read of the policy.
"""

from __future__ import annotations

import argparse
import json
import shlex
from typing import Any

from .calibration import CALIBRATION_FILES, resolve_calibration
from .client import DEFAULT_ENDPOINT, DEFAULT_MODEL, DecisionModel, _api_key_from_env
from .family import FAMILIES
from .policy import POLICIES, READS, THINK_BUDGET, THINK_GATE
from .readout import INPUT_REPEAT, LETTERS, MAX_CODES
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
    parser.add_argument(
        "--calibration", default=None, metavar="SPEC",
        help=("calibration record: a family (" + ", ".join(sorted(CALIBRATION_FILES))
              + ") or a path to a record. Default: chosen from the served model id"),
    )
    parser.add_argument("--family", default=None, choices=FAMILIES,
                        help="the contract the served model is read through. Default: taken from "
                             "the model id, then the tokenizer, otherwise assumed to be de2")
    parser.add_argument("--policy", default=None, choices=POLICIES,
                        help="the DE-2 read. Default: repeat-think for de2; de1 is always direct")
    parser.add_argument("--reads", default=None, choices=READS,
                        help="how often the question is written. Default: double for de2; de1 is always single")
    parser.add_argument("--reasoning", default=None, action=argparse.BooleanOptionalAction,
                        help="think when the read is unsure, then read again. Default: on for de2")
    parser.add_argument("--reasoning-prob", type=float, default=None, metavar="P",
                        help=f"think when the top probability is below P (default: {THINK_GATE})")
    parser.add_argument("--reasoning-len", type=int, default=None, metavar="N",
                        help=f"the most tokens a thought may run to (default: {THINK_BUDGET})")
    parser.add_argument("--compound", default=None, action=argparse.BooleanOptionalAction,
                        help="read a de1 choice above 26 options in two rounds. Default: on for de1")
    parser.add_argument("--timeout", type=float, default=120.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shisa-de", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check an endpoint, the served model, and the answer boundary")
    _common(doctor)
    doctor.add_argument("--json", action="store_true", help="print the report as JSON")
    doctor.add_argument("--probe", action="store_true",
                        help="also send one question and one chat request to check the served reads")

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
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        model = DecisionModel(
            base_url=args.base_url, model=args.model, tokenizer=args.tokenizer, timeout=args.timeout,
            image_top_logprobs=getattr(args, "image_top_logprobs", 20),
            calibration=resolve_calibration(args.calibration),
            family=args.family, policy=args.policy, reads=args.reads, reasoning=args.reasoning,
            reasoning_prob=args.reasoning_prob, reasoning_len=args.reasoning_len, compound=args.compound,
        )
    except ValueError as exc:
        parser.error(str(exc))
    with model:
        if args.command == "doctor":
            report = model.health(probe=args.probe)
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print(f"endpoint        {report['base_url']}")
                print(f"model           {report['model']}")
                print(f"family          {report['model_family']} ({report['family_source']})")
                print(f"readout         {report['readout_version']}, policy {report['policy']}")
                print(f"calibration     {report['calibration']}")
                print(f"  fitted on     {report.get('calibration_model')} "
                      f"({report.get('calibration_readout_version')}, "
                      f"{report.get('calibration_serving_shape')})")
                print(f"  match         {report['calibration_level'] if report.get('calibration_match') else 'NO'}")
                print(f"  applied       {'yes' if report.get('calibration_applied') else 'no'}")
                for reason in report.get("calibration_mismatch", []) + report.get("calibration_note", []):
                    print(f"                {reason}")
                print(f"models endpoint {report.get('models_status')}")
                if report.get("served_models") is not None:
                    listed = "yes" if report.get("model_listed") else "no"
                    print(f"model listed    {listed} (of {len(report['served_models'])} served ids)")
                print(f"boundary check  {report.get('boundary_check')}")
                if "read_probe" in report:
                    print(f"read probe      {report['read_probe']}")
                    print(f"chat system     {report['chat_system_render']}")
                if report.get("family_note"):
                    print(f"note            {report['family_note']}")
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
    options = question.options()
    de2 = model.family == "de2"
    if de2:
        # The first read of the DE-2 policy. A thought, when the gate asks for
        # one, follows this read and is not shown here.
        repeat = 1 if model.policy == "direct" else 2
        prompt = readout.render(state, question, repeat=repeat, max_options=MAX_CODES)
        slots = readout.code_slots(len(options))
        readout.check_code_boundary(readout.ensure_tokenizer().encode(prompt, add_special_tokens=False), slots)
    else:
        prompt = readout.render(state, question)
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
    body: dict[str, Any] = {"model": model.model, "prompt": prompt, "max_tokens": 1, "temperature": 0,
                            "logprobs": readout.max_logprobs}
    if de2:
        read = readout.read_codes(prompt, slots, debug=True)
        body.update(return_tokens_as_token_ids=True, logprob_token_ids=[slot.token_id for slot in slots])
        print(f"3. One request: max_tokens 1, temperature 0, the {len(slots)} answer codes by token id")
    else:
        read = readout.read(prompt, len(options), debug=True)
        print(f"3. One request: max_tokens 1, temperature 0, logprobs {readout.max_logprobs}")
    print("-" * 72)
    print(f"   POST {model.base_url}/v1/completions")
    print(f"   prompt tokens {read.prompt_tokens}, requests {read.requests}, sampled {read.sampled!r}")
    marks = {slot.letter: slot.letter for slot in slots} | {f"token_id:{slot.token_id}": slot.letter for slot in slots}
    for token, logprob in sorted(read.top_logprobs.items(), key=lambda kv: -kv[1]):
        marker = f"  <- answer slot {marks[token]}" if token in marks else ""
        print(f"   {token!r:>18}  {logprob:9.4f}{marker}")
    if read.missing_from_top:
        print(f"   fallback requests for codes not returned: {', '.join(read.missing_from_top)}")
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
    print(f"curl -s {shlex.quote(model.base_url + '/v1/completions')} \\")
    print("  -H 'Content-Type: application/json' \\")
    if _api_key_from_env():
        print('  -H "Authorization: Bearer ${SHISA_DE_API_KEY:-$SHISA_API_KEY}" \\')
    print(f"  -d {shlex.quote(json.dumps(body))}")
    print()
    if de2:
        top = max(read.probabilities.values())
        would = (model.policy == "repeat-think" and top < model.think_gate and len(options) <= 26)
        print(f"readout {model.readout_version}, policy {model.policy}; {MAX_CODES} codes available, "
              f"{len(options)} used")
        if model.policy != "direct":
            print(f"the user turn is written twice, joined by {INPUT_REPEAT!r}")
        if model.policy == "repeat-think":
            print(f"top probability {top:.4f} against the gate {model.think_gate}: "
                  f"{'a thought would follow this read' if would else 'this read is the answer'}")
    else:
        print(f"readout {model.readout_version}; {len(LETTERS)} letters available, {len(options)} used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
