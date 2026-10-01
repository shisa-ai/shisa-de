"""Bounded hosted API sweep. Run from the repository's installed environment."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import threading
import time
import zlib

import httpx

from shisa_de import Choice
from shisa_de.readout import Readout, READOUT_VERSION


class BudgetStop(RuntimeError):
    pass


class Meter(httpx.BaseTransport):
    """Count wire requests, including fallbacks; never retry failures."""
    def __init__(self, output, limit=1950, seconds=1080, transport=None):
        self.transport = transport or httpx.HTTPTransport(retries=0)
        self.output = output
        self.limit = limit
        self.seconds = seconds
        self.started = None
        self.lock = threading.Lock()
        self.rows = []
        self.count = 0
        self.tripped = False
        self.recovery = False
        self.local = threading.local()

    def handle_request(self, request):
        with self.lock:
            now = time.monotonic()
            if self.started is None:
                self.started = now
            if self.count >= self.limit or now - self.started >= self.seconds:
                raise BudgetStop('request or time cap reached')
            if self.tripped and not self.recovery:
                raise BudgetStop('circuit stopped after HTTP or transport failure')
            self.count += 1
            row = {'sequence': self.count, 'stage': getattr(self.local, 'stage', 'preflight'),
                   'utc': datetime.now(timezone.utc).isoformat(), 'path': request.url.path}
        started = time.monotonic()
        try:
            response = self.transport.handle_request(request)
            response.read()
            row['status'] = response.status_code
            row['request_id'] = response.headers.get('x-request-id')
            row['response_id'] = None
            if response.status_code >= 400:
                row['error'] = response.text[:1000]
                with self.lock:
                    self.tripped = True
            try:
                data = response.json()
                row['fingerprint'] = data.get('system_fingerprint')
                row['response_id'] = data.get('id')
                row['usage'] = data.get('usage')
                row['response_model'] = data.get('model')
            except (ValueError, AttributeError):
                row['json_error'] = True
            return response
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
            with self.lock:
                self.tripped = True
            raise
        finally:
            row['latency_ms'] = (time.monotonic() - started) * 1000
            with self.lock:
                self.rows.append(row)
                with self.output.open('a') as stream:
                    stream.write(json.dumps(row) + '\n')

    def close(self):
        self.transport.close()


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def validate(read, count):
    expected = {chr(65 + i) for i in range(count)}
    if set(read.probabilities) != expected or set(read.logprobs) != expected:
        raise ValueError('incomplete distribution')
    if not all(math.isfinite(v) for v in read.logprobs.values()):
        raise ValueError('nonfinite logprob')
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in read.probabilities.values()):
        raise ValueError('invalid probabilities')
    if not math.isclose(sum(read.probabilities.values()), 1, abs_tol=1e-9):
        raise ValueError('distribution does not sum to one')


def image(size, index):
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    rgb = bytes((255, index % 32, index % 17))
    raw = (b'\x00' + rgb * size) * size
    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', size, size, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


def prepare(readout, target, count, index):
    question = Choice('Which category matches the evidence?',
                      {f'category_{i}': f'The evidence identifies category {i}.' for i in range(count)})
    rng = random.Random(index)
    words = ['river', 'stone', 'cloud', 'garden', 'road', 'paper', 'green', 'quiet']
    padding = ' '.join(rng.choices(words, k=target))
    def render(length):
        return readout.render({'case': index, 'notes': padding[:length], 'category': index % count}, question)
    low, high = 0, len(padding)
    tok = readout.ensure_tokenizer()
    while low < high:
        mid = (low + high + 1) // 2
        if len(tok.encode(render(mid), add_special_tokens=False)) <= target:
            low = mid
        else:
            high = mid - 1
    prompt = render(low)
    tokens = len(tok.encode(prompt, add_special_tokens=False))
    readout.check_boundary(prompt, readout.slots(count))
    return prompt, tokens


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    meter = Meter(output / 'http.jsonl', args.max_requests, args.seconds)
    endpoint = os.environ.get('SHISA_DE_ENDPOINT', 'https://api.shisa.ai/openai').rstrip('/')
    if endpoint.endswith('/v1'):
        endpoint = endpoint[:-3]
    readout = Readout(endpoint, 'shisa-ai/shisa-de-1', api_key=os.environ.get('SHISA_DE_API_KEY') or os.environ.get('SHISA_API_KEY'),
                      timeout=60, transport=meter)
    metadata = {'started_utc': datetime.now(timezone.utc).isoformat(), 'endpoint': endpoint,
                'model': readout.model, 'readout_version': READOUT_VERSION, 'args': vars(args),
                'notes': 'No retries. HTTP latency includes body transfer; question latency includes local boundary checks and fallbacks. No server telemetry.'}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    stages = []
    question = Choice('What is the dominant color?', {'red': 'Red', 'blue': 'Blue'})
    try:
        tokenizer = readout.ensure_tokenizer()
        metadata['tokenizer_commit'] = tokenizer.init_kwargs.get('_commit_hash')
        readout.slots(26)
        readout.answer_prefix_id()
        if readout.model not in readout.list_models():
            raise ValueError('model not listed')
        specs = [('smoke', 1, 2, 128, 2, 'text')]
        specs += [(f'short-c{c}', c, max(16, c * 2), 256, 2, 'text') for c in [1, 2, 4, 8, 16, 32]]
        specs += [(f'context-{n}-c{c}', c, max(4, c), n, 2, 'text') for n in [2048, 8192, 16384, 30000] for c in [1, 8, 32]]
        specs += [(f'options-{k}', 8, 8, 1024, k, 'text') for k in [2, 8, 20, 26]]
        specs += [(f'image-{size}-c{c}', c, max(4, c), size, 2, 'image') for size in [32, 512, 1024] for c in [1, 8, 32]]
        specs += [(f'mixed-c{c}', c, c * 2, 8192, 2, 'mixed') for c in [8, 32]]
        metadata['planned_stages'] = len(specs)
        for stage_index, (name, concurrency, count, target, options, kind) in enumerate(specs):
            if meter.tripped or meter.count >= meter.limit or time.monotonic() - meter.started >= meter.seconds:
                break
            jobs = []
            for i in range(count):
                index = stage_index * 1000 + i
                is_image = kind == 'image' or (kind == 'mixed' and i % 3 == 0)
                if is_image:
                    jobs.append(('image', image(target if kind == 'image' else 512, index), 2, None))
                else:
                    k = 26 if kind == 'mixed' and i % 3 == 1 else options
                    prompt, tokens = prepare(readout, target, k, index)
                    jobs.append(('text', prompt, k, tokens))
            begin = time.monotonic()
            def execute(job):
                meter.local.stage = name
                start = time.monotonic()
                mode, payload, k, tokens = job
                row = {'kind': mode, 'options': k, 'local_prompt_tokens': tokens,
                       'input_sha256': hashlib.sha256(payload.encode()).hexdigest()}
                try:
                    result = readout.read_image({}, question, payload) if mode == 'image' else readout.read(payload, k)
                    validate(result, k)
                    row.update(ok=True, requests=result.requests, prompt_tokens=result.prompt_tokens,
                               answer=result.answer(), probabilities=result.probabilities, sampled=result.sampled,
                               missing=result.missing_from_top)
                    if mode == 'image' and result.answer() != 'A':
                        row['semantic_mismatch'] = True
                except BudgetStop as exc:
                    row.update(ok=False, stopped=str(exc))
                except Exception as exc:
                    row.update(ok=False, error=f'{type(exc).__name__}: {exc}')
                row['latency_ms'] = (time.monotonic() - start) * 1000
                return row
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                results = list(pool.map(execute, jobs))
            duration = time.monotonic() - begin
            rows = [r for r in meter.rows if r['stage'] == name]
            latencies = [r['latency_ms'] for r in rows]
            stage = {'name': name, 'concurrency': concurrency, 'duration_s': duration,
                     'questions': len(results), 'successful_questions': sum(r['ok'] for r in results),
                     'http_requests': len(rows), 'http_errors': sum('error' in r for r in rows),
                     'http_rps': len(rows) / duration, 'http_p50_ms': percentile(latencies, .5),
                     'http_p95_ms': percentile(latencies, .95), 'http_p99_ms': percentile(latencies, .99),
                     'results': results}
            stages.append(stage)
            (output / 'stages.json').write_text(json.dumps(stages, indent=2))
            print(json.dumps({k: v for k, v in stage.items() if k != 'results'}), flush=True)
            if any(not r['ok'] or r.get('semantic_mismatch') for r in results):
                break
        # Exactly one serial recovery probe, still inside the shared caps.
        meter.recovery = True
        meter.local.stage = 'recovery'
        prompt, _ = prepare(readout, 128, 2, 999999)
        result = readout.read(prompt, 2)
        validate(result, 2)
        metadata['recovery'] = 'passed'
    except Exception as exc:
        metadata['fatal_error'] = f'{type(exc).__name__}: {exc}'
        print(metadata['fatal_error'], flush=True)
    finally:
        readout.close()
        metadata.update(finished_utc=datetime.now(timezone.utc).isoformat(), http_requests=meter.count,
                        failed_requests=sum('error' in r for r in meter.rows),
                        fingerprints=sorted({r['fingerprint'] for r in meter.rows if r.get('fingerprint')}),
                        completed_stages=len(stages), circuit_tripped=meter.tripped,
                        load_elapsed_s=time.monotonic() - meter.started if meter.started else 0,
                        invalid_questions=sum('error' in r for s in stages for r in s['results']),
                        stopped_questions=sum('stopped' in r for s in stages for r in s['results']))
        (output / 'summary.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata), flush=True)
    return 1 if metadata.get('fatal_error') or metadata['failed_requests'] or any(not r['ok'] or r.get('semantic_mismatch') for s in stages for r in s['results']) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/stress-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--max-requests', type=int, default=1950, help='HTTP request cap, including fallbacks and recovery (maximum 1950).')
    parser.add_argument('--seconds', type=int, default=1080, help='Stop starting requests after this many seconds (maximum 1080).')
    args = parser.parse_args()
    if not 1 <= args.max_requests <= 1950 or not 1 <= args.seconds <= 1080:
        parser.error('caps must be positive and within the approved limits')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
