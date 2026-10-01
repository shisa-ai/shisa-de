"""Offline safety checks for the opt-in stress harness."""
from concurrent.futures import ThreadPoolExecutor
import json
import time

import httpx
import pytest

from scripts.stress_api import BudgetStop, Meter, percentile, validate
from shisa_de.readout import LetterRead


def make_meter(tmp_path, handler=None, **kwargs):
    return Meter(tmp_path / 'http.jsonl', transport=httpx.MockTransport(
        handler or (lambda request: httpx.Response(200, json={'system_fingerprint': 'test'}))
    ), **kwargs)


def test_concurrent_requests_never_exceed_budget(tmp_path):
    meter = make_meter(tmp_path, limit=7)
    with httpx.Client(transport=meter) as client:
        def request(_):
            try:
                client.post('https://test/v1/completions', json={})
                return True
            except BudgetStop:
                return False
        with ThreadPoolExecutor(max_workers=16) as pool:
            assert sum(pool.map(request, range(50))) == 7
    assert meter.count == len(meter.rows) == 7
    assert len((tmp_path / 'http.jsonl').read_text().splitlines()) == 7


def test_fallback_requests_each_consume_budget(tmp_path):
    meter = make_meter(tmp_path, limit=2)
    with httpx.Client(transport=meter) as client:
        for body in [{'logprobs': 20}, {'prompt_logprobs': 0}]:
            client.post('https://test/v1/completions', json=body)
        with pytest.raises(BudgetStop):
            client.post('https://test/v1/completions', json={'prompt_logprobs': 0})
    assert meter.count == 2


@pytest.mark.parametrize('status', [400, 429, 500, 503])
def test_error_stops_load_but_recovery_still_consumes_budget(tmp_path, status):
    meter = make_meter(tmp_path, lambda r: httpx.Response(status, json={'error': 'capacity'}), limit=2)
    with httpx.Client(transport=meter) as client:
        assert client.get('https://test/').status_code == status
        with pytest.raises(BudgetStop):
            client.get('https://test/')
        meter.recovery = True
        client.get('https://test/')
        with pytest.raises(BudgetStop):
            client.get('https://test/')
    assert meter.count == 2
    assert all('error' in r for r in meter.rows)


def test_timeout_stops_load_and_is_logged(tmp_path):
    def timeout(request):
        raise httpx.ReadTimeout('test timeout', request=request)
    meter = make_meter(tmp_path, timeout)
    with httpx.Client(transport=meter) as client:
        with pytest.raises(httpx.ReadTimeout):
            client.get('https://test/')
        with pytest.raises(BudgetStop):
            client.get('https://test/')
    row = json.loads((tmp_path / 'http.jsonl').read_text())
    assert 'ReadTimeout' in row['error']
    assert row['latency_ms'] >= 0
    assert meter.count == 1


def test_deadline_applies_even_to_recovery(tmp_path):
    meter = make_meter(tmp_path, seconds=1)
    meter.started = time.monotonic() - 2
    meter.recovery = True
    with httpx.Client(transport=meter) as client:
        with pytest.raises(BudgetStop):
            client.get('https://test/')
    assert meter.count == 0


def test_response_fingerprint_usage_and_stage_are_preserved(tmp_path):
    meter = make_meter(tmp_path, lambda r: httpx.Response(200, json={
        'system_fingerprint': 'test', 'usage': {'prompt_tokens': 100}, 'id': 'response', 'model': 'de1'
    }, headers={'x-request-id': 'request'}))
    meter.local.stage = 'options-26'
    with httpx.Client(transport=meter) as client:
        client.get('https://test/')
    row = meter.rows[0]
    assert row['fingerprint'] == 'test'
    assert row['usage']['prompt_tokens'] == 100
    assert row['request_id'] == 'request'
    assert row['stage'] == 'options-26'


def test_percentile_uses_nearest_rank():
    assert percentile([], .95) is None
    assert percentile([3, 1, 2], .5) == 2
    assert percentile(list(range(1, 101)), .95) == 95
    assert percentile([4], .99) == 4


def test_validate_accepts_complete_distribution():
    validate(LetterRead(logprobs={'A': -1, 'B': -1}, probabilities={'A': .5, 'B': .5}), 2)


@pytest.mark.parametrize('logprobs,probabilities', [
    ({'A': -1}, {'A': 1}),
    ({'A': float('nan'), 'B': -1}, {'A': .5, 'B': .5}),
    ({'A': -1, 'B': -1}, {'A': float('nan'), 'B': .5}),
    ({'A': -1, 'B': -1}, {'A': .2, 'B': .2}),
    ({'A': -1, 'B': -1}, {'A': -1, 'B': 2}),
])
def test_validate_rejects_bad_distribution(logprobs, probabilities):
    with pytest.raises(ValueError):
        validate(LetterRead(logprobs=logprobs, probabilities=probabilities), 2)
