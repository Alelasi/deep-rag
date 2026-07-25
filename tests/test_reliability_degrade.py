"""reliability 熔断与降级单测"""


def test_circuit_breaker_opens():
    from src.reliability import CircuitBreaker, DegradePolicy

    b = CircuitBreaker(DegradePolicy(max_failures=2, open_seconds=60))
    assert b.allow() is True
    b.record_failure()
    assert b.state == "closed"
    b.record_failure()
    assert b.state == "open"
    assert b.allow() is False
    b.record_success()
    assert b.state == "closed"


def test_degrade_answer_shape():
    from src.reliability import degrade_answer

    r = degrade_answer("429")
    assert r["degraded"] is True
    assert "不可用" in r["answer"] or "降级" in r["answer"]
