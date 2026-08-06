from mlx_vlm.server.junie import parent_watchdog


def test_parent_watchdog_stops_after_parent_changes(monkeypatch):
    parent_pids = iter((42, 42, 1))
    sleeps = []
    stopped = []
    monkeypatch.setattr(parent_watchdog.os, "getppid", lambda: next(parent_pids))
    monkeypatch.setattr(parent_watchdog.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        parent_watchdog, "_terminate_worker", lambda: stopped.append(True)
    )

    parent_watchdog._watch_parent(42)

    assert sleeps == [parent_watchdog.POLL_INTERVAL_S] * 2
    assert stopped == [True]


def test_parent_watchdog_is_disabled_without_gateway(monkeypatch):
    monkeypatch.delenv(parent_watchdog.GATEWAY_PID_ENV, raising=False)

    assert parent_watchdog.start_parent_watchdog() is False
