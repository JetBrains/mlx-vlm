from types import SimpleNamespace

from mlx_vlm.server.junie import memory
from mlx_vlm.server.runtime import runtime


def test_memory_stats_reports_zero_kv_cache_without_apc_manager(monkeypatch):
    monkeypatch.setattr(runtime, "apc_manager", None)

    stats = memory.memory_stats()

    assert stats["kv_cache_gb"] == 0
    assert set(stats) == {"total_gb", "peak_gb", "kv_cache_gb"}


def test_kv_cache_bytes_sums_pool_and_exact_sessions(monkeypatch):
    manager = SimpleNamespace(
        stats_snapshot=lambda: {
            "resident_bytes": 2 * 2**20,
            "exact_sessions": [{"kv_bytes": 3 * 2**20}, {"kv_bytes": 5 * 2**20}],
        }
    )
    monkeypatch.setattr(runtime, "apc_manager", manager)

    assert memory.kv_cache_bytes() == 10 * 2**20


def test_kv_cache_bytes_is_zero_when_stats_snapshot_raises(monkeypatch):
    manager = SimpleNamespace(stats_snapshot=lambda: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(runtime, "apc_manager", manager)

    assert memory.kv_cache_bytes() == 0
