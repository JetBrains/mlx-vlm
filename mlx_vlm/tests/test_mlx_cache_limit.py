import sys

import mlx.core as mx
import pytest

import mlx_vlm.server.app  # noqa: F401  (the package attribute is the FastAPI object)

server_app = sys.modules["mlx_vlm.server.app"]
from mlx_vlm_shared.server_settings import DEFAULT_CONFIG, is_valid_setting


def test_cache_limit_env_sets_allocator_cap(monkeypatch):
    calls = []
    monkeypatch.setattr(mx, "set_cache_limit", lambda limit: calls.append(limit))
    monkeypatch.setenv("MLX_VLM_CACHE_LIMIT_GB", "2.5")
    assert server_app._apply_mlx_cache_limit() == int(2.5 * (1 << 30))
    assert calls == [int(2.5 * (1 << 30))]


@pytest.mark.parametrize("raw", ["", "0", "-1", "abc"])
def test_cache_limit_unset_or_invalid_is_ignored(monkeypatch, raw):
    calls = []
    monkeypatch.setattr(mx, "set_cache_limit", lambda limit: calls.append(limit))
    monkeypatch.setenv("MLX_VLM_CACHE_LIMIT_GB", raw)
    assert server_app._apply_mlx_cache_limit() is None
    assert calls == []


def test_config_key_validation():
    assert DEFAULT_CONFIG["mlx_cache_limit_gb"] == 3
    assert is_valid_setting("mlx_cache_limit_gb", None)
    assert is_valid_setting("mlx_cache_limit_gb", 4)
    assert is_valid_setting("mlx_cache_limit_gb", 1.5)
    assert not is_valid_setting("mlx_cache_limit_gb", 0)
    assert not is_valid_setting("mlx_cache_limit_gb", True)
    assert not is_valid_setting("mlx_cache_limit_gb", "3")
