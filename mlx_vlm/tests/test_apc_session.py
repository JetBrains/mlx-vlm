"""Tests for session-based exact-mode APC storage (APCSession).

For hybrid attention/SSM models (e.g. Qwen 3.5/3.6), whole-snapshot exact
entries duplicate the full-attention K/V per stored prefix. Session storage
keeps ONE shared K/V set at the longest (anchor) length plus small
recurrent-state checkpoints at recent prefix lengths, and can resume from a
checkpoint before the point where a request diverges from the anchor.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_vlm.apc import APCManager, _common_prefix_len

from mlx_vlm.tests.test_apc_exact_mode import _make_tiny_qwen35


def _prefill(lm, tokens, cache=None):
    cache = cache if cache is not None else lm.make_cache()
    lm(mx.array([tokens]), cache=cache)
    mx.eval([c for c in cache if c is not None])
    return cache


PREFIX = [1, 5, 10, 20, 30, 40, 50, 60, 2, 3, 4, 6, 7, 8, 9, 11]
EXT = [12, 13, 14, 15, 16, 17, 18, 19]
SUFFIX = [21, 22, 23, 24, 25, 26, 27, 28, 29]


def test_common_prefix_len():
    assert _common_prefix_len((), (1, 2)) == 0
    assert _common_prefix_len((1, 2, 3), (1, 2, 3)) == 3
    assert _common_prefix_len((1, 2, 3), (1, 2, 4, 5)) == 2
    assert _common_prefix_len((1, 2), (1, 2, 3)) == 2
    assert _common_prefix_len((9,), (1, 2)) == 0


def test_hybrid_store_routes_to_session_not_entries():
    lm = _make_tiny_qwen35()
    apc = APCManager(num_blocks=64, block_size=16)

    cache = _prefill(lm, PREFIX)
    assert apc.store_exact_cache(PREFIX, cache)

    assert len(apc._sessions) == 1
    assert not apc._exact_cache  # no whole-snapshot duplicate in memory
    sess = next(iter(apc._sessions.values()))
    assert sess.token_ids == tuple(PREFIX)
    assert list(sess.checkpoints) == [len(PREFIX)]
    # Full-attention slots hold KV, recurrent slots hold None (and vice versa
    # inside the checkpoint rows).
    kv_slots = [c is not None for c in sess.kv_caches]
    state_slots = [c is not None for c in sess.checkpoints[len(PREFIX)]]
    assert any(kv_slots) and any(state_slots)
    assert all(k != s for k, s in zip(kv_slots, state_slots))


def test_extending_store_shares_one_kv_set():
    lm = _make_tiny_qwen35()
    apc = APCManager(num_blocks=64, block_size=16)

    cache = _prefill(lm, PREFIX)
    apc.store_exact_cache(PREFIX, cache)
    # Continue the same conversation and store the longer prefix.
    lm(mx.array([EXT]), cache=cache)
    mx.eval([c for c in cache if c is not None])
    apc.store_exact_cache(PREFIX + EXT, cache)

    assert len(apc._sessions) == 1
    sess = next(iter(apc._sessions.values()))
    # Anchor advanced to the longer prefix; both checkpoints retained.
    assert sess.token_ids == tuple(PREFIX + EXT)
    assert sorted(sess.checkpoints) == [len(PREFIX), len(PREFIX) + len(EXT)]
    # The anchor KV covers the longest length exactly once.
    kv = next(c for c in sess.kv_caches if c is not None)
    assert kv.offset == len(PREFIX) + len(EXT)


def test_lookup_hits_latest_checkpoint():
    lm = _make_tiny_qwen35()
    apc = APCManager(num_blocks=64, block_size=16)

    cache = _prefill(lm, PREFIX)
    apc.store_exact_cache(PREFIX, cache)
    lm(mx.array([EXT]), cache=cache)
    mx.eval([c for c in cache if c is not None])
    apc.store_exact_cache(PREFIX + EXT, cache)

    restored, plen = apc.lookup_exact_cache(PREFIX + EXT + SUFFIX)
    assert restored is not None
    assert plen == len(PREFIX) + len(EXT)
    snap = apc.stats_snapshot()
    assert snap["exact_hits"] == 1
    assert snap["matched_tokens"] == plen


def test_lookup_rolls_back_to_checkpoint_before_divergence():
    """A request that diverges mid-history resumes from an earlier
    checkpoint — impossible with whole-snapshot entries, whose full stored
    sequence must be a prefix of the request."""
    lm = _make_tiny_qwen35()
    apc = APCManager(num_blocks=64, block_size=16)

    cache = _prefill(lm, PREFIX)
    apc.store_exact_cache(PREFIX, cache)
    lm(mx.array([EXT]), cache=cache)
    mx.eval([c for c in cache if c is not None])
    apc.store_exact_cache(PREFIX + EXT, cache)

    # Diverges 4 tokens into EXT: common prefix = len(PREFIX) + 4, so the
    # anchor-length checkpoint cannot match, but the PREFIX one can.
    edited = PREFIX + EXT[:4] + [63, 62, 61] + SUFFIX
    restored, plen = apc.lookup_exact_cache(edited)
    assert restored is not None
    assert plen == len(PREFIX)


def test_warm_resume_matches_split_prefill():
    """Resuming from a session checkpoint is EXACT: byte-identical to
    prefilling the same token sequence in the same two steps without APC.

    (A single-shot cold prefill of the full sequence is NOT the reference:
    the linear-attention chunked scan is batch non-invariant, so different
    prefill chunking legitimately produces tiny fp diffs — see the module
    docstring in mlx_vlm/apc.py.)
    """
    lm = _make_tiny_qwen35()
    apc = APCManager(num_blocks=64, block_size=16)

    warm_src = _prefill(lm, PREFIX)
    apc.store_exact_cache(PREFIX, warm_src)

    full = PREFIX + SUFFIX
    restored, plen = apc.lookup_exact_cache(full)
    assert plen == len(PREFIX)
    lm(mx.array([full[plen:]]), cache=restored)
    mx.eval([c for c in restored if c is not None])

    # Reference: same two-step prefill, no APC round-trip.
    ref = _prefill(lm, PREFIX)
    lm(mx.array([SUFFIX]), cache=ref)
    mx.eval([c for c in ref if c is not None])

    for i, (cw, cc) in enumerate(zip(restored, ref)):
        if getattr(cw, "keys", None) is not None:
            off = cc.offset
            assert cw.offset == off
            assert (
                mx.max(mx.abs(cw.keys[..., :off, :] - cc.keys[..., :off, :])).item()
                == 0
            ), f"layer {i} keys diverge"
            assert (
                mx.max(
                    mx.abs(cw.values[..., :off, :] - cc.values[..., :off, :])
                ).item()
                == 0
            ), f"layer {i} values diverge"
        elif getattr(cw, "cache", None) is not None:
            for j, (sa, sb) in enumerate(zip(cw.cache, cc.cache)):
                if sa is not None and sb is not None:
                    assert (
                        mx.max(mx.abs(sa - sb)).item() == 0
                    ), f"layer {i} state[{j}] diverges"


def test_restore_does_not_corrupt_session():
    """Mutating a restored cache must not affect the stored session."""
    lm = _make_tiny_qwen35()
    apc = APCManager(num_blocks=64, block_size=16)

    cache = _prefill(lm, PREFIX)
    apc.store_exact_cache(PREFIX, cache)

    full = PREFIX + SUFFIX
    restored_a, _ = apc.lookup_exact_cache(full)
    lm(mx.array([SUFFIX]), cache=restored_a)  # mutate the restored copy
    mx.eval([c for c in restored_a if c is not None])

    restored_b, plen = apc.lookup_exact_cache(full)
    assert plen == len(PREFIX)
    for cb in restored_b:
        if getattr(cb, "keys", None) is not None:
            assert cb.offset == len(PREFIX)


def test_checkpoint_count_is_capped(monkeypatch):
    monkeypatch.setenv("APC_SESSION_CHECKPOINTS", "3")
    lm = _make_tiny_qwen35()
    apc = APCManager(num_blocks=64, block_size=16)

    cache = lm.make_cache()
    tokens: list[int] = []
    for step in range(5):
        chunk = [step * 8 + k + 1 for k in range(8)]
        lm(mx.array([chunk]), cache=cache)
        mx.eval([c for c in cache if c is not None])
        tokens += chunk
        apc.store_exact_cache(tokens, cache)

    sess = next(iter(apc._sessions.values()))
    assert len(sess.checkpoints) == 3
    # Most recent lengths survive; the anchor KV still covers all of them.
    assert sorted(sess.checkpoints) == [24, 32, 40]
    assert sess.token_ids == tuple(tokens)


def test_divergent_conversation_creates_second_session(monkeypatch):
    monkeypatch.setenv("APC_EXACT_SESSIONS", "2")
    lm = _make_tiny_qwen35()
    apc = APCManager(num_blocks=64, block_size=16)

    a = PREFIX
    b = [2, 4, 8, 16, 32, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    apc.store_exact_cache(a, _prefill(lm, a))
    apc.store_exact_cache(b, _prefill(lm, b))

    assert len(apc._sessions) == 2
    ra, la = apc.lookup_exact_cache(a + SUFFIX)
    rb, lb = apc.lookup_exact_cache(b + SUFFIX)
    assert ra is not None and la == len(a)
    assert rb is not None and lb == len(b)


def test_sessions_disabled_falls_back_to_entries(monkeypatch):
    monkeypatch.setenv("APC_EXACT_SESSIONS", "0")
    lm = _make_tiny_qwen35()
    apc = APCManager(num_blocks=64, block_size=16)

    cache = _prefill(lm, PREFIX)
    assert apc.store_exact_cache(PREFIX, cache)
    assert not apc._sessions
    assert len(apc._exact_cache) == 1
    restored, plen = apc.lookup_exact_cache(PREFIX + SUFFIX)
    assert restored is not None and plen == len(PREFIX)


def test_clear_drops_sessions():
    lm = _make_tiny_qwen35()
    apc = APCManager(num_blocks=64, block_size=16)
    apc.store_exact_cache(PREFIX, _prefill(lm, PREFIX))
    assert apc._sessions
    apc.clear()
    assert not apc._sessions
    restored, plen = apc.lookup_exact_cache(PREFIX + SUFFIX)
    assert restored is None and plen == 0


def test_stats_snapshot_reports_sessions():
    lm = _make_tiny_qwen35()
    apc = APCManager(num_blocks=64, block_size=16)
    apc.store_exact_cache(PREFIX, _prefill(lm, PREFIX))
    snap = apc.stats_snapshot()
    (info,) = snap["exact_sessions"]
    assert info["anchor_tokens"] == len(PREFIX)
    assert info["checkpoints"] == [len(PREFIX)]
    assert info["kv_bytes"] > 0
