"""Tests for the mid-prefill cancel snapshot.

A prefill that outlives the client's retry timeout used to lose all its
work: the cancel dropped the prompt batch, and the identical resent request
re-prefilled from scratch. Now ``BatchGenerator.remove`` snapshots the
row's progress into the APC exact cache (and, under the disk scope, the
disk tier) via ``PromptProcessingBatch.harvest_partial_prefill`` before
dropping it, so the retry resumes where the cancelled request stopped.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_vlm.apc import APCManager, _sequence_hash
from mlx_vlm.generate import BatchGenerator

from mlx_vlm.tests.test_apc_exact_mode import _make_tiny_qwen35


PROMPT = [(i * 7 + 3) % 60 for i in range(64)]


class _StoppingCriteria:
    def __init__(self):
        self.eos_token_ids = []

    def __call__(self, token):
        return False

    def add_eos_token_ids(self, tokens):
        pass

    def reset(self, *args, **kwargs):
        pass


class _Processor:
    def __init__(self):
        self.tokenizer = self
        self.stopping_criteria = _StoppingCriteria()

    def decode(self, tokens):
        return ""


def _embeds(lm, ids):
    return {"inputs_embeds": lm.model.embed_tokens(mx.array([ids]))}


def _make_generator(lm, manager, prefill_step_size=8):
    return BatchGenerator(
        lm,
        _Processor(),
        max_tokens=4,
        prefill_step_size=prefill_step_size,
        apc_manager=manager,
        compute_logprobs=False,
    )


def _prefill_until(gen, min_columns, max_steps=64):
    """Drive gen.next() until at least min_columns prompt columns are done."""
    for _ in range(max_steps):
        batch = gen._prompt_batch
        if batch is not None and batch._processed_prompt_columns >= min_columns:
            return batch._processed_prompt_columns
        gen.next()
    raise AssertionError("prefill did not reach the requested column count")


def test_remove_mid_prefill_snapshots_progress():
    lm = _make_tiny_qwen35()
    manager = APCManager(num_blocks=4)
    gen = _make_generator(lm, manager)
    try:
        assert gen.apc_mode == "exact"
        kwargs = _embeds(lm, PROMPT)
        (uid,) = gen.insert([PROMPT], max_tokens=4, prompt_kwargs=[kwargs])
        processed = _prefill_until(gen, 24)
        assert processed < len(PROMPT)
        assert gen.remove(uid)
        warm, matched = manager.lookup_exact_cache(
            PROMPT,
            extra_hash=gen._apc_extra_hash(_embeds(lm, PROMPT)),
        )
        assert matched == processed
        assert warm is not None
    finally:
        gen.close()


def test_retry_after_cancel_resumes_from_snapshot():
    lm = _make_tiny_qwen35()
    manager = APCManager(num_blocks=4)
    gen = _make_generator(lm, manager)
    try:
        kwargs = _embeds(lm, PROMPT)
        (uid,) = gen.insert([PROMPT], max_tokens=4, prompt_kwargs=[kwargs])
        processed = _prefill_until(gen, 24)
        gen.remove(uid)
        # The identical retry must pick up the warm prefix instead of a
        # cold start.
        pick = gen._apc_pick_for((2, list(PROMPT), 4, _embeds(lm, PROMPT), None, None))
        assert pick is not None
        assert pick["prefix_len"] == processed
    finally:
        gen.close()


def test_cancel_snapshot_is_written_to_disk_under_all_scope(
    monkeypatch, tmp_path
):
    from mlx_vlm.apc import DiskBlockStore

    monkeypatch.setenv("APC_DISK_EXACT_SCOPE", "all")
    lm = _make_tiny_qwen35()
    disk = DiskBlockStore(tmp_path, namespace="unit")
    manager = APCManager(num_blocks=4, disk=disk)
    gen = _make_generator(lm, manager)
    try:
        kwargs = _embeds(lm, PROMPT)
        (uid,) = gen.insert([PROMPT], max_tokens=4, prompt_kwargs=[kwargs])
        processed = _prefill_until(gen, 24)
        gen.remove(uid)
        key = _sequence_hash(
            tuple(PROMPT[:processed]),
            gen._apc_extra_hash(_embeds(lm, PROMPT)),
            manager.block_size,
        )
        assert disk.has_exact_or_pending(key)
    finally:
        gen.close()
        manager.close()


def test_remove_before_any_prefill_stores_nothing():
    lm = _make_tiny_qwen35()
    manager = APCManager(num_blocks=4)
    gen = _make_generator(lm, manager)
    try:
        (uid,) = gen.insert(
            [PROMPT], max_tokens=4, prompt_kwargs=[_embeds(lm, PROMPT)]
        )
        assert gen.remove(uid)  # still queued, no prompt batch yet
        warm, matched = manager.lookup_exact_cache(PROMPT, extra_hash=0)
        assert matched == 0
        assert warm is None
    finally:
        gen.close()
