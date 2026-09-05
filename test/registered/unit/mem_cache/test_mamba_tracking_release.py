from array import array

import pytest
import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("keep", [None, 0])
def test_mamba_release_and_reallocate_preserves_ownership(lazy, keep):
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=8,
        n_groups=1,
        num_heads=1,
        head_dim=8,
        state_size=4,
        conv_kernel=2,
    )
    pool = HybridReqToTokenPool(
        size=4,
        mamba_size=24,
        mamba_spec_state_size=0,
        max_context_len=16,
        device="cpu",
        enable_memory_saver=False,
        cache_params=Mamba2CacheParams(shape=shape, layers=[0]),
        mamba_layer_ids=[0],
        enable_mamba_extra_buffer=True,
        enable_mamba_extra_buffer_lazy=lazy,
    )

    def request(rid):
        return Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", [1]),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )

    req, other = request("first"), request("second")
    initial = pool.mamba_allocator.available_size()
    pool.alloc([req])
    kept = (
        req.kv.mamba_ping_pong_track_buffer[keep : keep + 1].clone()
        if keep is not None
        else None
    )
    req.kv.mamba_cow_src_index = torch.tensor(7)
    req.kv.mamba_last_track_seqlen = 12
    req.mamba_branching_seqlen = 8
    pool.free_mamba_cache(req, keep)
    pool.free(req)
    for field in (
        "mamba_pool_idx",
        "mamba_ping_pong_track_buffer",
        "mamba_next_track_idx",
        "mamba_last_track_idx",
        "mamba_last_track_seqlen",
        "mamba_cow_src_index",
    ):
        assert getattr(req.kv, field) is None
    assert req.kv.mamba_needs_clear is False
    assert req.mamba_branching_seqlen is None
    assert pool.mamba_allocator.available_size() == initial - (keep is not None)
    pool.alloc([other, req])

    def owned(r):
        slots = [int(r.kv.mamba_pool_idx)] + r.kv.mamba_ping_pong_track_buffer.tolist()
        return {slot for slot in slots if slot >= 0}

    assert owned(req).isdisjoint(owned(other))
    if kept is not None:
        assert int(kept[0]) not in owned(req) | owned(other)
    for r in (req, other):
        pool.free_mamba_cache(r)
        pool.free(r)
    if kept is not None:
        pool.mamba_allocator.free(kept)
    assert pool.mamba_allocator.available_size() == initial
