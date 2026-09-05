import ast
import asyncio
import concurrent.futures
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from sglang.srt.disaggregation import utils
from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class ArrayDecoder(VideoDecoderWrapper):
    def __init__(self):
        self.frames = np.full((4, 8, 8, 3), 17, dtype=np.uint8)
        self.closed = False

    def __len__(self):
        return len(self.frames)

    @property
    def avg_fps(self):
        return 2.0

    def get_frames_as_tensor(self, indices):
        return self.frames[indices]

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "kind", ["numpy-list", "pil-list", "tensor", "structured", "mixed"]
)
def test_encoder_dispatches_video_representations(kind):
    frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in (5, 9)]
    decoder = ArrayDecoder()
    structured = [
        {"frame_image": frame, "timestamp": i} for i, frame in enumerate(frames)
    ]
    videos = {
        "numpy-list": [frames],
        "pil-list": [[Image.fromarray(frame) for frame in frames]],
        "tensor": [np.stack(frames)],
        "structured": [structured],
        "mixed": [structured, frames, decoder],
    }[kind]
    processor = object.__new__(EncoderPreprocessor)
    processor.model_type = "glm5_next"
    processor.video_processor = None
    processor.vision_config = {"video": {"fps": 2}}
    processor.server_args = SimpleNamespace(mm_enable_dp_encoder=False)

    def loaded(items, modalities):
        futures = []
        for video in videos:
            future = concurrent.futures.Future()
            future.set_result(video)
            futures.append(future)
        return futures, None

    processor._submit_data_loading_tasks = loaded
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        processor.io_executor = executor
        with patch(
            "sglang.srt.disaggregation.encoder.preprocessor.get_parallel",
            return_value=SimpleNamespace(attn_tp_size=1),
        ):
            result, kwargs = asyncio.run(
                processor._flatten_and_load_videos(["synthetic"] * len(videos))
            )
    assert len(result) == len(videos)
    assert len(kwargs["video_metadata"]) == len(videos)
    assert kwargs["do_sample_frames"] is False
    assert kwargs["return_metadata"] is True
    np.testing.assert_array_equal(np.asarray(result[0]), np.stack(frames))
    if kind == "mixed":
        np.testing.assert_array_equal(result[2], decoder.frames)
        assert decoder.closed


@pytest.mark.parametrize("side", ["prefill", "decode"])
@pytest.mark.parametrize("seq_len,expected", [(8, []), (11, [7, 0, 3, 0, 0, 4])])
def test_dsa_tail_payload_uses_req_kv_owner(side, seq_len, expected):
    # Execute the actual nested payload independently of network/bootstrap setup.
    source = Path(utils.__file__).with_name(f"{side}.py")
    nodes = [
        n
        for n in ast.walk(ast.parse(source.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_dsa_tail_payload"
    ]
    assert len(nodes) == 1
    pool = SimpleNamespace(kpool_use_compress=True, index_kpool=4, tail_extra_slots=0)
    request = SimpleNamespace(kv=ReqKvInfo(req_pool_idx=7))
    namespace = {
        "get_dsa_tail_state_indices": utils.get_dsa_tail_state_indices,
        "req": request,
        "decode_req": SimpleNamespace(req=request),
        "seq_len": seq_len,
        "self": SimpleNamespace(
            token_to_kv_pool=pool,
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool),
        ),
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace
    )
    assert namespace["_dsa_tail_payload"]() == expected


def _infos(*ptrs):
    return list(ptrs), [64] * len(ptrs), [16] * len(ptrs)


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("has_draft", [False, True])
def test_hybrid_dsa_registration_covers_target_and_draft(compress, has_draft):
    def dsa(index, tail):
        pool = object.__new__(DSATokenToKVPool)
        pool.kpool_use_compress = compress
        pool.get_state_buf_infos = lambda: _infos(index)
        pool.get_compress_tail_buf_infos = lambda: (
            _infos(tail) if compress else ([], [], [])
        )
        return pool

    target = object.__new__(HybridLinearKVPool)
    target.use_dsa = True
    target.full_kv_pool = dsa(101, 102)
    target.mamba_pool = SimpleNamespace(
        get_contiguous_buf_infos=lambda: _infos(100),
        get_state_dim_per_tensor=lambda: [4],
        get_state_conv_shard_groups=lambda: [[]],
        get_state_slice_outer_counts=lambda: [1],
        get_state_layer_ids=lambda: [0],
    )
    draft = dsa(201, 202) if has_draft else None
    args = KVArgs()
    utils.setup_state_kv_args(args, target, draft)
    expected = [(StateType.MAMBA, [100]), (StateType.DSA, [101])]
    if compress:
        expected.append((StateType.DSA_TAIL, [102]))
    if has_draft:
        expected.append((StateType.DSA, [201]))
        if compress:
            expected.append((StateType.DSA_TAIL, [202]))
    assert list(zip(args.state_types, args.state_data_ptrs)) == expected
    assert all(
        len(values) == len(expected)
        for values in (
            args.state_data_lens,
            args.state_item_lens,
            args.state_dim_per_tensor,
            args.state_conv_shard_groups,
            args.state_slice_outer_counts,
            args.state_layer_ids,
        )
    )
