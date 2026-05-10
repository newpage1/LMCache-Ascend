# SPDX-License-Identifier: Apache-2.0
import torch

from lmcache_ascend.v1.kv_format import (
    KVCacheFormat,
    get_tuple_byte_offsets,
    get_tuple_bytes_per_token,
)
from lmcache_ascend.v1.kv_layer_groups import _get_kv_cache_group_key_and_info
from tests.v1.utils import generate_dsa_c8_kv_cache, generate_dsa_kv_cache


def test_detect_dsa_c8_kv_format():
    kv_caches = generate_dsa_c8_kv_cache(
        num_blocks=2,
        device="cpu",
        num_layers=1,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_size=16,
    )

    kv_format = KVCacheFormat.detect(kv_caches)

    assert kv_format == KVCacheFormat.DSA_C8_KV
    assert kv_format.is_dsa_format()
    assert kv_format.is_tuple_format()
    assert kv_format.requires_raw_byte_transfer()
    assert kv_format.get_kv_size() == 4
    assert "raw-byte NPU transfer support is not implemented" in (
        kv_format.unsupported_transfer_message("UnitTestConnector")
    )


def test_mixed_tuple_lengths_are_undefined():
    dsa_c8_cache = generate_dsa_c8_kv_cache(
        num_blocks=2,
        device="cpu",
        num_layers=1,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_size=16,
    )[0]
    dsa_cache = generate_dsa_kv_cache(
        num_blocks=2,
        device="cpu",
        num_layers=1,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        dsa_head_dim=128,
        block_size=16,
        dtype=torch.bfloat16,
    )[0]

    assert KVCacheFormat.detect([dsa_c8_cache, dsa_cache]) == KVCacheFormat.UNDEFINED


def test_dsa_c8_mixed_dtype_group_uses_raw_byte_storage():
    kv_cache = generate_dsa_c8_kv_cache(
        num_blocks=2,
        device="cpu",
        num_layers=1,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        indexer_heads=64,
        indexer_head_dim=128,
        block_size=16,
    )[0]

    key, storage_shape, dtype = _get_kv_cache_group_key_and_info(kv_cache)

    expected_bytes_per_token = sum(
        tensor.shape[-2] * tensor.shape[-1] * tensor.element_size()
        for tensor in kv_cache
    )

    assert get_tuple_bytes_per_token(kv_cache) == expected_bytes_per_token
    assert get_tuple_byte_offsets(kv_cache) == [
        (0, 1024),
        (1024, 1152),
        (1152, 9344),
        (9344, 9472),
    ]
    assert key[-1] == "raw_bytes"
    assert storage_shape == torch.Size([2, 16, expected_bytes_per_token])
    assert dtype == torch.uint8


def test_homogeneous_dsa_group_keeps_logical_dtype_storage():
    kv_cache = generate_dsa_kv_cache(
        num_blocks=2,
        device="cpu",
        num_layers=1,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        dsa_head_dim=128,
        block_size=16,
        dtype=torch.bfloat16,
    )[0]

    _, storage_shape, dtype = _get_kv_cache_group_key_and_info(kv_cache)

    assert storage_shape == torch.Size([2, 16, 512 + 64 + 128])
    assert dtype == torch.bfloat16
