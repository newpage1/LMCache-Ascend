# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum, auto
from typing import List, Tuple, Union

# Third Party
from lmcache.logging import init_logger
import torch

logger = init_logger(__name__)


class KVCacheFormat(Enum):
    """
    The storage format enumeration of KV cache is used to distinguish
    the KV cache data structures of different versions of vLLM.

    The order of csrc-backed enum values MUST match the KVCacheFormat
    definition in kernels/types.h to ensure correct interoperability
    between Python and C++ code. Python-only guard formats must fail before
    they are passed to csrc transfer kernels.
    """

    UNDEFINED = 0

    MERGED_KV = auto()
    """Merge format (eg: vLLM 0.9.2 ...)
    layer: [num_kv, num_blocks, block_size, num_heads, head_dim]
    """

    SEPARATE_KV = auto()
    """Separation format (eg: vLLM 0.11.0+ ...)
    layer: tuple: (K_tensor, V_tensor)
    - K_tensor.shape = [num_blocks, block_size, num_heads, head_dim]
    - V_tensor.shape = [num_blocks, block_size, num_heads, head_dim]

    eg: kvcaches[0] = (K, V)

    SGLang NPU Layer-Concatenated
    kvcaches = [K_all_layers, V_all_layers]
    - K_tensor.shape = [layer_nums, num_blocks, block_size, num_heads, head_dim]
    - V_tensor.shape = [layer_nums, num_blocks, block_size, num_heads, head_dim]
    """

    MLA_KV = auto()
    """MLA format for DeepSeek V2/V3 models
    layer: tuple: (k_cache, v_cache) where K and V have different dimensions
    - k_cache.shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    - v_cache.shape = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]

    This format is used when K/V shapes differ (detected automatically).
    """

    DSA_KV = auto()
    """DSA (Deep Sparse Attention) format for DeepSeek V3.2 sparse models
    layer: tuple: (k_cache, v_cache, dsa_k_cache)
    - k_cache.shape = [num_blocks, block_size, num_kv_heads, kv_lora_rank]
    - v_cache.shape = [num_blocks, block_size, num_kv_heads, qk_rope_head_dim]
    - dsa_k_cache.shape = [num_blocks, block_size, 1, 128]

    This format is used for sparse attention with lightning indexer.
    """

    DSA_C8_KV = auto()
    """Sparse C8 DSA format for DeepSeek V4 / DSA C8 models
    layer: tuple: (kv_lora_cache, k_rope_cache, indexer_k_cache, indexer_scale_cache)
    - kv_lora_cache dtype is typically bfloat16
    - k_rope_cache dtype is typically bfloat16
    - indexer_k_cache dtype is typically int8
    - indexer_scale_cache dtype is typically float16

    This format is mixed-dtype and must be transferred as a byte layout rather
    than as a single logical tensor dtype.
    """

    def is_separate_format(self) -> bool:
        return self == KVCacheFormat.SEPARATE_KV

    def is_merged_format(self) -> bool:
        return self == KVCacheFormat.MERGED_KV

    def is_mla_format(self) -> bool:
        return self == KVCacheFormat.MLA_KV

    def is_dsa_format(self) -> bool:
        return self in (KVCacheFormat.DSA_KV, KVCacheFormat.DSA_C8_KV)

    def is_tuple_format(self) -> bool:
        return self in (
            KVCacheFormat.SEPARATE_KV,
            KVCacheFormat.MLA_KV,
            KVCacheFormat.DSA_KV,
            KVCacheFormat.DSA_C8_KV,
        )

    def requires_raw_byte_transfer(self) -> bool:
        return self == KVCacheFormat.DSA_C8_KV

    def unsupported_transfer_message(self, connector_name: str) -> str:
        if self.requires_raw_byte_transfer():
            return (
                f"{connector_name} detected DeepSeek V4 / DSA Sparse C8 "
                "mixed-dtype KV cache layout. LMCache-Ascend can identify this "
                "layout, but raw-byte NPU transfer support is not implemented "
                "in this connector yet."
            )
        return f"{connector_name} does not support KV cache format: {self.name}"

    def get_kv_size(self) -> int:
        if self == KVCacheFormat.DSA_C8_KV:
            return 4
        elif self == KVCacheFormat.DSA_KV:
            return 3
        elif self in (KVCacheFormat.SEPARATE_KV, KVCacheFormat.MLA_KV):
            return 2
        elif self == KVCacheFormat.MERGED_KV:
            return 1
        return 0

    @staticmethod
    def detect(
        kvcaches: List[Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        use_mla: bool = False,
    ) -> "KVCacheFormat":
        """
        Automatically detect KV cache format based on data structure.

        Detection logic:
        1. DSA_C8_KV: tuple with 4 elements
           (kv_lora_cache, k_rope_cache, indexer_k_cache, indexer_scale_cache)
        2. DSA_KV: tuple with 3 elements (k_cache, v_cache, dsa_k_cache)
        3. MLA_KV: tuple with 2 elements where K/V shapes differ
        4. SEPARATE_KV: tuple with 2 elements where K/V shapes are same
        5. MERGED_KV: single tensor with specific shape patterns
        """
        if not kvcaches:
            return KVCacheFormat.UNDEFINED

        first_cache = kvcaches[0]

        # SGLang NPU: kvcaches = [K_tensor, V_tensor]
        if isinstance(kvcaches, list) and len(kvcaches) == 2:
            if isinstance(first_cache, torch.Tensor) and first_cache.ndim == 5:
                return KVCacheFormat.SEPARATE_KV

        if isinstance(first_cache, tuple):
            tuple_len = len(first_cache)

            # DSA_C8_KV: tuple with 4 elements for Sparse C8 DSA.
            if tuple_len == 4:
                if all(isinstance(t, torch.Tensor) for t in first_cache):
                    first_shape = first_cache[0].shape
                    if all(t.shape[:2] == first_shape[:2] for t in first_cache[1:]):
                        logger.debug(
                            "Detected DSA_C8_KV format: shapes=%s dtypes=%s",
                            [t.shape for t in first_cache],
                            [t.dtype for t in first_cache],
                        )
                        return KVCacheFormat.DSA_C8_KV

            # DSA_KV: tuple with 3 elements (k_cache, v_cache, dsa_k_cache)
            if tuple_len == 3:
                k_cache, v_cache, dsa_k_cache = first_cache
                if all(isinstance(t, torch.Tensor) for t in first_cache):
                    if k_cache.shape != v_cache.shape:
                        logger.debug(
                            f"Detected DSA_KV format: k_shape={k_cache.shape}, "
                            f"v_shape={v_cache.shape}, dsa_k_shape={dsa_k_cache.shape}"
                        )
                        return KVCacheFormat.DSA_KV

            # MLA_KV or SEPARATE_KV: tuple with 2 elements
            if tuple_len == 2:
                k_cache, v_cache = first_cache
                if isinstance(k_cache, torch.Tensor) and isinstance(
                    v_cache, torch.Tensor
                ):
                    # MLA_KV: K/V shapes differ
                    if k_cache.shape != v_cache.shape:
                        logger.debug(
                            f"Detected MLA_KV format: k_shape={k_cache.shape}, "
                            f"v_shape={v_cache.shape}"
                        )
                        return KVCacheFormat.MLA_KV
                    # SEPARATE_KV: K/V shapes are same
                    return KVCacheFormat.SEPARATE_KV

            return KVCacheFormat.SEPARATE_KV

        elif isinstance(first_cache, torch.Tensor):
            ndim = first_cache.ndim
            shape = first_cache.shape

            # Flash Attention: [2, num_blocks, block_size, num_heads, head_size]
            if ndim == 5 and shape[0] == 2:
                return KVCacheFormat.MERGED_KV

            # Flash Infer: [num_blocks, 2, block_size, num_heads, head_size]
            if ndim == 5 and shape[1] == 2:
                return KVCacheFormat.MERGED_KV

        return KVCacheFormat.UNDEFINED
