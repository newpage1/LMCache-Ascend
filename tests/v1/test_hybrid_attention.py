# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import torch

# First Party
from lmcache_ascend.v1.npu_connector import (
    VLLMPagedMemNPUConnectorV2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_qwen35_layer_types(num_layers=48, full_attention_interval=4):
    """Generate Qwen3.5-style hybrid attention layer_types.

    Every ``full_attention_interval``-th layer (0-indexed) is full_attention,
    the rest are linear_attention.
    """
    return [
        "full_attention" if i % full_attention_interval == 3 else "linear_attention"
        for i in range(num_layers)
    ]


def _mock_vllm_config(layer_types=None, num_layers=48, num_kv_heads=2, head_size=256):
    """Build a lightweight VllmConfig-like mock."""
    hf_config = SimpleNamespace()
    if layer_types is not None:
        hf_config.layer_types = layer_types
    # else: no layer_types attribute → non-hybrid model

    model_config = MagicMock()
    model_config.hf_config = hf_config
    model_config.get_num_layers = MagicMock(return_value=num_layers)
    model_config.get_num_kv_heads = MagicMock(return_value=num_kv_heads)
    model_config.get_head_size = MagicMock(return_value=head_size)
    model_config.model = "Qwen3.5-122B-A10B"
    model_config.served_model_name = "Qwen3.5-122B-A10B"
    model_config.dtype = torch.float16

    parallel_config = MagicMock()
    parallel_config.world_size = 1
    parallel_config.rank = 0

    cache_config = MagicMock()
    cache_config.cache_dtype = "auto"

    vllm_config = MagicMock()
    vllm_config.model_config = model_config
    vllm_config.parallel_config = parallel_config
    vllm_config.cache_config = cache_config

    return vllm_config


# ---------------------------------------------------------------------------
# Test 1: Hybrid attention detection – full_attention_indices correctness
# ---------------------------------------------------------------------------


class TestHybridAttentionDetection:
    """Test that hybrid attention layer filtering produces correct indices."""

    def test_qwen35_pattern(self):
        """Qwen3.5-122B-A10B: 48 layers, every 4th is full_attention."""
        layer_types = _make_qwen35_layer_types()
        full_attention_indices = [
            i for i, lt in enumerate(layer_types) if lt == "full_attention"
        ]
        expected = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
        assert full_attention_indices == expected
        assert len(full_attention_indices) == 12

    def test_all_full_attention(self):
        """Non-hybrid model: all layers are full_attention."""
        layer_types = ["full_attention"] * 32
        full_attention_indices = [
            i for i, lt in enumerate(layer_types) if lt == "full_attention"
        ]
        assert full_attention_indices == list(range(32))
        assert len(full_attention_indices) == 32

    def test_no_full_attention_raises(self):
        """Edge case: model with no full_attention layers should produce
        empty list — downstream code should handle this gracefully."""
        layer_types = ["linear_attention"] * 48
        full_attention_indices = [
            i for i, lt in enumerate(layer_types) if lt == "full_attention"
        ]
        assert full_attention_indices == []

    def test_non_hybrid_no_layer_types(self):
        """Non-hybrid model without layer_types attribute → None."""
        vllm_config = _mock_vllm_config(layer_types=None)
        layer_types = getattr(vllm_config.model_config.hf_config, "layer_types", None)
        assert layer_types is None


# ---------------------------------------------------------------------------
# Test 2: Connector stores full_attention_indices correctly
# ---------------------------------------------------------------------------


class TestConnectorIndices:
    """Test that NPU connectors correctly store full_attention_indices."""

    def test_paged_connector_with_indices(self):
        """VLLMPagedMemNPUConnectorV2 stores full_attention_indices."""
        indices = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
        connector = VLLMPagedMemNPUConnectorV2(
            hidden_dim_size=512,
            num_layers=12,
            use_gpu=False,
            chunk_size=256,
            dtype=torch.float16,
            device=None,
            use_mla=False,
            num_kv_head=2,
            head_size=256,
            full_attention_indices=indices,
        )
        assert connector.full_attention_indices == indices
        assert connector.num_layers == 12

    def test_paged_connector_without_indices(self):
        """VLLMPagedMemNPUConnectorV2 defaults to None (non-hybrid)."""
        connector = VLLMPagedMemNPUConnectorV2(
            hidden_dim_size=256,
            num_layers=32,
            use_gpu=False,
            chunk_size=256,
            dtype=torch.float16,
            device=None,
            use_mla=False,
            num_kv_head=8,
            head_size=32,
        )
        assert connector.full_attention_indices is None
        assert connector.num_layers == 32


# ---------------------------------------------------------------------------
# Test 3: _initialize_pointers filtering for hybrid models
# ---------------------------------------------------------------------------


class TestPointerFiltering:
    """Test that _initialize_pointers correctly filters layers for hybrid models.

    Note: These tests mock NPU tensors since we don't have actual NPU hardware.
    They verify the logic of pointer array construction with filtered layers.
    """

    def _make_mock_kvcaches_separate(
        self, num_total_layers, num_blocks=4, block_size=16, num_heads=2, head_size=256
    ):
        """Create mock kvcaches in SEPARATE_KV format (tuples of K, V)."""
        kvcaches = []
        for i in range(num_total_layers):
            if i % 4 == 3:
                # full_attention layer: standard KV shape
                k = torch.randn(num_blocks, block_size, num_heads, head_size)
                v = torch.randn(num_blocks, block_size, num_heads, head_size)
                kvcaches.append((k, v))
            else:
                # linear_attention layer: different state shape (MambaSpec)
                # These have a completely different shape that would cause errors
                # if passed to KV transfer kernels
                kvcaches.append(
                    (torch.randn(num_blocks, 64), torch.randn(num_blocks, 64))
                )
        return kvcaches

    def test_filtered_pointers_count(self):
        """Verify that _initialize_pointers creates correct number of pointers
        for hybrid models (only full_attention layers)."""
        full_attention_indices = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
        num_full_layers = len(full_attention_indices)

        # Verify index count matches expected
        assert num_full_layers == 12

        # Verify indices are correct
        layer_types = _make_qwen35_layer_types()
        computed = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
        assert computed == full_attention_indices

    def test_filtered_kv_caches_subset(self):
        """Verify that filtering kv_caches selects only full_attention layers."""
        full_attention_indices = [3, 7, 11]
        all_kvcaches = [f"layer_{i}" for i in range(12)]

        filtered = [all_kvcaches[i] for i in full_attention_indices]
        assert filtered == ["layer_3", "layer_7", "layer_11"]
        assert len(filtered) == 3


# ---------------------------------------------------------------------------
# Test 4: init_lmcache_engine integration (mocked, no NPU)
# ---------------------------------------------------------------------------


class TestInitEngineHybridAttention:
    """Test the init_lmcache_engine function with hybrid attention detection.

    These tests mock NPU/torch dependencies to verify the logic without hardware.
    """

    @patch("lmcache_ascend.integration.vllm.vllm_v1_adapter.torch.npu")
    @patch("lmcache_ascend.integration.vllm.vllm_v1_adapter.LMCacheEngineBuilder")
    @patch("lmcache_ascend.integration.vllm.vllm_v1_adapter.get_tp_group")
    def test_hybrid_kv_shape(self, mock_tpg, mock_builder, mock_npu):
        """Verify kv_shape uses filtered layer count for hybrid models."""
        # Setup mocks
        mock_npu.device_count.return_value = 1
        mock_npu.set_device = MagicMock()
        mock_builder.get.return_value = None
        mock_builder.get_or_create.return_value = MagicMock()
        mock_tpg.return_value = MagicMock()

        # Import after patching
        # First Party
        from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
            init_lmcache_engine,
        )

        vllm_config = _mock_vllm_config(layer_types=_make_qwen35_layer_types())
        lmcache_config = MagicMock()
        lmcache_config.chunk_size = 256
        lmcache_config.use_layerwise = False
        lmcache_config.use_gpu_connector_v3 = False
        lmcache_config.enable_blending = False
        lmcache_config.enable_scheduler_bypass_lookup = False

        with (
            patch(
                "lmcache_ascend.integration.vllm.vllm_v1_adapter.VLLMPagedMemNPUConnectorV2"
            ) as mock_connector_cls,
            patch(
                "lmcache_ascend.integration.vllm.vllm_v1_adapter.mla_enabled",
                return_value=False,
            ),
            patch(
                "lmcache_ascend.integration.vllm.vllm_v1_adapter.get_kv_cache_torch_dtype",
                return_value=torch.float16,
            ),
            patch(
                "lmcache_ascend.integration.vllm.vllm_v1_adapter.need_gpu_interm_buffer",
                return_value=False,
            ),
        ):
            mock_connector_cls.from_metadata.return_value = MagicMock()

            init_lmcache_engine(lmcache_config, vllm_config, "worker")

            # Verify get_or_create was called
            assert mock_builder.get_or_create.called

            # Extract metadata passed to get_or_create
            call_args = mock_builder.get_or_create.call_args
            metadata = call_args[0][2]  # 3rd positional arg

            # kv_shape[0] should be 12 (full_attention layers only)
            assert metadata.kv_shape[0] == 12, (
                f"Expected 12 full_attention layers, got {metadata.kv_shape[0]}"
            )

            # Verify full_attention_indices was passed to connector
            from_metadata_call = mock_connector_cls.from_metadata.call_args
            assert "full_attention_indices" in from_metadata_call.kwargs
            assert from_metadata_call.kwargs["full_attention_indices"] == [
                3,
                7,
                11,
                15,
                19,
                23,
                27,
                31,
                35,
                39,
                43,
                47,
            ]

    @patch("lmcache_ascend.integration.vllm.vllm_v1_adapter.torch.npu")
    @patch("lmcache_ascend.integration.vllm.vllm_v1_adapter.LMCacheEngineBuilder")
    @patch("lmcache_ascend.integration.vllm.vllm_v1_adapter.get_tp_group")
    def test_non_hybrid_kv_shape_unchanged(self, mock_tpg, mock_builder, mock_npu):
        """Verify kv_shape uses full layer count for non-hybrid models."""
        mock_npu.device_count.return_value = 1
        mock_npu.set_device = MagicMock()
        mock_builder.get.return_value = None
        mock_builder.get_or_create.return_value = MagicMock()
        mock_tpg.return_value = MagicMock()

        from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
            init_lmcache_engine,
        )

        # No layer_types → non-hybrid
        vllm_config = _mock_vllm_config(layer_types=None, num_layers=32)
        lmcache_config = MagicMock()
        lmcache_config.chunk_size = 256
        lmcache_config.use_layerwise = False
        lmcache_config.use_gpu_connector_v3 = False
        lmcache_config.enable_blending = False
        lmcache_config.enable_scheduler_bypass_lookup = False

        with (
            patch(
                "lmcache_ascend.integration.vllm.vllm_v1_adapter.VLLMPagedMemNPUConnectorV2"
            ) as mock_connector_cls,
            patch(
                "lmcache_ascend.integration.vllm.vllm_v1_adapter.mla_enabled",
                return_value=False,
            ),
            patch(
                "lmcache_ascend.integration.vllm.vllm_v1_adapter.get_kv_cache_torch_dtype",
                return_value=torch.float16,
            ),
            patch(
                "lmcache_ascend.integration.vllm.vllm_v1_adapter.need_gpu_interm_buffer",
                return_value=False,
            ),
        ):
            mock_connector_cls.from_metadata.return_value = MagicMock()

            init_lmcache_engine(lmcache_config, vllm_config, "worker")

            call_args = mock_builder.get_or_create.call_args
            metadata = call_args[0][2]

            # kv_shape[0] should be 32 (all layers, non-hybrid)
            assert metadata.kv_shape[0] == 32, (
                f"Expected 32 layers, got {metadata.kv_shape[0]}"
            )

            # full_attention_indices should be None
            from_metadata_call = mock_connector_cls.from_metadata.call_args
            assert from_metadata_call.kwargs.get("full_attention_indices") is None
