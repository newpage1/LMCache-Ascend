# DeepSeek V4 Adaptation Workflow

This workflow is for completing LMCache-Ascend adaptation and validation after
vLLM-Ascend has already added DeepSeek V4 Pro/Flash support.

## Goal

- Adapt LMCache-Ascend for DeepSeek V4 Pro/Flash KV layouts.
- Validate Python metadata, build, connector guardrails, and 910C runtime
  behavior.
- Push the adaptation branch to a personal fork before opening an upstream PR.

## Ground Rules

- Do not push directly to upstream `LMCache/LMCache-Ascend`.
- Do not configure global credentials on shared validation machines.
- Keep validation work under a private working directory on shared machines.
- Do not start public API services unless explicitly required for model smoke
  tests.
- Prefer mounted local source and local wheelhouse over downloading source code
  on shared machines.

## Phase 1: Support Gate

1. Confirm `vllm-ascend` supports the target DeepSeek V4 variant and hardware.
2. Use official vLLM-Ascend docs and local adaptation code as evidence.
3. Confirm the DeepSeek V4 KV layout from upstream vLLM and runtime
   `register_kv_caches(...)`, not from model name alone.
4. Stop if vLLM-Ascend does not support the target model or attention path.

Evidence to collect:

```bash
git -C /path/to/vllm-ascend show v0.18.0:vllm_ascend/patch/platform/patch_kv_cache_interface.py
git -C /path/to/vllm show HEAD:vllm/model_executor/layers/deepseek_v4_attention.py
```

## Phase 2: Implementation

Keep Python-side detection and unsupported-transfer guardrails separate from
kernel support.

Required areas:

- `lmcache_ascend/v1/kv_format.py`
- `lmcache_ascend/v1/kv_layer_groups.py`
- `lmcache_ascend/v1/npu_connector/npu_connectors.py`
- `lmcache_ascend/mindspore/v1/npu_connector.py`
- `tests/v1/utils.py`
- targeted DeepSeek V4 tests under `tests/v1/`

For Sparse C8, the current safe stage is:

- detect 4-tensor DSA Sparse C8 layout.
- preserve mixed dtype metadata as raw-byte storage.
- use the VLLM paged NPU connector raw-byte fallback for functional validation.
- keep non-paged and 310P paths fail-fast until they have dedicated validation.

Do not add a Python enum value to the C++ ABI until a high-performance raw-byte
kernel contract and offsets are implemented. The fallback path uses PyTorch
`index_select` / `index_copy_` and does not call the existing C++ transfer
kernel with `DSA_C8_KV`.

## Phase 3: Local Validation

Run syntax and whitespace checks first:

```bash
python3 -c "import ast, pathlib; files=[
    'lmcache_ascend/v1/kv_format.py',
    'lmcache_ascend/v1/kv_layer_groups.py',
    'lmcache_ascend/v1/npu_connector/npu_connectors.py',
    'lmcache_ascend/mindspore/v1/npu_connector.py',
    'tests/v1/utils.py',
    'tests/v1/test_deepseek_v4_kv_format.py',
]; [ast.parse(pathlib.Path(f).read_text()) for f in files]; print('syntax ok')"

git diff --check
```

If `torch` and `pytest` are available:

```bash
python3 -m pytest tests/v1/test_deepseek_v4_kv_format.py
```

## Phase 4: 910C Source-Mounted Validation

Prepare source locally, then copy or mount it to the shared 910C machine:

```bash
/path/to/lmcache-work/LMCache
/path/to/lmcache-work/LMCache-Ascend
/path/to/lmcache-work/wheelhouse
```

Start a one-shot or temporary vLLM-Ascend container with local source mounted.
Use a private working directory and avoid global account configuration:

```bash
docker run -it --rm \
  --shm-size=200g --privileged --net=host --runtime=ascend \
  -v /path/to/lmcache-work:/work \
  --entrypoint /bin/bash \
  quay.io/ascend/vllm-ascend:v0.18.0
```

Install dependencies from a local wheelhouse when possible:

```bash
pip install --no-index --find-links=/work/wheelhouse \
  -r /work/LMCache/requirements/common.txt
pip install --no-index --find-links=/work/wheelhouse \
  -r /work/LMCache/requirements/test.txt
```

Install LMCache from mounted source:

```bash
cd /work/LMCache
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.4.3 \
NO_CUDA_EXT=1 pip install -v --no-build-isolation -e .
```

Install LMCache-Ascend from mounted source:

```bash
cd /work/LMCache-Ascend
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_ASCEND=0.4.2 \
SOC_VERSION=Ascend910_9382 \
pip install -v --no-build-isolation -e .
```

Run targeted checks:

```bash
python3 -m pytest tests/v1/test_deepseek_v4_kv_format.py
python3 -c "import lmcache_ascend; print(lmcache_ascend.__version__)"
```

If package-level imports are blocked by missing LMCache dependencies, run a
direct-file byte-layout check first:

```bash
PYTHONPATH=/work/LMCache:/work/LMCache-Ascend python3 - <<'PY'
import torch
from lmcache_ascend.v1.kv_format import (
    KVCacheFormat,
    get_tuple_byte_offsets,
    get_tuple_bytes_per_token,
)

kv_cache = (
    torch.rand([2, 16, 1, 512], dtype=torch.bfloat16),
    torch.rand([2, 16, 1, 64], dtype=torch.bfloat16),
    torch.randint(-128, 127, [2, 16, 64, 128], dtype=torch.int8),
    torch.rand([2, 16, 64, 1], dtype=torch.float16),
)
assert KVCacheFormat.detect([kv_cache]) == KVCacheFormat.DSA_C8_KV
assert get_tuple_bytes_per_token(kv_cache) == 9472
assert get_tuple_byte_offsets(kv_cache) == [
    (0, 1024),
    (1024, 1152),
    (1152, 9344),
    (9344, 9472),
]
PY
```

## Phase 5: Runtime Layout Capture

Before implementing raw-byte kernels, capture runtime layout for both Pro and
Flash:

- `kv_cache_config.kv_cache_groups`
- layer indices per group
- tuple length per layer
- tensor shape, dtype, and device per tuple element
- whether SWA-only layers are absent from the main LMCache-visible KV cache
- whether the scale tensor shape is `[num_blocks, block_size, num_heads, 1]`

Save the layout output in a private log first. Only publish sanitized summaries.

## Phase 6: Commit And Push

Use a branch named:

```bash
adapt/deepseek-v4-pro-flash
```

Commit after local and 910C checks:

```bash
git status --short
git add README.md docs/deepseek_v4_adaptation_workflow.md \
  lmcache_ascend/v1/kv_format.py \
  lmcache_ascend/v1/kv_layer_groups.py \
  lmcache_ascend/v1/npu_connector/npu_connectors.py \
  lmcache_ascend/mindspore/v1/npu_connector.py \
  tests/v1/utils.py tests/v1/test_deepseek_v4_kv_format.py
git commit -m "Adapt DeepSeek V4 KV layout detection"
```

Push only to a personal fork:

```bash
git remote add fork git@github.com:<user>/LMCache-Ascend.git
git push -u fork adapt/deepseek-v4-pro-flash
```

Open the upstream PR from the personal fork after the validation summary is
complete.
