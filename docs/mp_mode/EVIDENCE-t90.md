# EVIDENCE — t90（T85-02 协议 v2 + T85-03 真实 provider 集成）

任务：repair-round-2（t85 复审收窄版）。只改 `repos/LMCache/` 与
`repos/LMCache-Ascend/`；`repos/vllm-ascend/` 未触碰（T85-01 的
`ring_rows`/`restore_target` 引擎钩子归 t87）。

## 变更

### repos/LMCache — `lmcache/v1/multiprocess/layerwise_protocol.py`

- `BOUNDARY_STATE_SCHEMA_V2 = 2`（v1 常量与语义原样保留）。
- `BoundaryStateDescriptor` 追加默认字段
  `ring_length=-1 / page_offset=-1 / written_rows=()`：
  - v2：要求 `ring_length>0`、`page_offset>=0`、
    `row_indices == range(page_offset, page_offset+ring_length)`（整页身份）、
    `written_rows` 非空/唯一/页内（`0<=r<ring_length`）。
  - v1：携带任何 v2 字段 → `ValueError`（不允许走私环几何）。
  - checksum 公式不变（`schema_version` 参与），v2 摘要用 schema 2 计算。

### repos/LMCache-Ascend

- `dsv41_boundary_codec.py`：v2 全页模式现在产出**真正的 schema v2
  descriptor**（`schema_version=2` + `ring_length/page_offset/written_rows`
  + schema-2 checksum）；v1 稀疏模式字节不变。
- `dsv41_mp_adapter.py` `_restore_resource_payload`：v1 payload 拒绝之后新增
  三层身份校验（全部 fail-closed `RuntimeError`）：
  1. descriptor `schema_version != 2` 而 payload 是 v2 → 拒
     （`is schema v1 but its payload is v2`）。
  2. `ring_length / page_offset / ring_cursor / compression_phase`
     header↔descriptor 逐项相等，否则
     `boundary payload <field> <x> disagrees with descriptor <y>`。
  3. checksum 复核（`compute_boundary_checksum` 七参与写入时一致），
     不符 → `forged descriptor`。
  既有整页跨度/token/restore_target 钩子链不变。

## 测试（dsv41-layerwise 聚合，`tests/test_s1_protocol_v2_real_provider.py` 新增 8 用例）

- 协议 v2：合法字段、缺 ring_length 拒、错跨度拒（自洽 checksum）、
  written 重复/越界拒、v1 走私 v2 字段拒、v1 八字段构造回归不变。
- **真实 provider 集成（T85-03）**：按路径只读加载 t81 引擎字节
  `repos/vllm-ascend/vllm_ascend/attention/dsv41_layerwise_c2.py`，
  用其**真实** `provide_boundary_rows` + 真实 `_boundary_records` 记录簿 +
  真实 `RING_ROWS_PER_PAGE` 常量；`ring_rows`/`restore_target` 以属性协议
  附着在真实函数对象上（t87 落引擎侧，本测试注明归属）：
  - store：`set_boundary_provider` → 冷步 → 边界帧 descriptor
    `schema_version==2 / ring_length==32 / page_offset==0 /
    written_rows==(10,12)`，commit.boundaries[0] 即该 descriptor，
    无 `no ring geometry` 告警。
  - restore：脏页 3 → 整页 rebase（写入行精确回来、未写行归零、
    `captured_page=0 -> target_page=3` 日志），无 `no restore_target`。
  - 真实 fail-closed 语义：捕获 end_token=127 但步边界 128 → 真实
    provider 拒绝（引擎侧 `ends at token` 日志）→ 快照 cold-only。
  - 伪造 cursor/phase：descriptor A 配 payload B →
    `ring_cursor 3 disagrees` / `compression_phase 1 disagrees`；
    诚实配对仍可恢复。
- 旧字节对照：`git stash` 两仓（协议+codec+adapter 全回退）后新文件
  collection 即 `ImportError: cannot import name 'BOUNDARY_STATE_SCHEMA_V2'`
  （旧字节必败）；恢复后全量 **170 passed / 0 failed**（162+8）。
  既有 `test_s1_f1_f2_rebase.py` 篡改用例期望更新为新的身份守卫错误
  （`ring_length 8 disagrees with descriptor 3`，span 规则前先触发）。

## 验证

- `python3 -m py_compile`：三文件通过。
- `git -C repos/LMCache diff --check` → exit 0。
- `git -C repos/LMCache-Ascend diff --check` → exit 0。
- `repos/vllm-ascend` 工作树零改动（status 干净）。
