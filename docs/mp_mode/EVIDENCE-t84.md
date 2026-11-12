# EVIDENCE — t84: S1 transport F1/F2（整页 rebase + descriptor v2）

日期：2026-11-13 · 分支：dsv41_layerwise · 任务：t84（修复 t81 findings F1/F2）

## 范围

只改 `repos/LMCache-Ascend/`（transport 仓）。`repos/vllm-ascend/`、`repos/LMCache/`、
`ascend-perf/`、`code/` 未触碰（git 工作区 0 行）。

## F1：整页复制 + rebase 到新请求页

缺陷（t81 F1）：`_store_step_boundaries` 只编码捕获步写入的稀疏行，
`_restore_resource_payload` 的 boundary 分支把 payload 按**捕获时行号**写回——
跨请求 prefix 复用时字节落在旧物理页上，且未覆盖的 ring 尾部残留上一个请求的 slot。

修复（`dsv41_mp_adapter.py`）：

- **store 侧**：provider 协议 v2 增加 `ring_rows(resource_id)` 钩子；有几何信息时
  按 ring 页整页编码（未写行归零，plan §6.2），descriptor 行身份=整页；无钩子的
  t77/t81 旧 provider → WARNING「no ring geometry」+ 快照保持 cold-only（不伪造页，
  与"无 provider"同语义，旧行为不崩）。
- **restore 侧**：捕获页仅为 provenance（plan §6）；`restore_target(resource_id,
  boundary_token) -> (target_page, ring_rows)` 给出新请求自己的目标页；整页复制
  写入 `[target_page*R, (target_page+1)*R)`，尾部 slot 全覆盖；无钩子 / 返回 None /
  ring 不匹配 / 越界 → RuntimeError fail-closed（缺组件拒绝恢复，不允许静默冷混）。

## F2：schema v2 与 ring 完整性

缺陷（t81 F2）：`encode_boundary_rows` 只接受 2D tensor + 稀疏行，descriptor 无
ring 完整性字段。

修复（`dsv41_boundary_codec.py`）：

- payload header v2：`{"v":2, dtype, shape, ring_length, page_offset, ring_cursor,
  compression_phase, written_rows, boundary_token}`；>=2D 页张量（如 compressor 的
  `[rows, heads, 2*head]`）按尾维展平过线、header 记录形状。
- descriptor（协议 dataclass 在 repos/LMCache，出范围不可加字段）：v2 语义 =
  `row_indices` 恒为整页绝对行（长度==ring_length、连续），checksum 绑定整页身份；
  restore 侧校验「descriptor 行跨度 == payload ring 页」+「payload boundary_token ==
  descriptor token」（不一致拒绝）。
- v1 稀疏 payload：codec 层仍可解码（t81 引擎桥接的字节格式兼容，
  `test_dsv41_layerwise_c2_transport.py` 6 passed 复核）；**transport restore 路径
  明确拒绝 v1**（无 ring 几何不可 rebase，重新以 v2 存储是唯一迁移路径）。

## 验证

- 聚合套件（当前字节）：**162 passed / 0 failed**（153 既有 + 9 新增；
  `dsv41-layerwise/tests/test_s1_f1_f2_rebase.py` 9 用例 +
  `test_s1_boundary_chain.py` 12 用例更新为 v2 全页语义）。
- 新用例覆盖：v2 全页编码（含 3D 页张量/未写行归零/header 字段）、跨页行/ring 0/
  越界拒绝、v1 兼容解码、**跨请求复用**（捕获页 1 → 目标页 3，尾部 9.0 脏行被归零
  覆盖、源页不被写回）、无几何钩子 cold-only、v1 restore 拒绝、无 restore_target
  钩子/None/跨拓扑 ring 不匹配/越界全部 fail-closed、伪造 descriptor 行跨度拒绝、
  异 token payload 拒绝。
- 旧实现反证：`git -C LMCache-Ascend-dsv41_layerwise stash`（回 40741aa 字节）→
  新用例与 boundary 链用例多处 FAILED（稀疏 payload/无目标页/脏尾残留）；恢复
  rsync 后 21 passed。
- t81 引擎 UT 复核（v1 桥接兼容）：`test_dsv41_layerwise_c2_transport.py`
  --noconftest **6 passed**（对 repos/LMCache-Ascend 新字节）。
- `python3 -m py_compile` 两文件过；`git -C repos/LMCache-Ascend diff --check`
  exit 0。

## 已知边界（如实记录）

- provider 协议 v2 的 `ring_rows`/`restore_target` 钩子由引擎侧（vllm-ascend，
  t81 的 `dsv41_layerwise_c2.py` provider）在后续任务实现；在实现前 boundary
  store 走 cold-only WARNING（不崩、不伪造），boundary restore 永远 fail-closed。
- S1 lookup→retrieve→restore 的端到端命中链仍以聚合测试（真实 ZMQ server + 真实
  codec 字节）为准；c01 NPU 级验证不在 t84 验收内。
