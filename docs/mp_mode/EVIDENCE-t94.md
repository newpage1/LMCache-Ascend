# EVIDENCE — t94（F3：transport 多请求 boundary 提交）

任务：修复 transport `_store_step_boundaries` 只取 `entries[0]`（F3）。
t92 已让引擎按 (step, request, resource) 记录 boundary 并支持
`request_id` 点名；本任务补齐 transport 侧，使多请求步（QPS16 批
prefill）的每个可归属请求都能带自己的页/行/end token 提交与恢复。
只改 `repos/LMCache-Ascend/`。

## 变更（`lmcache_ascend/integration/vllm/dsv41_mp_adapter.py`）

1. `_store_step_boundaries` 逐请求循环：不再只取 `entries[0]`。每个
   `(request_id, start, end, blocks)` 条目独立产出 boundary spec——
   独立 prefix key（`{request_id}:layerwise`）、独立 snapshot id
   （`_snapshot_id_for(step, request)` 追加 `-r{sha256(request)[:8]}`）、
   独立 end token；provider 经新助手 `_call_boundary_provider` 以
   `request_id=` 点名调用（t92 协议），旧三参 provider 位置兼容回退。
   provider 对某请求返回 None → 该请求 cold（WARNING
   `returned no rows for resource … of request …; cold-only for this
   request, the rest of the batch is unaffected`），不拖累同批。
2. `_commit_step_snapshot` 每请求一次 commit：ops 按
   `spec.request_id` 分组，各自 `snapshot_id/key/operation_ids/
   boundaries`；拒绝时错误指明请求；step 级状态在全部请求提交后
   一起清理。
3. `_step_boundary_descriptors` 维度为 step→request→resource。
4. `_restore_resource_payload` 的 `restore_target` 钩子同样先带
   `request_id=` 点名（TypeError 回退两参旧钩子）。
5. `enqueue_step_transports` 撤销 S0 单请求 guard：t77 scheduler-MQ
   落地、t75 服务器单 owner 选举就绪、commit 已按请求分离后，
   多请求步按 `(request, resource, operation)` 确定性排序逐 spec
   dispatch；docstring 记录演进依据。这是 F3 端到端成立的必要
   条件（多请求 store 步在 hook/wait 两条 lane 都要经过 enqueue）。

## 测试（dsv41-layerwise 聚合 `tests/test_t94_multi_request_boundaries.py`，7 用例）

- 双请求步：2 个 boundary 帧（各自 key/token/snapshot id）、2 个
  commit（各自 key/id/boundaries 只含本请求 descriptor、written_rows
  各归各）。
- 4/8 并发参数化：全部请求有帧有 commit，snapshot id 全不重复。
- 同 prefix（同 end token）双请求：named provider 区分各自页。
- 部分可归属：req-b provider None → req-b cold（commit 无
  boundaries）、req-a 照常带 boundary 提交，批不受拖累，WARNING
  指名请求。
- restore 跨请求隔离：脏页后各自整页 rebase 到自己的目标页
  （`restore_target(request_id=)`），兄弟页互不污染。
- 真 ZMQ 服务器：同一注册下两个请求各自 store+commit（独立
  snapshot id/key），两次 lookup 各命中自己的快照与 matched_end
  （256/512），sibling 不互相覆盖。

既有适配：`tests/test_connector.py::test_mixed_hit_and_miss_batch`
从断言「one request per step」S0 拒绝改为断言 t94 行为（两请求 store
全部过线、pending 清空、2 个 per-request commit、key/snapshot id 各
自独立）。

## 验证

- 聚合 `177 passed / 0 failed`（170+7）。
- 旧字节对照：`git stash`（回到 ef2d992，`entries[0]`+guard）后新
  用例 **6/7 failed**（唯真 ZMQ 服务器用例通过——它钉的是服务器端
  per-key 快照语义，与 adapter 字节无关），恢复后全绿。
- `python3 -m py_compile` 通过；`git -C repos/LMCache-Ascend diff
  --check` exit 0。
- `repos/vllm-ascend`、`repos/LMCache` 工作树零改动。

## 已知边界

- 多请求步中 provider 无法归属（歧义/无记录）的请求保持 cold-only，
  不猜页（引擎侧 t92 语义）；本任务不改变 fail-closed 契约。
- snapshot id 在同 worker 内 per-request 唯一；跨 worker 由 t75 的
  `-g{generation}` owner 后缀保证。
