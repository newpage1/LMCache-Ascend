# t106 证据：live lookup 桥接 + 批量 store 派发（2026-11-13）

提交：`cd66ff5`（LMCache-Ascend 494be7e..cd66ff5，fast-forward push）
基线三仓：vllm-ascend 41ffd7fe4 / LMCache 3e6ab3e / LMCache-Ascend cd66ff5
测试：聚合 harness **194 passed**（185 基线 + 9 新增 `tests/test_t106_bridge.py`）；
旧字节对照（494be7e）：新用例 6 failed / 3 passed（bridge/吞吐用例必败）。

## 一、实现内容（全部在 LMCache-Ascend）

1. **批量派发**：`enqueue_step_transports` 先把本步全部 spec 逐个提交（store 预先
   `_extract_resource_payload`），再统一 settle——t103 c01 的串行「一 spec 一 RTT」
   消除。
2. **水位去重**：`_stored_token_end[(request, resource)]`，end<=水位 的重复区间
   直接跳过（debug「already stored; overlapping range not re-sent」）——t103 c01
   27 个唯一区间被重发成 ~527 ops 的解码风暴根除。
3. **落选限流**：lost-election WARNING 第 1 次 + 每 200 次输出（t103 c01 217k 条
   告警风暴），其余降 debug。
4. **跨请求 key**：`dsv41_prefix_key.py`（sha256(model|salt|packed 全 prompt)）；
   wrapper `build_connector_meta` 为每个 metadata 条目注入
   `request_configs["dsv41.prefix_key"]`；worker 端 `StepRequestView.prefix_key`
   → spec/commit 的 key（无前缀时回退 `{request_id}:layerwise`）。
5. **调度器桥接**：`DSV41LayerwiseMPSchedulerAdapter` override
   `maybe_submit_lookup_request`（stock 透传 + 并发提交 layerwise lookup）、
   `check_lookup_result`（stock 命中优先并 drop+drain；layerwise 命中→校验
   plan/layout→返回 matched_end 并**立即 release lease**——租约不释放会让同 key
   下一次 lookup 挂死 RPC handler）、`cleanup_lookup_result`（drain+release）。
   miss 保持诚实：None 即 miss，绝不伪造命中。

## 二、c01 实证（容器 dsh-apf-c01-t106，v0.30.0rc2-a3，DSV4.1-Flash-1017-w8a8）

- 单请求 200：cold 6.9s / 同 prompt warm 3.2s（HTTP 均 200，无 Traceback、无
  C2Error、engine 全程存活）。
- store 链：submitted 2160 / dispatched 582 / already-stored 去重 305 /
  snapshot committed 47 / lost-election 16（限流生效）。
- **吞吐探针（64k×QPS16×48+warmup10）未达标（如实报告）**：15 分钟 0 完成、
  success_total=0，running=1/引擎，卡住请求的 prefill 停在 [32768,40960) 窗口
  （64k 的第 4/8 个 8192-token 调度步）。逐窗 store（每窗 24 资源×4 TP worker
  的同步提交/结算 RPC 屏障）跟不上 prefill 步进——每窗 ~90s 串行。48 请求估
  ~20 小时，探针中止（bench 进程已 kill）。**吞吐验收（③）未达成**：需要
  store 下线异步化/与调度步解耦的架构决策，超出本次「批量+去重+限流」的修复面。
- **live RETRIEVE>0 / boundary restore>0 未达成（如实报告）**：36 次 provider
  拒绝全部为 `ends at token 3529 but the transport asked for 3328`——引擎步
  终点 3529（全 prompt）vs store chunk 终点 3328（13×256），t109 item⑤ 记录
  的引擎 `step_end_token` 钩子缺失所致；无 boundary → 快照 boundaries=() →
  server lookup 诚实 miss（is_valid=False）→ 无 RETRIEVE op → 无 restore。
  桥接本身按设计工作（未命中不伪造）。修复需 vllm-ascend 侧
  `provide_boundary_rows` 暴露步终点（约一函数钩子），超出本任务「只改
  LMCache-Ascend」边界——按 captain 指令停下报告。

## 三、产物

- serve 日志：c01 `/home/z00919396/zr/logs/t106/serve-t106.log`（QPS16 探针段
  tokens=[32768,40960) 为最后派发窗）
- bridge 冒烟：`logs/t106/bridge3-{cold,warm}.json`；探针编排
  `logs/t106/bench-t106-probe.log`
- lmcache server 日志（root 写入）：`logs/t71/lmc-server.log`（实例 id 沿用
  start-deps 默认 t71）
- 测试：dsv41-layerwise `tests/test_t106_bridge.py`（9 用例：水位去重/限流/
  prefix key 流转/桥接 hit·stock-priority·pending·miss·foreign·cleanup）
- 清理：容器已 stop+rm，NPU 8×No running processes，无 dsh-apf 残留
