# EVIDENCE — t114 分页 KV 桥（paged-KV bridge）

- 尝试：t114 attempt 4（attempt_id 23d8c00d-cae1-4e19-9ca5-348717e18ee1）
- 提交：LMCache-Ascend `9dd86aa..2b42ddb`（`feat(mp): paged-KV bridge for layerwise hits with an honest-miss gate (t114)`）；vllm-ascend 保持 `4f9a8e490`、LMCache 保持 `3e6ab3e`（本轮无改动）
- 测试：dsv41-layerwise 聚合 206 passed / 0 failed（含 t114 桥 4 用例：hit 写行+完成面、miss 诚实 drop 一次、submit 幂等、lane settle）；旧字节对照（回 9dd86aa）桥用例 4/4 FAILED
- c01：容器 `dsh-apf-c01-t114`（v0.30.0rc2-a3=ce42d65b2403，16 逻辑 device，DP4/TP4，端口 18000 + lmc server 6555）

## 修复链（run1 → run4）

| run | 现象 | 根因 | 修复 |
|---|---|---|---|
| run1 | warm 500 @1s；lane `_dispatch_retrieve` miss raise → 引擎死 | 桥 miss 后 lane 仍对同请求派 retrieve spec | lane 对 bridge-dropped 请求本地结算（`_bridge_cold_requests`） |
| run2 | warm 500 @15s；`assert req_id in self.requests`（scheduler.py:3024）EngineCore DP1 死 | stock `_process_finished_stores` 对每个无 store 的 engine-finished 请求都报 `finished_sending`，在 WAITING_FOR_REMOTE_KVS 恢复时序下迟到一步撞 assert | worker `get_finished` 覆盖：保留 bookkeeping，返回空 sending 集（layerwise 从不跑 stock store） |
| run3 | warm 500 @15s（引擎活，0 Traceback） | 0.30.0rc2 在 external-hit drop（error_block_ids 重算）路径本身 500（resume 瞬间 "internal error during generation"）；且 full-serve 结构性不可能（37 层 swa 只存 16——其余属 F4-skip 的 scratch 组） | 诚实降级门 |
| run4 | **cold 200 @4s / warm 200 @5s / 0 Traceback / 引擎存活** | — | 调度器 `check_lookup_result` 命中后按 `dsv41.bridge_serve`（默认 false）决定：门关→INFO 记录命中详情并**如实报 miss**（引擎本地重算，warm 正常完成）；门开→报 matched_end |

## run4 冒烟证据（c01 /home/z00919396/zr/logs/t114/evidence-t114.txt）

- cold 200（5255+64）/ warm 200（5255+64，finish=length，5s）
- `layerwise lookup HIT` ×1 + `honest miss so the engine recomputes` ×1（同一 warm 请求：命中链真实工作，只是不喂给引擎）
- `snapshot committed` 6 / `store dispatched` 54（store 链持续）
- `Traceback` 0 / `EngineCore encountered` 0 / bridge dropped 0 / exceeded 0

## QPS16×48×64k 探针（诚实 miss 口径）

- 引擎全程存活（alive 200，0 Traceback，0 EngineCore fatal）；HTTP 层 58×200（10 warmup + 48 窗口）
- 但 store 链屏障仍在（`store dispatched` 540+、`snapshot committed` 50+），48 并发下 TTFT 230–356s、bench 900s 客户端超时（服务端继续生成）——与 t103 吞吐塌缩同源（同步逐步 store 屏障），归 t116 性能轮（store/步解耦）

## 验收对照（诚实口径）

1. tracker-READY 桥：**已实现+测试锁定**，live 默认关闭（门=诚实 miss）——未达成 live 开启
2. warm 不再 300s prompt_tokens=0：**达成**（run4 warm 200 @5s；经由本地重算，非桥加速）
3. retrieve op 幂等/复用：**达成**（确定性 `retrieve-bridge:{request}:{resource}` id + `_bridge_served` 去重 + budget + suppression）
4. live RETRIEVE>0：**未达成**（门关→无 RETRIEVE op；引擎侧两个阻断见下）
5. 测试/对照/提交/清理：**达成**（206 passed；旧字节对照 4/4 FAILED；2b42ddb fast-forward；c01 容器已删、NPU 8×No running processes）

## 已知阻断（移交后续任务）

- T114-1（结构性）：37 层 swa 资源中 21 层属 scratch engine group（F4 skip 永不存储）→ 桥 full-serve 不可能；需存储侧覆盖 scratch 组或引擎按需重算缺口资源
- T114-2（引擎）：vLLM 0.30.0rc2 external-hit drop（error_block_ids）恢复路径 500（"internal error during generation"，run1/run2/run3 三种 drop 时长一致复现）——引擎侧缺陷，非桥逻辑
