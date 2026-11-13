# EVIDENCE — t103: transport F4/F5（覆盖缺口不杀 serving + boundary 步终点对齐）

任务：t103（LMCache-Ascend transport 修复，基于引擎 41ffd7fe4）
仓库：`repos/LMCache-Ascend` @ `HEAD`（`dsv41_mp_adapter.py`，单文件 + 本证据）
基线：`9fea0dc`（t94 后字节）；引擎侧 `repos/vllm-ascend` @ `41ffd7fe4`（t97，本任务不改动）

## 变更

### F4：未覆盖的 engine group 不再杀 worker
1. hook lane（`submit_layer_store`）：store spec 构建前以 `_store_covers_resource` 过滤，
   未覆盖组 debug 日志 `store skip: engine group … not covered by the N block group(s) recorded …`，跳过不 raise。
2. retrieve lane（`schedule_step_retrieves`）：specs 从 covered 资源构建（空 blocks 不再生成必炸的 token-range spec）。
3. `_blocks_for_engine_group`：覆盖缺失分支 raise→WARNING「do not cover engine group … treating that group as empty」+`()`；
   「outside the registered engine groups」保留 raise（配置错误仍 fail-closed）。

### F5：boundary 只骑步终点
1. 可选引擎钩子 `provider.step_end_token(request_id)`：存在且 ≠ store chunk 终点时，
   boundary_token 对齐引擎步终点并 INFO `boundary token … aligned to the engine step end`。
   （引擎 41ffd7fe4 尚未提供该钩子——对齐分支就绪、生产路径走 2。）
2. provider 拒绝（无该终点行）→专用 F5 WARNING `boundary stays cold … did not accept store end …
   only rides a store whose end equals the step end; partial-chunk tails stay cold by design`，
   该请求 boundary 不出 worker，同批其他请求不受影响。

## 本地验证

- 聚合测试：183 passed（177 基线 + t103 5 条 + 受控 retrieve 整页恢复 1 条）。
- 旧字节对照：`git stash` 回 `9fea0dc` 后 t103 新用例 3/5 FAILED（gap raise / 对齐 / 专用警告），恢复后全绿。
- 引擎 UT（41ffd7fe4，`tests/ut/attention/test_dsv41_layerwise_*.py --noconftest`）：46 passed（t97 复验保留）。
- 受控 retrieve（acceptance ③）：`test_aligned_boundary_restores_full_page_bytes` —— F5 对齐后的 boundary 帧
  `_restore_resource_payload` 恢复到引擎目标页，写入行字节精确、未写行归零（torch.equal）。

## c01 实证（QPS16 × 48 请求，18000 原口径）

容器 `dsh-apf-c01-t103`（`ce42d65b2403`，dsh-apf 前缀，全套 driver/模型挂载）；
lmcache MP server `127.0.0.1:6555`（chunk 256 / L1 100GB / LRU）；serve 就绪 01:26Z（run2）。
单请求冒烟：chat 200 / 5.9s（reasoning 正常）——t97「首 forward 不再 DSV41LayerwiseC2Error」复验保留。

探针：`bench.py --shared-tokens 58982 --tail-tokens 6554 --qps 16 --max-requests 48 --warmup 10 --base-url http://127.0.0.1:18000`
（首跑误带 `/v1` 后缀 404 ×48 不计；正式跑 t103-probe2）

**不崩证据（acceptance ③「不崩+boundary dispatched>0」半句）**：
- 引擎全程存活：探针期间与结束后 `/v1/models`=200、`/metrics` 持续可采（40 分钟洪峰无重启、无 ApiServer died）。
- `Traceback` = 0；`DSV41LayerwiseC2Error` / `without a bound step view` = 0；`recorded block groups … do not cover` 崩溃 = 0（F4 生效，日志中该措辞只可能以 WARNING 出现）。
- `boundary store dispatched over the wire` = **96**（>0）；`store dispatched over the wire`（token-range）= **2112**；
  `store snapshot committed` = **133**。
- F5 诚实语义在线可见（t109 对完整证据快照 244,054 行重数修正）：
  `boundary stays cold` = `did not accept store end` = **348**；`aligned to the engine step end` = **0**；
  `boundary page restored` = **0**；单 owner 选举 `lost the single-owner election` WARNING = 217,401 条正常工作。
- 证据快照：c01 `logs/t103/serve-t103-run2.evidence-snapshot.log`（244,054 行，ccl 02:19 止）+ `probe-t103-2.log` + `summary-t103-probe2.json`。

**t109 勘误**：初版证据的 1920/117/294/195,938 为部分窗口计数；以本节完整快照重数（2112/133/348/217,401）为准。

**acceptance ③「live restore 闭环」未达成（t109 明确降级，不冒充）**：
`boundary page restored` = 0 —— live 链路没有发生任何 restore。原因（架构性，非本轮 F4/F5 缺陷）：
stock scheduler 的 lookup 走标准 LMCache 存储，而 layerwise 字节存在 transfer 模块自有快照里、
且快照 key 按请求隔离，跨请求复用无桥接 ⇒ 生产路径无 RETRIEVE op ⇒ restore 永不触发。
`dispatched=96` 只是 store 半链证据，**不得当作 restore 闭环**。live lookup 桥接（scheduler
`maybe_submit_lookup_request` 查 layerwise 快照 + prefix/token-hash 跨请求 key）归 **t106**（captain m07106 裁决）。

**引擎 step_end_token 钩子未提供（t109 item ⑤降级口径）**：引擎侧（repos/vllm-ascend @41ffd7fe4）未实现
`provider.step_end_token(request_id)`；transport 已支持该钩子（有则对齐+INFO，无则 provider 仲裁），
当前所有 boundary 走安全 cold（348 条），`aligned=0`。引擎钩子实现归引擎侧后续任务，本报告按
captain 口径分别报告 aligned 与 cold 计数。

**如实披露（超出本任务验收句、影响 t106/t108）**：48/48 请求在客户端超时（bench req-timeout）；
服务端 35 分钟内 0 完成（`request_success_total` 恒 0），decode 前进 <1 token/90s——S0/S1 的
**同步逐步 store 链在 48 并发下吞吐塌缩**（每步 24 资源 × 提交/派发/等待 RPC 串行；27 个去重 token 区间被
重复提交，op 计数爬至 ~527 = 跨 decode 步重提交 + 兄弟落选风暴 217k 条）。引擎不崩、单请求正常（200/5.9s），
负载吞吐属 t106（live lookup 桥接）与 t108（性能）范围，本报告不冒充收益。

## t109 修复（t104 findings，本节为增补）

- **item ②**：`schedule_step_retrieves` 的 `begin_step_admission` 改为只要求 **covered** 资源
  （`[r.resource_id for r in covered]`）。旧行为把未覆盖组也列为 required，layer runtime 会以
  `does not cover every required resource` 拒绝窗口（带 RETRIEVE 的缺组步直接 raise）。
  回归：`test_retrieve_lane_skips_uncovered_group_and_admits`（双 engine group，旧字节必败）。
- **item ③**：`_store_covers_resource` 区分「已注册组本次未覆盖」（False=跳过）与「组不在注册表」
  （`RuntimeError`，fail-closed，不静默跳过）。回归：`test_store_covers_resource_fails_closed_on_unregistered_group`（旧字节必败）。
- **item ④**：删除恒真断言；hook/wait-time 双 lane 的跳过日志统一 `store skip:` 前缀、retrieve lane
  `retrieve skip:` 前缀；断言未覆盖资源不进 wire spec。
- **item ⑤/①/⑥**：见上「t109 勘误」「未达成」「降级口径」三节——证据数字修正（dispatched=96 /
  restored=0 / cold=348 / aligned=0），live restore 未达成明确移交 t106，引擎钩子降级口径分别报告。
- 本地验证：聚合 **185 passed**（183 基线 + 2 新增）；旧字节对照 2 个新增用例失败（证明其为回归锁）。

## 只读边界

- 本任务未改动 `repos/LMCache/`、`repos/vllm-ascend/`（引擎 41ffd7fe4 为 t97 交付字节，直接消费）。
- 未触碰他人容器（t97 容器由其 owner 释放后我才启动）；结束后 `dsh-apf-c01-t103` 已 stop+rm，
  `npu-smi` 8×No running processes，`docker ps -a` 无 dsh-apf 残留。
