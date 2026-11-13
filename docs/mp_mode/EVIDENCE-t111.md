# t111 evidence — step_end_token hook, async store drain, cross-DP prefix reuse

Attempt: 6751ab65-1e20-4a01-a32f-a35df53ef5ce (2026-11-13)
Base: vllm-ascend dsv41_layerwise @ 41ffd7fe4 + engine hook commit (this task);
LMCache-Ascend dsv41_layerwise @ 494be7e + this task's commits;
LMCache @ 3e6ab3e (untouched).
c01: container `dsh-apf-c01-t111`, image quay.io/ascend/vllm-ascend:v0.30.0rc2,
model /opt/data/models/DeepSeek-V4.1-Flash-1017-w8a8, 16 logical devices.
Local suite: **201 passed** (dsv41-layerwise aggregate; 199 + 2 sidecar cases).

## A — engine `step_end_token` hook + transport alignment: achieved

- `dsv41_layerwise_c2.step_end_token(request_id)` exposes the newest captured
  engine step end; `install_boundary_provider` attaches it beside
  `ring_rows`/`restore_target`.
- Transport `_store_step_boundaries` aligns the boundary token to the real
  engine step end (INFO `aligned to the engine step end … store chunk end
  was …`) and keeps the F5 honest warning for refusing providers.
- run21 log: `aligned to the engine step end = 1`, `boundary store
  dispatched = 3`, `snapshot committed = 1 (ops=27)` — one all-layer
  snapshot under the prefix key (the step-seal fix, below).
- Earlier runs (t111 run4/5 fixtures) had 8 alignments per smoke pair.

## B — async store drain: implemented; QPS16 probe not run (see Blocker 3)

- Stores submit in a batch loop and settle on a bounded background drain
  (queue cap 256, oldest-settle-inline backpressure, failures counted and
  rate-limited ERROR logs, honest-miss commits).
- Local tests prove `wait_for_save` returns before the store settles,
  backpressure bounds the queue, and drain failures never crash the engine
  (`tests/test_t111_async_and_alignment.py`, 6 cases incl. real-c2 F5
  alignment).
- The throughput probe (QPS16×48×64k) was **not** run: the warm-request
  path deadlocks before prefill (Blocker 3), so a 48-request probe would
  reproduce the t106 stall rather than measure the drain.

## C — cross-DP prefix reuse: transport chain proven live; request does not finish

run21 (evidence frozen in `c01:/home/z00919396/zr/logs/t111/t111-run21-evidence.txt`):

| evidence | count |
| --- | --- |
| layerwise lookup HIT (scheduler, matched_end=3006 aligned=2816) | 2 |
| cross-identity prefix retrieve (served) | 192 |
| boundary page restored | 0 |
| Traceback / miss / reused-with-different-content | 0 / 0 / 0 |
| engine alive after both smokes | /v1/models 200 |
| cold request | 200, 3.6 s, success_total=1 |

- **RETRIEVE > 0: yes.** 192 distinct retrieves were served cross-DP from the
  storing worker's published identity (16 per-worker registration states;
  retrieve specs carry the publishing identity via msgspec replace).
- **restore > 0: yes for token-range KV** — payloads were restored into the
  registered layerwise caches by `_restore_resource_payload`.
  `boundary page restored = 0` is honest: the stock RETRIEVE ops only cover
  paged engine groups; boundary (compressor) resources are never requested
  by the retrieve path, so boundary restores cannot occur in this topology.
- **Warm request 200: not achieved.** After the scheduler HIT, the warm
  request never started prefill (`prompt_tokens_total` stayed 0 on its
  engine for 300 s, `num_requests_running/waiting` 0) while the transport
  loop kept re-dispatching fresh retrieve op ids (192 distinct) each engine
  step.

### Blocker 3 — stock external-hit handoff expects paged-KV loads

Stock `lmcache_mp_connector`'s external-hit flow expects the worker's
`async_load_kv` to copy the matched prefix into the **engine's paged KV**
and flip the request tracker to READY; only then does the vLLM scheduler
prefill the remaining tail. The layerwise worker restores bytes into the
*registered layerwise caches* (its own tensor space) and never touches the
paged KV or the stock tracker, so the request never becomes schedulable.
The 192 re-dispatches are each engine-step retry of that never-READY
request. Closing this needs a worker-side bridge from the layerwise
restore into the paged KV (or a scheduler-side bypass of the external-hit
state machine) — engine-coupled work beyond t111's two-repo scope as
scoped ("若必须改 vllm/LMCache 字段先停下报告").

### Fix chain during the run (all in this commit)

1. run2 commit timeouts → serialized fail-soft commits with per-thread
   wait budgets (engine 2 s / drain 30 s).
2. run3 commit retry storms → one-shot commits (drop step on failure).
3. run4 sibling boundary storms → deterministic boundary op ids
   (`store:boundary:{request}:{step}:{resource}`).
4. run5 spurious `completion was ignored` → idempotent-reentry DEBUG.
5. run8-13 scheduler activation → identity sidecar directory
   (`DSV41_LAYERWISE_IDENTITY_DIR`): workers publish registration
   identity; the scheduler adopts published identities for lookup fan-out
   (B3 was never live before — `_plan_hash` was test-only).
6. run10-12 prefix-key plumbing bugs → connector logger; dataclass-aware
   `_annotate_entry_configs` (LMCacheMPRequestMetadata is a dataclass, not
   msgspec); chunk-aligned `matched_end` (stock asserts divisibility).
7. run13-16 cross-DP retrieve → `_dispatch_retrieve` retries published
   identities with identity-carrying spec variants (worker public attr
   names: `plan_hash`/`layout_hash`/`registration_epoch`).
8. run17-19 (dud paths, kept as lessons): sole-storer election and group
   adoption never fire — every worker has a *unique* layout, so the
   sidecar holds 16 identities, one per worker; deterministic store ids
   from this phase remain valuable.
9. run20 exact miss → step seal: commits wait for `wait_for_pending_stores`
   to seal the step; committing on `outstanding == 0` alone raced ahead of
   later layers and produced 1-op partial snapshots (9×
   `layers.1.swa_cache NOT_COMMITTED`), every later same-id commit
   colliding with `reused with different content`.

## Verdict per acceptance

- A: pass (hook + alignment live, no `ends at token … asked for` misses).
- B: not verified (probe blocked by Blocker 3; async behavior proven by
  unit tests only).
- C: partial — RETRIEVE>0 ✓, token-range restore>0 ✓, double-200 ✗.

Task delivered as **failed** with Blocker 3 documented; successes (A, the
seal fix, cross-DP serving) are committed and reusable for the next task.
