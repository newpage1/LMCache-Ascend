# Cross-Node Backend Metrics

## Scope

These metrics cover:

- pooled remote DDR through `AscendP2PBackend`
- PD-disaggregated KV transfer through `AscendPDBackend`
- push, eager-pull, and delayed-pull modes
- CPU and NPU registered buffers

They use LMCache's existing Prometheus registry. vLLM-Ascend remains
responsible for inference-level metrics such as TTFT and request latency.

## Metrics

| Metric | Type | Meaning |
| --- | --- | --- |
| `lmcache:ascend_backend_operations` | Counter | Backend operation outcomes |
| `lmcache:ascend_backend_chunks` | Counter | Hit, missed, reused, transferred, or stored chunks |
| `lmcache:ascend_backend_bytes` | Counter | Bytes submitted to HCCL |
| `lmcache:ascend_backend_operation_duration_seconds` | Histogram | Backend operation duration |

Additional labels are intentionally low-cardinality:

| Label | Values |
| --- | --- |
| `backend` | `p2p`, `pd` |
| `backend_role` | `client`, `server`, `sender`, `receiver` |
| `mode` | `push`, `pull`, `delay_pull` |
| `device` | `cpu`, `npu` |
| `operation` | `lookup`, `allocate`, `transfer`, `transfer_device`, `store` |
| `result` | `success`, `error`, `hit`, `partial`, `miss`, `reused` |

Peer IP, KV key, lookup ID, pull ID, and request ID must not be metric labels.
Use Prometheus target labels such as `instance`, `pod`, `node`, and `cluster`
to identify machines.

## Transfer-Time Semantics

KV transfer latency has three distinct meanings:

| Layer | Measurement | Meaning |
| --- | --- | --- |
| Host completion | `operation="transfer"` | Wall-clock time from the backend calling a synchronous or asynchronous transfer API until that call confirms completion. Includes HCCL operation construction, submission, queueing, device transfer, synchronization/event polling, and host scheduling. |
| HCCL device execution | `operation="transfer_device"` | NPU timing-event elapsed time around HCCL `write_batch/read_batch` on the transport stream. Excludes Python and host polling overhead, but includes HCCL runtime, device DMA, transport-stream queueing, and network transfer. |
| Network wire time | external HCCL/NIC telemetry | Pure network time cannot be isolated accurately inside LMCache. Use msprof/HCCL profiling and RoCE/NIC counters for link throughput, packets, retries, ECN, and drops. |

For synchronous HCCL transfers, timing is read after transport-stream
synchronization. For asynchronous transfers, timing is read after the end
event reports completion.

Delayed-pull preserves the existing pipeline. Every submitted HCCL
micro-batch carries timing events, which are reported only after the load
stream has waited for the reads. No per-micro-batch synchronization is added.
Delayed-pull therefore reports `transfer_device`, but not a Host-completion
`transfer` duration.

Transferred bytes are calculated from the actual HCCL operation size:

```text
HCCL page size * number of transferred objects
```

## P2P Semantics

- P2P client lookup reports remote DDR hit and miss chunks.
- P2P server lookup reports availability in the server's local DDR.
- Push transfer is measured around server-side `async_batched_write`.
- Eager-pull transfer is measured around client-side `async_batched_read`.
- Delayed-pull device transfer is measured for each HCCL read micro-batch.

## PD Semantics

- Receiver lookup reports chunks already reusable by decode.
- Receiver `allocate` reports allocation and registration of remote targets.
- Push transfer is measured around sender-side `batched_write`.
- Eager-pull transfer is measured around receiver-side `batched_read`.
- Eager-pull `store` reports insertion after transfer completion.
- Delayed-pull device transfer is measured for each HCCL read micro-batch.

## PromQL

Remote DDR P2P chunk hit rate:

```promql
sum(rate(lmcache:ascend_backend_chunks_total{
  backend="p2p",backend_role="client",operation="lookup",result="hit"
}[5m]))
/
sum(rate(lmcache:ascend_backend_chunks_total{
  backend="p2p",backend_role="client",operation="lookup",result=~"hit|miss"
}[5m]))
```

PD decoder reuse rate:

```promql
sum(rate(lmcache:ascend_backend_chunks_total{
  backend="pd",backend_role="receiver",operation="lookup",result="reused"
}[5m]))
/
sum(rate(lmcache:ascend_backend_chunks_total{
  backend="pd",backend_role="receiver",operation="lookup",result=~"reused|miss"
}[5m]))
```

Host-completion transfer P99:

```promql
histogram_quantile(
  0.99,
  sum by (le, backend, backend_role, mode, device) (
    rate(lmcache:ascend_backend_operation_duration_seconds_bucket{
      operation="transfer",result="success"
    }[5m])
  )
)
```

HCCL device-side transfer P99:

```promql
histogram_quantile(
  0.99,
  sum by (le, backend, backend_role, mode, device) (
    rate(lmcache:ascend_backend_operation_duration_seconds_bucket{
      operation="transfer_device",result="success"
    }[5m])
  )
)
```

HCCL effective throughput:

```promql
sum by (backend, backend_role, mode, device) (
  rate(lmcache:ascend_backend_bytes_total{
    operation="transfer",result="success"
  }[5m])
)
```
