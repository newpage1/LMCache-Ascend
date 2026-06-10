# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional
import logging

# Canonical transfer_spec keys.
TS_RECEIVER_ID = "receiver_id"
TS_SENDER_ID = "sender_id"
TS_REMOTE_BUFFER_UUIDS = "remote_buffer_uuids"
TS_REMOTE_MEM_INDEXES = "remote_mem_indexes"
TS_REMOTE_INDEXES = "remote_indexes"
TS_STREAM = "stream"
TS_TRANSFER_COMPLETION_CALLBACK = "transfer_completion_callback"

logger = logging.getLogger(__name__)


@dataclass
class TimedTransferEvent:
    """NPU event plus device-side timing metadata for deferred transfers."""

    event: Any
    start_event: Any
    end_event: Any
    callback: Optional[Callable[[float, int], None]]
    num_bytes: int
    reported: bool = False

    def report(self) -> None:
        if self.reported or self.callback is None:
            return
        try:
            self.callback(
                self.start_event.elapsed_time(self.end_event) / 1000,
                self.num_bytes,
            )
            self.reported = True
        except Exception:
            logger.warning("Transfer completion callback failed", exc_info=True)


def resolve_peer_id(transfer_spec: Mapping[str, Any]) -> str:
    """Return peer id from transfer_spec.

    Prefer receiver id and fall back to sender id for legacy/read call sites.
    """
    receiver_id = transfer_spec.get(TS_RECEIVER_ID)
    if receiver_id is not None:
        return receiver_id

    sender_id = transfer_spec.get(TS_SENDER_ID)
    if sender_id is not None:
        return sender_id

    raise KeyError(
        f"transfer_spec must contain either '{TS_RECEIVER_ID}' or '{TS_SENDER_ID}'"
    )
