# SPDX-License-Identifier: Apache-2.0
"""Stable cross-request prefix key for the t106 layerwise bridge.

The scheduler side derives the key when submitting a lookup and the
wrapper connector injects the same digest into every stock metadata
entry it can annotate; the worker adapter commits snapshots under it.
One digest function shared by both sides keeps the strings identical
without either side seeing the other's token ids.
"""

# Standard
import hashlib


def dsv41_prefix_key(
    model_name: str,
    cache_salt: str,
    packed_token_ids: bytes,
) -> str:
    """Digest the stable per-prompt identity into a lookup key.

    The digest covers the full packed prompt token ids plus the model
    name and the per-user cache salt — never the request id, worker id,
    or a token range — so every step of one request and every later
    request with the same prompt derives the same string (t106 item ②).

    Args:
        model_name: Served model name from the scheduler adapter.
        cache_salt: Per-user isolation salt (may be empty).
        packed_token_ids: The request's packed prompt token ids.

    Returns:
        A ``dsv41-`` prefixed hex digest identifying the prompt.
    """
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(b"|")
    digest.update(cache_salt.encode("utf-8"))
    digest.update(b"|")
    digest.update(packed_token_ids)
    return f"dsv41-{digest.hexdigest()[:32]}"
