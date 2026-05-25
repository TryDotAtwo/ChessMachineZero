"""Deterministic trace hashing."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from chess_machine_zero.vm.trace_packet import TracePacket


def trace_hash_hex(trace: Iterable[TracePacket]) -> str:
    digest = hashlib.sha256()
    for packet in trace:
        for token in packet.to_tokens():
            digest.update(int(token).to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()
