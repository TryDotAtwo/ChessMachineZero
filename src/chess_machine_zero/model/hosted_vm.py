"""Transformer-hosted trace VM decoding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from chess_machine_zero.model.machine_transformer import CMZMachineTransformer
from chess_machine_zero.vm.trace_packet import TraceOp, TracePacket


@dataclass(frozen=True, slots=True)
class TransformerHostedVM:
    """Autoregressive trace decoder backed only by CMZMachineTransformer."""

    model: CMZMachineTransformer
    device: torch.device | None = None

    def decode_continuation(self, prompt_trace: Sequence[TracePacket], packet_count: int) -> tuple[TracePacket, ...]:
        if not prompt_trace:
            raise ValueError("prompt_trace must be nonempty")
        if packet_count < 0:
            raise ValueError("packet_count must be nonnegative")
        active_device = self.device or next(self.model.parameters()).device
        tokens = torch.tensor([packet.to_tokens() for packet in prompt_trace], dtype=torch.long, device=active_device).unsqueeze(0)
        decoded: list[TracePacket] = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(packet_count):
                if tokens.shape[1] > self.model.max_seq_len:
                    raise ValueError("decode context exceeds model max_seq_len")
                output = self.model(tokens)
                next_fields = [int(logits[0, -1].argmax().item()) for logits in output.field_logits]
                packet = TracePacket.from_tokens(next_fields)
                decoded.append(packet)
                next_tensor = torch.tensor(packet.to_tokens(), dtype=torch.long, device=active_device).view(1, 1, TracePacket.WIDTH)
                tokens = torch.cat([tokens, next_tensor], dim=1)
        return tuple(decoded)

    def decode_until_halt(self, prompt_trace: Sequence[TracePacket], max_packets: int) -> tuple[TracePacket, ...]:
        if max_packets <= 0:
            raise ValueError("max_packets must be positive")
        decoded: list[TracePacket] = []
        for packet in self.decode_continuation(prompt_trace, packet_count=max_packets):
            decoded.append(packet)
            if packet.op is TraceOp.PROGRAM_HALT:
                return tuple(decoded)
        raise ValueError("PROGRAM_HALT was not decoded within max_packets")
