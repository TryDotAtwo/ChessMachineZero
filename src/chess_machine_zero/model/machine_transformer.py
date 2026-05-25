"""Causal trace-token transformer with 2D attention-head constraint."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chess_machine_zero.vm.trace_packet import TracePacket


@dataclass(frozen=True, slots=True)
class CMZTransformerOutput:
    hidden: torch.Tensor
    field_logits: tuple[torch.Tensor, ...]

    @property
    def packet_width(self) -> int:
        return len(self.field_logits)


class CMZMachineTransformer(nn.Module):
    """Minimal Milestone 3 transformer for next trace-packet prediction."""

    def __init__(
        self,
        field_vocab_sizes: tuple[int, ...],
        d_model: int = 256,
        n_heads: int = 128,
        n_layers: int = 12,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        assert d_model // n_heads == 2
        if len(field_vocab_sizes) != TracePacket.WIDTH:
            raise ValueError(f"field_vocab_sizes requires {TracePacket.WIDTH} entries")
        if any(size <= 1 for size in field_vocab_sizes):
            raise ValueError("every field vocabulary must contain at least 2 tokens")
        self.field_vocab_sizes = tuple(int(size) for size in field_vocab_sizes)
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.field_embeddings = nn.ModuleList(nn.Embedding(size, d_model) for size in self.field_vocab_sizes)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.field_heads = nn.ModuleList(nn.Linear(d_model, size) for size in self.field_vocab_sizes)

    def forward(self, packet_tokens: torch.Tensor) -> CMZTransformerOutput:
        if packet_tokens.ndim != 3 or packet_tokens.shape[-1] != TracePacket.WIDTH:
            raise ValueError(f"packet_tokens must have shape [batch, seq, {TracePacket.WIDTH}]")
        batch_size, seq_len, _ = packet_tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        packet_tokens = packet_tokens.long()
        embeddings = []
        for field_index, embedding in enumerate(self.field_embeddings):
            field_tokens = packet_tokens[:, :, field_index]
            if int(field_tokens.max()) >= self.field_vocab_sizes[field_index]:
                raise ValueError(f"field {field_index} token exceeds configured vocabulary")
            embeddings.append(embedding(field_tokens))
        hidden = torch.stack(embeddings, dim=0).sum(dim=0)
        positions = torch.arange(seq_len, device=packet_tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = hidden + self.position_embedding(positions)
        mask = causal_mask(seq_len, packet_tokens.device)
        hidden = self.layers(hidden, mask=mask)
        hidden = self.norm(hidden)
        logits = tuple(head(hidden) for head in self.field_heads)
        return CMZTransformerOutput(hidden=hidden, field_logits=logits)


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
