"""Checkpoint IO for transformer-hosted VM models."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
from dataclasses import dataclass

import torch

from chess_machine_zero.model.machine_transformer import CMZMachineTransformer


MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class CheckpointEntry:
    checkpoint_id: str
    path: Path
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    latest_checkpoint_id: str
    entries: tuple[CheckpointEntry, ...]


def save_transformer_checkpoint(model: CMZMachineTransformer, path: str | Path) -> None:
    checkpoint = {
        "field_vocab_sizes": model.field_vocab_sizes,
        "d_model": model.d_model,
        "n_heads": model.n_heads,
        "n_layers": len(model.layers.layers),
        "d_ff": model.layers.layers[0].linear1.out_features,
        "max_seq_len": model.max_seq_len,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, Path(path))


def load_transformer_checkpoint(path: str | Path) -> CMZMachineTransformer:
    checkpoint: dict[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {"field_vocab_sizes", "d_model", "n_heads", "n_layers", "d_ff", "max_seq_len", "state_dict"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint missing required fields: {sorted(missing)}")
    model = CMZMachineTransformer(
        field_vocab_sizes=tuple(int(value) for value in checkpoint["field_vocab_sizes"]),
        d_model=int(checkpoint["d_model"]),
        n_heads=int(checkpoint["n_heads"]),
        n_layers=int(checkpoint["n_layers"]),
        d_ff=int(checkpoint["d_ff"]),
        max_seq_len=int(checkpoint["max_seq_len"]),
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


class CheckpointRegistry:
    """Filesystem registry with an explicit manifest and latest pointer."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / MANIFEST_NAME

    def save(self, model: CMZMachineTransformer, checkpoint_id: str, metrics: dict[str, float]) -> CheckpointEntry:
        if not checkpoint_id:
            raise ValueError("checkpoint_id must be nonempty")
        if any(sep in checkpoint_id for sep in ("/", "\\", ":")):
            raise ValueError("checkpoint_id must not contain path separators")
        checkpoint_path = self.root / f"{checkpoint_id}.pt"
        save_transformer_checkpoint(model, checkpoint_path)
        manifest = self.load_manifest(empty_ok=True)
        entries = [entry for entry in manifest.entries if entry.checkpoint_id != checkpoint_id]
        entry = CheckpointEntry(checkpoint_id=checkpoint_id, path=checkpoint_path, metrics=dict(metrics))
        entries.append(entry)
        self._write_manifest(CheckpointManifest(latest_checkpoint_id=checkpoint_id, entries=tuple(entries)))
        return entry

    def load_manifest(self, empty_ok: bool = False) -> CheckpointManifest:
        if not self.manifest_path.exists():
            if empty_ok:
                return CheckpointManifest(latest_checkpoint_id="", entries=())
            raise FileNotFoundError(f"checkpoint manifest not found: {self.manifest_path}")
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        entries = tuple(
            CheckpointEntry(
                checkpoint_id=str(item["checkpoint_id"]),
                path=self.root / str(item["path"]),
                metrics={str(key): float(value) for key, value in item["metrics"].items()},
            )
            for item in data["entries"]
        )
        return CheckpointManifest(latest_checkpoint_id=str(data["latest_checkpoint_id"]), entries=entries)

    def latest_path(self) -> Path:
        manifest = self.load_manifest()
        for entry in manifest.entries:
            if entry.checkpoint_id == manifest.latest_checkpoint_id:
                return entry.path
        raise ValueError(f"latest checkpoint missing from manifest: {manifest.latest_checkpoint_id}")

    def _write_manifest(self, manifest: CheckpointManifest) -> None:
        payload = {
            "latest_checkpoint_id": manifest.latest_checkpoint_id,
            "entries": [
                {
                    "checkpoint_id": entry.checkpoint_id,
                    "path": entry.path.name,
                    "metrics": entry.metrics,
                }
                for entry in manifest.entries
            ],
        }
        self.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
