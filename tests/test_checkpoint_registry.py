from __future__ import annotations

from chess_machine_zero.model.checkpoint import CheckpointRegistry, load_transformer_checkpoint, save_transformer_checkpoint
from chess_machine_zero.model.machine_transformer import CMZMachineTransformer


def test_checkpoint_registry_manifest_tracks_versions_and_latest(tmp_path) -> None:
    registry = CheckpointRegistry(tmp_path)
    model_a = CMZMachineTransformer((18, 128, 128, 128, 16, 16, 16), d_model=16, n_heads=8, n_layers=1, d_ff=32)
    model_b = CMZMachineTransformer((18, 128, 128, 128, 16, 16, 16), d_model=16, n_heads=8, n_layers=1, d_ff=32)

    entry_a = registry.save(model_a, checkpoint_id="ckpt-a", metrics={"exact_match": 1.0})
    entry_b = registry.save(model_b, checkpoint_id="ckpt-b", metrics={"exact_match": 1.0})
    manifest = registry.load_manifest()

    assert entry_a.checkpoint_id == "ckpt-a"
    assert entry_b.checkpoint_id == "ckpt-b"
    assert manifest.latest_checkpoint_id == "ckpt-b"
    assert tuple(entry.checkpoint_id for entry in manifest.entries) == ("ckpt-a", "ckpt-b")
    assert registry.latest_path() == entry_b.path
    assert load_transformer_checkpoint(registry.latest_path()).field_vocab_sizes == model_b.field_vocab_sizes


def test_checkpoint_loader_rejects_incomplete_checkpoint(tmp_path) -> None:
    bad_path = tmp_path / "bad.pt"
    bad_path.write_bytes(b"not-a-valid-torch-checkpoint")
    try:
        load_transformer_checkpoint(bad_path)
    except Exception as exc:
        assert type(exc).__name__ in {"UnpicklingError", "RuntimeError", "ValueError", "EOFError"}
    else:
        raise AssertionError("incomplete checkpoint loaded successfully")
