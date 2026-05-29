from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_native_docker_scripts_publish_rust_dashboard_for_windows_browser() -> None:
    run_script = (ROOT / "docker/native/run_native_container.ps1").read_text(encoding="utf-8")
    dashboard_script = (ROOT / "docker/native/start_dashboard.ps1").read_text(encoding="utf-8")

    assert '$DashboardHost = "127.0.0.1"' in run_script
    assert "$DashboardHost`:$DashboardPort`:8768" in run_script
    assert "docker run -d --name $Name --gpus all" in run_script
    assert "-p" in run_script
    assert '"${Root}:/work"' in run_script

    assert 'param(' in dashboard_script
    assert "$Name = \"cmz-native-dev\"" in dashboard_script
    assert "docker exec $Name bash -lc" in dashboard_script
    assert "docker exec -d $Name bash -lc" in dashboard_script
    assert "cd /work/native" in dashboard_script
    assert "cargo run -p cmz-dashboard -- --host 0.0.0.0 --port $Port" in dashboard_script
    assert "chess_machine_zero.dashboard.server" not in dashboard_script
    assert "tee -a $DashboardLog $StreamLog" in dashboard_script
    assert "http://127.0.0.1:$Port" in dashboard_script


def test_rust_dashboard_source_is_display_only_native_trace_consumer() -> None:
    source = (ROOT / "native/crates/cmz-dashboard/src/lib.rs").read_text(encoding="utf-8")

    assert "rust_native_dashboard" in source
    assert "policy_select_move" in source
    assert "policy_decoder_used" in source
    assert "trace_verified_legal" in source
    assert "white_readable_log" in source
    assert "black_token_trace" in source
    assert "\\\"python_hot_path\\\":false" in source
