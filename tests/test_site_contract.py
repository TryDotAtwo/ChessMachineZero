import json
from pathlib import Path

from vm_compiler.compiler import build_position_reconstruction_artifact


ROOT = Path(__file__).parents[1]
HTML = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
JS = (ROOT / "site" / "app.js").read_text(encoding="utf-8")


def test_site_describes_the_current_executable_artifact_exactly():
    for exact_contract in (
        "[1 × 2048 × 128]",
        "[1 × 3 × 128]",
        "[1 × 2045 × 128]",
        "[1 × 64 × 128]",
        "Generic ops × 45",
        "2064 events",
        "host chess logic",
        "none",
    ):
        assert exact_contract in HTML


def test_site_does_not_claim_unimplemented_recurrent_stages():
    assert "LEGAL_SET</code> and terminal status remain the next artifact stages" in HTML
    assert "Browser view replays an exact fixture" in HTML
    assert "policy head" not in HTML.lower()
    assert "tree search" not in HTML.lower()


def test_numeric_history_fixture_uses_three_native_tokens_per_move():
    assert "move: [52, 54, 96]" in JS
    assert "move: [57, 55, 102]" in JS
    assert "move: [71, 63, 97]" in JS
    assert "400 plies" in JS


def test_pages_deployment_publishes_only_the_static_site():
    workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert "path: site" in workflow
    assert "actions/deploy-pages@v4" in workflow


def test_site_trace_is_an_exact_export_of_the_executable_ssa_graph():
    trace_path = ROOT / "site" / "artifact_trace.json"
    assert trace_path.exists(), "site must publish the generated artifact trace"

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    artifact = build_position_reconstruction_artifact()
    expected_operations = [
        {
            "index": index,
            "opcode": operation.opcode.name,
            "inputs": list(operation.inputs),
            "outputs": list(operation.outputs),
            "attributes": list(operation.attributes),
        }
        for index, operation in enumerate(artifact.operations, start=1)
    ]

    assert trace["artifact"] == "position_latest_event_v1"
    assert trace["operation_count"] == 45 == len(artifact.operations)
    assert trace["operations"] == expected_operations
    assert trace["runtime_opcodes"] == sorted({op["opcode"] for op in expected_operations})
    assert trace["final_output"] == {"value": 45, "shape": ["B", 64, 128]}


def test_site_exposes_matrix_equations_and_all_raw_ssa_operations():
    required_proof = (
        "Executable tensor trace",
        "Q × Kᵀ",
        "[B,64,2064]",
        "A × V",
        "[B,64,128]",
        "45 serialized SSA operations",
        'id="ssaOperations"',
        'id="runtimeOpcodes"',
    )
    for marker in required_proof:
        assert marker in HTML

    assert "artifact_trace.json" in JS


def test_site_maps_every_runtime_opcode_to_a_generic_tensor_primitive():
    primitive_equations = (
        "X[:, start:end:stride, :]",
        "frozen.unsqueeze(0).expand(B, …)",
        "X × W",
        "X + Wposition",
        "X + Y",
        "concat(X₁ … Xₙ, rows)",
        "one_hot(argmax(X))",
        "hardmax(Q × Kᵀ) × V",
    )
    for equation in primitive_equations:
        assert equation in HTML
