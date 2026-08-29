import json
import subprocess
from pathlib import Path

from vm_compiler.compiler import build_position_reconstruction_artifact
from vm_compiler.site_trace import build_numeric_site_trace


ROOT = Path(__file__).parents[1]
HTML = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
JS = (ROOT / "site" / "app.js").read_text(encoding="utf-8")
CSS = (ROOT / "site" / "styles.css").read_text(encoding="utf-8")


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


def test_site_publishes_exact_numeric_matrices_from_a_real_execution():
    numeric_path = ROOT / "site" / "numeric_trace.json"
    assert numeric_path.exists(), "site must publish a numerical execution trace"

    published = json.loads(numeric_path.read_text(encoding="utf-8"))
    expected = build_numeric_site_trace()

    assert published == expected
    assert published["fixture"]["moves"] == [[52, 54, 96], [47, 45, 102], [54, 45, 96]]
    assert len(published["operations"]) == 45
    assert published["values"]["v0"]["shape"] == [1, 2048, 128]
    assert published["values"]["v45"]["shape"] == [1, 64, 128]
    assert published["operations"][-1]["equation"] == "v45 = hardmax(v41 @ transpose(v40)) @ v44"
    assert published["operations"][-1]["output"] == "v45"
    assert published["operations"][-1]["derived"] == ["op45_scores", "op45_attention"]
    assert published["derived"]["op45_scores"]["shape"] == [1, 64, 2064]
    assert published["derived"]["op45_attention"]["shape"] == [1, 64, 2064]
    assert published["derived"]["op45_attention"]["nnz"] == 64
    assert published["tensors"]["w0"]["name"] == "latest_square_key_projection"


def test_site_renders_numeric_matrix_inspector_not_shape_only_cards():
    for marker in (
        'id="numericOperations"',
        'id="matrixFlow"',
        'id="cellExplanation"',
    ):
        assert marker in HTML

    assert "--matrix-ink: #101416" in CSS


def test_matrix_inspector_explains_real_row_by_column_products_for_every_opcode():
    script = r"""
const inspector = require('./site/matrix_inspector.js');
const trace = require('./site/numeric_trace.json');
const left = {shape:[1,2,4], entries:[[0,0,1,3],[0,0,3,2]]};
const weight = {shape:[4,2], entries:[[1,0,5],[3,0,-1]]};
const dot = inspector.dotProduct(left, weight, 0, 0, 0);
const concat = inspector.concatSource([{shape:[1,2,4]},{shape:[1,3,4]},{shape:[1,1,4]}], 4);
const winner = inspector.hardmaxWinner({shape:[1,2,4],entries:[[0,1,2,1]]}, 0, 1);
const models = [...new Set(trace.operations.map(op => op.opcode))].map(inspector.operationModel);
console.log(JSON.stringify({
  total: dot.total,
  terms: dot.terms,
  ruToken: inspector.tokenMeaning(52, 'ru'),
  enToken: inspector.tokenMeaning(96, 'en'),
  modelCount: models.filter(Boolean).length,
  opcodeCount: new Set(trace.operations.map(op => op.opcode)).size
  ,concat,
  winner
}));
"""
    completed = subprocess.run(
        ["node", "-e", script],
        cwd=ROOT,
        check=True,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    result = json.loads(completed.stdout)

    assert result == {
        "total": 13,
        "terms": [
            {"k": 0, "left": 0, "right": 0, "product": 0},
            {"k": 1, "left": 3, "right": 5, "product": 15},
            {"k": 2, "left": 0, "right": 0, "product": 0},
            {"k": 3, "left": 2, "right": -1, "product": -2},
        ],
        "ruToken": "клетка e2",
        "enToken": "white pawn",
        "modelCount": 8,
        "opcodeCount": 8,
        "concat": {"input": 1, "row": 2},
        "winner": 2,
    }


def test_matrix_inspector_has_bilingual_cell_and_operation_controls():
    required = (
        'id="languageRu"',
        'id="languageEn"',
        'id="operationTitle"',
        'id="matrixFlow"',
        'id="cellExplanation"',
        'id="outputRow"',
        'id="outputColumn"',
    )
    for marker in required:
        assert marker in HTML
    assert '<script defer src="./matrix_inspector.js"></script>' in HTML


def test_matrix_inspector_presents_tensor_meaning_instead_of_ssa_names():
    script = r"""
const inspector = require('./site/matrix_inspector.js');
const cases = ['ROW_ROUTE', 'TOKEN_PROJECT', 'POSITION_ADD', 'RESIDUAL_ADD', 'FROZEN_EXPAND', 'ROW_CONCAT', 'HARDMAX_STE'];
console.log(JSON.stringify({
  modes: Object.fromEntries(cases.map(opcode => [opcode, inspector.highlightMode(opcode)])),
  hardmaxRu: inspector.operationPresentation('HARDMAX_STE', 'ru'),
  expandRu: inspector.operationPresentation('FROZEN_EXPAND', 'ru'),
  projectEn: inspector.operationPresentation('TOKEN_PROJECT', 'en'),
  hardmaxFlowClass: inspector.flowClass('HARDMAX_STE'),
  matmulFlowClass: inspector.flowClass('TOKEN_PROJECT'),
  optionRu: inspector.operationOptionText(11, 'Hardmax: выбор максимума в каждой строке'),
  cellRu: inspector.outputCellText('ru', 2, 3, 0.5),
  routedRow: inspector.routeSourceRow('v1 = route(v0, rows=3:1203:3)', 2),
  routeInputPurpose: inspector.tensorPurpose('ROW_ROUTE', 'input', 'ru'),
  addFrozenPurpose: inspector.tensorPurpose('POSITION_ADD', 'frozen', 'ru'),
}));
"""
    completed = subprocess.run(
        ["node", "-e", script],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["hardmaxFlowClass"] == "matrix-flow word-operator-flow"
    assert result["matmulFlowClass"] == "matrix-flow"
    assert result["optionRu"] == "11 · Hardmax: выбор максимума в каждой строке"
    assert result["cellRu"] == "Выход[batch 0, строка 2, столбец 3] = 0.5"
    assert result["routedRow"] == 9
    assert result["routeInputPurpose"] == "История ходов и служебный контекст: строка = событие, столбец = токен"
    assert result["addFrozenPurpose"] == "Frozen-матрица из артефакта: неизменяемое смещение для каждой строки и каждого признака"
    assert result["modes"] == {
        "ROW_ROUTE": "cell",
        "TOKEN_PROJECT": "row-column",
        "POSITION_ADD": "cell",
        "RESIDUAL_ADD": "cell",
        "FROZEN_EXPAND": "cell",
        "ROW_CONCAT": "cell",
        "HARDMAX_STE": "row-argmax",
    }
    assert result["hardmaxRu"] == {
        "operator": "ARGMAX ПО СТРОКЕ",
        "input": "Оценки вариантов в каждой строке",
        "output": "One-hot: 1 только у максимального варианта",
        "equation": "Для каждой строки: найти максимальную оценку → поставить 1 в её столбце, остальные значения обнулить",
    }
    assert result["expandRu"] == {
        "operator": "КОПИРОВАНИЕ В BATCH",
        "input": "Frozen-матрица исходных клеток: строка = клетка, столбец = токен содержимого",
        "output": "Та же матрица для каждой партии в batch",
        "equation": "Frozen-матрица не вычисляется из входа: она хранится в артефакте и копируется в batch без изменения значений",
    }
    assert result["projectEn"] == {
        "operator": "MATRIX MULTIPLICATION",
        "input": "Input activations: row = processed item, column = input feature",
        "output": "Projected activations: row = the same item, column = output feature",
        "equation": "Each output cell is the dot product of one input row and one frozen-weight column",
    }
