import json
import re
import subprocess
from pathlib import Path

from vm_compiler.compiler import build_position_reconstruction_artifact
from vm_compiler.site_trace import asset_version, build_numeric_site_trace, refresh_site_asset_versions
from vm_compiler.site_semantics import build_site_semantics


ROOT = Path(__file__).parents[1]
HTML = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
JS = (ROOT / "site" / "app.js").read_text(encoding="utf-8")
CSS = (ROOT / "site" / "styles.css").read_text(encoding="utf-8")


def test_static_asset_versions_are_content_derived_and_resolve(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    assets = {
        "styles.css": "body { color: lime; }\r\n",
        "i18n.js": "window.i18n = 1;\n",
        "trace_model.js": "window.model = 1;\n",
        "app.js": "window.app = 1;\n",
        "matrix_inspector.js": "window.inspector = 1;\n",
        "numeric_trace.json": '{"fixture":"exact"}\n',
    }
    for name, contents in assets.items():
        (site / name).write_text(contents, encoding="utf-8", newline="")
    (site / "index.html").write_text(
        '<link rel="stylesheet" href="./styles.css"><script src="./i18n.js"></script><script src="./trace_model.js"></script><script src="./app.js"></script><script src="./matrix_inspector.js"></script><a id="downloadTrace" href="./numeric_trace.json">JSON</a>',
        encoding="utf-8",
    )

    refresh_site_asset_versions(site)
    first = (site / "index.html").read_text(encoding="utf-8")
    versioned = dict(re.findall(r'\./([^?" ]+)\?v=([0-9a-f]{16})', first))
    assert set(versioned) == set(assets)
    assert all((site / name).is_file() for name in versioned)
    assert versioned["styles.css"] == asset_version(site / "styles.css")
    assert versioned["numeric_trace.json"] == asset_version(site / "numeric_trace.json")

    (site / "app.js").write_text("window.app = 2;\n", encoding="utf-8")
    refresh_site_asset_versions(site)
    second = (site / "index.html").read_text(encoding="utf-8")
    changed = dict(re.findall(r'\./([^?" ]+)\?v=([0-9a-f]{16})', second))
    assert changed["app.js"] != versioned["app.js"]
    assert {name: changed[name] for name in assets if name != "app.js"} == {name: versioned[name] for name in assets if name != "app.js"}
    lf_copy = tmp_path / "styles-lf.css"
    lf_copy.write_text(assets["styles.css"].replace("\r\n", "\n"), encoding="utf-8")
    assert asset_version(site / "styles.css") == asset_version(lf_copy)


def test_published_asset_versions_match_current_content_and_app_fetches_download_link():
    published = dict(re.findall(r'\./([^?" ]+)\?v=([0-9a-f]{16})', HTML))
    expected = {"styles.css", "i18n.js", "trace_model.js", "app.js", "matrix_inspector.js", "numeric_trace.json"}
    assert set(published) == expected
    for name, version in published.items():
        path = ROOT / "site" / name
        assert path.is_file()
        assert version == asset_version(path)

    script = r"""
let requested;
global.document = {getElementById(id) { return id === 'downloadTrace' ? {href: './numeric_trace.json?v=from-link'} : null; }};
global.window = {addEventListener() {}};
global.fetch = (url) => { requested = url; return new Promise(() => {}); };
global.TraceModel = {}; global.TraceI18n = {};
require('./site/app.js');
process.stdout.write(requested);
"""
    completed = subprocess.run(["node", "-e", script], cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "./numeric_trace.json?v=from-link"


def test_site_semantics_covers_every_exported_tensor_value_and_attention_derivative():
    artifact = build_position_reconstruction_artifact()
    semantics = build_site_semantics(artifact)

    assert set(semantics["tensors"]) == {f"w{index}" for index in range(24)}
    assert set(semantics["values"]) == {f"v{index}" for index in range(46)}
    assert set(semantics["derived"]) == {"op45_scores", "op45_attention"}
    assert semantics["tensors"]["w5"]["en"]["purpose"] != semantics["tensors"]["w2"]["en"]["purpose"]
    assert semantics["derived"]["op45_scores"]["en"]["columns"] == "candidate event (K transposed)"


def test_site_describes_the_current_executable_artifact_exactly():
    trace = build_numeric_site_trace()
    assert trace["operation_count"] == 45
    assert trace["final_output"] == {"value": 45, "shape": ["B", 64, 128]}
    assert trace["values"]["v0"]["shape"] == [1, 2048, 128]
    assert trace["values"]["v45"]["shape"] == [1, 64, 128]


def test_site_does_not_claim_unimplemented_recurrent_stages():
    assert "position subgraph" in HTML
    assert "Run artifact" not in HTML
    assert "JavaScript chess replay" in HTML
    assert "policy head" not in HTML.lower()
    assert "tree search" not in HTML.lower()


def test_numeric_history_fixture_uses_three_native_tokens_per_move():
    trace = build_numeric_site_trace()
    assert trace["fixture"]["moves"] == [[52, 54, 96], [47, 45, 102], [54, 45, 96]]
    assert "positionAfter" not in JS


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
    trace = build_numeric_site_trace()
    attention = trace["operations"][-1]
    assert attention["equation"] == "v45 = hardmax(v41 @ transpose(v40)) @ v44"
    assert attention["derived"] == ["op45_scores", "op45_attention"]
    assert 'id="matrixFlow"' in HTML
    assert 'id="numericOperations"' in HTML


def test_site_maps_every_runtime_opcode_to_a_generic_tensor_primitive():
    trace = build_numeric_site_trace()
    assert {operation["opcode"] for operation in trace["operations"]} == {
        "ROW_ROUTE", "TOKEN_PROJECT", "POSITION_ADD", "RESIDUAL_ADD",
        "FROZEN_EXPAND", "ROW_CONCAT", "HARDMAX_STE", "HULL_ATTN_2D",
    }


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


def test_numeric_trace_has_stable_source_provenance_and_structured_semantics():
    trace = build_numeric_site_trace()

    assert trace["provenance"]["executor"] == "vm_compiler.reference_executor"
    assert trace["provenance"]["dtype"] == "float32"
    assert len(trace["provenance"]["artifact_sha256"]) == 64
    assert len(trace["provenance"]["source_sha256"]) == 64
    assert trace["semantics"]["values"]["v45"]["producer"]["operation"] == 45
    assert trace["operations"][0]["attributes_info"]["start"] == 3
    assert trace["operations"][-1]["attributes_info"]["candidate_count"] == 2064


def test_trace_metadata_and_attention_spot_checks_are_independently_numeric():
    trace = build_numeric_site_trace()
    scores = {tuple(entry[:-1]): entry[-1] for entry in trace["derived"]["op45_scores"]["entries"]}
    attention = {tuple(entry[:-1]): entry[-1] for entry in trace["derived"]["op45_attention"]["entries"]}

    assert trace["operations"][0]["attributes_info"] == {"start": 3, "end": 1203, "stride": 3, "axis": "row"}
    assert trace["values"]["v0"]["shape"] == [1, 2048, 128]
    assert scores[(0, 28, 466)] == 2.0664076805114746
    assert scores[(0, 28, 465)] == 2.0664072036743164
    assert scores[(0, 28, 466)] - scores[(0, 28, 465)] == 2.0 ** -21
    assert attention[(0, 28, 466)] == 1


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
  winner,
  integer: [inspector.integerCoordinate('28.5', 63), inspector.integerCoordinate('', 63), inspector.integerCoordinate('63', 63)],
  compact: inspector.compactSquare(28)
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
        "integer": [{"ok": False}, {"ok": False}, {"ok": True, "value": 63}],
        "compact": "d5",
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
    assert re.search(r'<script defer src="\./matrix_inspector\.js\?v=[0-9a-f]{16}"></script>', HTML)


def test_trace_model_rejects_bad_fixture_and_decodes_only_exported_output_board():
    script = r"""
const trace = require('./site/numeric_trace.json');
const model = require('./site/trace_model.js');
const board = model.decodeBoard(trace);
const broken = structuredClone(trace); broken.fixture.moves = [[52,54,96]];
const missing = structuredClone(trace); delete missing.semantics.values;
console.log(JSON.stringify({d5: board.d5, e2: board.e2, history: model.decodeFixture(trace).map(move => move.uci), invalid: model.validateTrace(broken).ok, missing: model.validateTrace(missing).ok}));
"""
    completed = subprocess.run(["node", "-e", script], cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"d5": "white pawn", "e2": "empty", "history": ["e2e4", "d7d5", "e4d5"], "invalid": False, "missing": False}


def test_language_bootstrap_localizes_a_failure_before_trace_load_completes():
    script = r"""
const nodes = Object.fromEntries(['languageRu', 'languageEn', 'traceStatus'].map(id => [id, {
  id, handlers: {}, textContent: '', dataset: id === 'traceStatus' ? {i18n: 'loading'} : {}, classList: {toggle() {}},
  addEventListener(event, handler) { this.handlers[event] = handler; }
}]));
global.document = {documentElement: {}, querySelectorAll(selector) { return selector === '[data-i18n]' ? [nodes.traceStatus] : []; }, getElementById(id) { return nodes[id] || null; }};
const events = [];
global.window = {dispatchEvent(event) { events.push(event.detail); }};
global.CustomEvent = class { constructor(type, init) { this.type = type; this.detail = init.detail; }};
const i18n = require('./site/i18n.js');
i18n.setReady();
const readyRu = nodes.traceStatus.textContent;
nodes.languageEn.handlers.click();
i18n.apply('en'); // app receives the language event.
i18n.apply('en'); // inspector receives it after app's successful render.
const readyEn = nodes.traceStatus.textContent;
nodes.languageRu.handlers.click();
i18n.apply('ru');
i18n.apply('ru');
const readyRuAgain = nodes.traceStatus.textContent;
i18n.setLoadError('trace schema mismatch');
const ru = nodes.traceStatus.textContent;
nodes.languageEn.handlers.click();
console.log(JSON.stringify({readyRu, readyEn, readyRuAgain, ru, en: nodes.traceStatus.textContent, events}));
"""
    completed = subprocess.run(["node", "-e", script], cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "readyRu": "Точный экспорт загружен: v45 декодирован без JavaScript chess replay.",
        "readyEn": "Exact export loaded: v45 decoded without JavaScript chess replay.",
        "readyRuAgain": "Точный экспорт загружен: v45 декодирован без JavaScript chess replay.",
        "ru": "Ошибка проверки экспорта: trace schema mismatch",
        "en": "Export validation error: trace schema mismatch",
        "events": ["en", "ru", "en"],
    }
    inspector = (ROOT / "site" / "matrix_inspector.js").read_text(encoding="utf-8")
    assert "window.addEventListener(\"trace-language\"" in inspector
    assert 'byId("languageRu").addEventListener' not in inspector


def test_exported_semantics_preserve_source_specific_weights_patterns_and_producers():
    script = r"""
const trace = require('./site/numeric_trace.json');
const inspector = require('./site/matrix_inspector.js');
console.log(JSON.stringify({
  w0: trace.semantics.tensors.w0.en.purpose,
  w23: trace.semantics.tensors.w23.en.purpose,
  castleNoMatch: trace.semantics.patterns.castle[4].ru,
  epNoMatch: trace.semantics.patterns.en_passant[28].en,
  outputProducer: trace.semantics.values.v45.producer,
  scoreAxes: trace.semantics.derived.op45_scores.ru,
  attentionMode: inspector.highlightMode('HULL_ATTN_2D'),
  allOperationModels: [...new Set(trace.operations.map(op => op.opcode))]
    .every(opcode => inspector.operationModel(opcode) !== null),
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
    assert "s=8*(file-1)+(rank-1)" in result["w0"]
    assert "2^-21" in result["w23"]
    assert result["castleNoMatch"] == "нет совпадения / padding"
    assert result["epNoMatch"] == "no match / padding"
    assert result["outputProducer"] == {"operation": 45, "opcode": "HULL_ATTN_2D", "inputs": ["v41", "v40", "v44"]}
    assert result["scoreAxes"]["rows"] == "выходная компактная клетка (строка Q)"
    assert result["scoreAxes"]["columns"] == "событие-кандидат (K транспонирован)"
    assert result["attentionMode"] == "attention"
    assert result["allOperationModels"] is True
