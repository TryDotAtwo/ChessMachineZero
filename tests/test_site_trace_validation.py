"""Real Node validation of the published trace, with isolated malformed clones."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
CASES = [
    (f"missing-{group}-{key}", f"delete t.{group}.{key}")
    for group, keys in (
        ("tensors", [f"w{i}" for i in range(24)]),
        ("values", [f"v{i}" for i in range(46)]),
        ("derived", ["op45_scores", "op45_attention"]),
    )
    for key in keys
]
CASES += [
    ("input-extent", "t.values.v0.shape[1] = 1"),
    ("weight-extent", "t.tensors.w0.shape[0] = 127"),
    ("intermediate-extent", "t.values.v34.shape[2] = 28"),
    ("scores-extent", "t.derived.op45_scores.shape[2] = 2063"),
    ("missing-entries", "delete t.values.v1.entries"),
    ("wrong-layout", "t.values.v1.layout = 'dense'"),
    ("wrong-nnz", "t.values.v1.nnz += 1"),
    ("duplicate-coordinate", "t.values.v45.entries.push(t.values.v45.entries[0].slice()); t.values.v45.nnz += 1"),
    ("out-of-bounds", "t.values.v45.entries[0][1] = 64"),
    ("negative-coordinate", "t.values.v45.entries[0][1] = -1"),
    ("fractional-coordinate", "t.values.v45.entries[0][1] = 0.5"),
    ("wrong-coordinate-rank", "t.values.v45.entries[0].unshift(0)"),
    ("infinite-entry", "t.values.v45.entries[0][3] = Infinity"),
    ("nan-weight", "t.tensors.w0.entries[0][2] = NaN"),
    ("non-fp32-entry", "t.tensors.w0.entries[0][2] = 1 + 2 ** -30"),
    ("stored-zero", "t.values.v45.entries[0][3] = 0"),
    ("empty-board", "t.values.v45.entries = []; t.values.v45.nnz = 0"),
    ("wrong-piece-board", "t.values.v45.entries.find(e => e[1] === 28)[2] = 106"),
    ("coupled-board-payload", "t.values.v45.entries.find(e => e[1] === 28)[2] = 106; t.values.v44.entries.find(e => e[1] === 466)[2] = 106"),
    ("fixture-null", "t.fixture.moves[0] = null"),
    ("fixture-object", "t.fixture.moves[0] = {}"),
    ("fixture-inconsistent-input", "t.values.v0.entries.find(e => e[1] === 3)[2] = 53"),
    ("unused-input-not-fixture", "t.values.v0.entries.find(e => e[1] === 0)[2] = 11"),
    ("missing-provenance", "delete t.provenance"),
    ("bad-provenance-hash", "t.provenance.source_sha256 = 'not a hash'"),
    ("wrong-executor", "t.provenance.executor = 'native CUDA'"),
    ("missing-source-identifiers", "delete t.provenance.source_identifiers"),
    ("missing-attributes", "delete t.operations[44].attributes_info"),
    ("impossible-top-k", "t.operations[44].attributes_info.top_k = 999"),
    ("wrong-route-stride", "t.operations[0].attributes_info.stride = 2"),
    ("wrong-producer", "t.semantics.values.v45.producer.inputs = ['v40']"),
    ("missing-operation-value", "t.operations[1].inputs[0] = 'v999'"),
    ("wrong-opcode", "t.operations[1].opcode = 'UNKNOWN'"),
    ("wrong-operation-index", "t.operations[1].index = 1"),
    ("null-semantics", "t.semantics.tensors.w0 = null"),
    ("missing-semantic-purpose", "delete t.semantics.tensors.w0.en.purpose"),
    ("blank-semantic-axis", "t.semantics.values.v45.ru.rows = '   '"),
    ("missing-stream-metadata", "delete t.semantics.event_streams"),
    ("wrong-stream-boundary", "t.semantics.event_streams[1].start = 63"),
    ("missing-pattern-description", "delete t.semantics.patterns.castle[0].ru"),
    ("missing-sample", "delete t.operations[0].sample"),
    ("wrong-sample-value", "t.operations[0].sample.output_value = 0"),
    ("wrong-sample-term", "t.operations[44].sample.score_terms[0][1] += 1"),
    ("route-arithmetic", "t.values.v1.entries[0][2] = 53"),
    ("expand-arithmetic", "t.values.v4.entries[0][3] = 0.5"),
    ("projection-arithmetic", "setCell(t.values.v5, [0,0,0], 1)"),
    ("residual-arithmetic", "setCell(t.values.v8, [0,0,0], 1)"),
    ("bias-arithmetic", "t.values.v10.entries[0][3] = 3.5"),
    ("hardmax-arithmetic", "t.values.v11.entries[0][2] = 0"),
    ("concat-arithmetic", "t.values.v38.entries[0][2] = 12"),
    ("query-arithmetic", "t.values.v41.entries[0][3] += 0.25"),
    ("epsilon-sized-score-drift", "t.derived.op45_scores.entries[0][3] = Math.fround(t.derived.op45_scores.entries[0][3] + 2 ** -21)"),
    ("wrong-attention-winner", "t.derived.op45_attention.entries.find(e => e[1] === 28)[2] = 465"),
]


@pytest.fixture(scope="module")
def node_results():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the real site validator acceptance")
    script = r"""
const fs = require('fs');
const api = require('./site/trace_model.js');
const raw = fs.readFileSync('./site/numeric_trace.json', 'utf8');
function setCell(matrix, coordinates, value) {
  const entry = matrix.entries.find(e => coordinates.every((v,i) => e[i] === v));
  if (entry) entry[entry.length - 1] = value;
  else { matrix.entries.push([...coordinates, value]); matrix.nnz += 1; }
}
const baseline = JSON.parse(raw);
const result = {baseline: api.validateTrace(baseline), board: api.decodeBoard(baseline),
  moves: api.decodeFixture(baseline).map(m => m.uci), cases: {}};
for (const [name, mutation] of CASES) {
  const trace = JSON.parse(raw);
  new Function('t', 'setCell', mutation)(trace, setCell);
  try { result.cases[name] = api.validateTrace(trace); }
  catch (error) { result.cases[name] = {threw: error.message}; }
}
// A prior successful validation must not whitelist a subsequently mutated object.
baseline.values.v45.entries.find(e => e[1] === 28)[2] = 106;
result.mutatedPreviouslyValid = api.validateTrace(baseline);
console.log(JSON.stringify(result));
""".replace("CASES", json.dumps(CASES))
    completed = subprocess.run(
        [node, "-e", script], cwd=ROOT, check=True, capture_output=True, text=True, timeout=90
    )
    assert completed.stderr == ""
    return json.loads(completed.stdout)


def test_exact_published_trace_and_literal_fixture_board_are_accepted(node_results):
    assert node_results["baseline"] == {"ok": True}
    assert node_results["moves"] == ["e2e4", "d7d5", "e4d5"]
    assert node_results["board"]["d5"] == "white pawn"
    assert node_results["board"]["e2"] == "empty"
    assert node_results["board"]["e4"] == "empty"
    assert node_results["board"]["d7"] == "empty"
    assert node_results["board"]["a1"] == "white rook"


@pytest.mark.parametrize("case", [name for name, _ in CASES])
def test_malformed_or_inconsistent_trace_fails_closed(node_results, case):
    result = node_results["cases"][case]
    assert result.get("ok") is False, (case, result)
    assert isinstance(result.get("error"), str) and result["error"].strip()


def test_validation_does_not_cache_by_mutable_object_identity(node_results):
    assert node_results["mutatedPreviouslyValid"].get("ok") is False
