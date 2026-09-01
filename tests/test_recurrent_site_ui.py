import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "site" / "index.html").read_text(encoding="utf-8")


def test_page_leads_with_full_vm_and_retains_the_position_microscope():
    for marker in (
        'id="fullVmStages"',
        'id="fullVmOperations"',
        'id="fullVmMatrixFlow"',
        'id="fullVmProof"',
        'id="contextInput"',
        'id="contextOutput"',
        'id="feedbackEdge"',
        'data-i18n="ssaGlossary"',
        'id="legacyPositionInspector"',
    ):
        assert marker in HTML
    assert "2877" in HTML
    assert "45 SSA" not in HTML
    assert re.search(
        r'<script defer src="\./recurrent_inspector\.js\?v=[0-9a-f]{16}"></script>', HTML
    )


def test_recurrent_browser_model_validates_topology_feedback_and_scalar_proofs():
    script = r"""
const trace = require('./site/recurrent_trace.json');
const inspector = require('./site/recurrent_inspector.js');
const mutations = {
  count: t => t.operation_count -= 1,
  stageGap: t => t.stages[1].first_operation += 1,
  feedback: t => t.fixture.feedback.next_input_context_sha256 = '0'.repeat(64),
  forgedStatus: t => t.fixture.output.status = 'WHITE_WIN',
  missingSample: t => delete t.operations[10].sample,
  badProof: t => t.operations[8].sample.proof.result += 1,
  forgedValue: t => { t.operations[8].sample.output.value += 1; t.operations[8].sample.proof.result += 1; },
  unknownOpcode: t => t.operations[0].opcode = 'CHESS_MOVE',
};
const rejected = {};
for (const [name, mutate] of Object.entries(mutations)) {
  const clone = structuredClone(trace); mutate(clone); rejected[name] = inspector.validateTrace(clone).ok;
}
const matmul = trace.operations.find(op => op.sample.proof.kind === 'matmul');
const hardmax = trace.operations.find(op => op.sample.proof.kind === 'argmax');
const grouped = trace.operations.find(op => op.sample.proof.kind === 'grouped_matmul');
console.log(JSON.stringify({
  baseline: inspector.validateTrace(trace), rejected,
  matmul: inspector.proofEquation(matmul.sample.proof, 'ru'),
  hardmax: inspector.proofEquation(hardmax.sample.proof, 'ru'),
  grouped: inspector.proofEquation(grouped.sample.proof, 'en'),
  readyRu: inspector.readyStatus(trace, 'ru'),
  readyEn: inspector.readyStatus(trace, 'en'),
  models: [...new Set(trace.operations.map(op => inspector.operationModel(op.opcode)))].sort(),
}));
"""
    completed = subprocess.run(
        ["node", "-e", script], cwd=ROOT, text=True, encoding="utf-8",
        capture_output=True, check=True,
    )
    result = json.loads(completed.stdout)
    assert result["baseline"] == {"ok": True}
    assert result["rejected"] == {name: False for name in (
        "count", "stageGap", "feedback", "forgedStatus", "missingSample",
        "badProof", "forgedValue", "unknownOpcode"
    )}
    assert "строка" in result["matmul"] and "столбец" in result["matmul"]
    assert "нулев" in result["matmul"]
    assert "ARGMAX" in result["hardmax"]
    assert "group" in result["grouped"].lower()
    assert result["readyRu"] == "Полный trace проверен: 2877 ops / 192 frozen tensors"
    assert result["readyEn"] == "Full trace validated: 2877 ops / 192 frozen tensors"
    assert result["models"] == [
        "add", "argmax", "attention", "concat", "expand", "grouped_matmul",
        "matmul", "reshape", "route", "transpose",
    ]


def test_recurrent_ui_has_no_chess_interpreter_and_names_frozen_matrix_meanings():
    javascript = (ROOT / "site" / "recurrent_inspector.js").read_text(encoding="utf-8")
    assert "applyMove" not in javascript
    assert "isLegal" not in javascript
    assert "chess.js" not in javascript
    assert "sample.cards" in javascript
    assert "operation.frozen" in javascript
    assert "ARGMAX" in javascript
