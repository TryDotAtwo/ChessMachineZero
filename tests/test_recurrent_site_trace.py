import json
from pathlib import Path

import pytest

from vm_compiler.graph import OpCode
from vm_compiler.recurrent_circuit import build_recurrent_artifact


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def published():
    path = ROOT / "site" / "recurrent_trace.json"
    assert path.exists(), "the site must publish the full recurrent trace"
    return json.loads(path.read_text(encoding="utf-8"))


def test_recurrent_manifest_covers_the_exact_full_artifact(published):
    circuit, output = build_recurrent_artifact()
    artifact = circuit.artifact(output)

    assert published["schema_version"] == 1
    assert published["artifact"] == "recurrent_frozen_vm_v1"
    assert published["operation_count"] == len(circuit.operations) == 2877
    assert published["tensor_count"] == len(circuit.tensors) == 192
    assert published["input"]["shape"] == ["B", 2048, 128]
    assert published["output"]["shape"] == ["B", 2045, 128]
    assert published["final_output"] == f"v{circuit.operations[-1].outputs[0]}"
    assert published["runtime_opcodes"] == sorted({op.opcode.name for op in artifact.operations})

    for index, (record, operation) in enumerate(
            zip(published["operations"], artifact.operations), 1):
        assert record["index"] == index
        assert record["opcode"] == operation.opcode.name
        assert record["inputs"] == [f"v{value}" for value in operation.inputs]
        assert record["output"] == f"v{operation.outputs[0]}"
        assert record["attributes"] == list(operation.attributes)
        assert record["output_shape"] == ["B", *circuit.shapes[operation.outputs[0]]]


def test_stages_cover_every_operation_once_and_explain_both_languages(published):
    covered = []
    for stage in published["stages"]:
        covered.extend(range(stage["first_operation"], stage["last_operation"] + 1))
        for language in ("ru", "en"):
            assert stage[language]["name"].strip()
            assert stage[language]["purpose"].strip()
    assert covered == list(range(1, published["operation_count"] + 1))

    for operation in published["operations"]:
        assert operation["stage"] in {stage["id"] for stage in published["stages"]}
        for language in ("ru", "en"):
            assert operation["semantics"][language]["name"].strip()
            assert operation["semantics"][language]["purpose"].strip()
        assert "SSA:" not in operation["semantics"]["ru"]["name"]
        assert operation["sample"]["output"]["window"]["values"]


def test_every_generic_opcode_has_a_scalar_proof_and_real_matrix_windows(published):
    expected_kinds = {
        OpCode.ROW_ROUTE.name: "route",
        OpCode.TOKEN_PROJECT.name: "matmul",
        OpCode.POSITION_ADD.name: "add",
        OpCode.HULL_ATTN_2D.name: "attention",
        OpCode.RESIDUAL_ADD.name: "add",
        OpCode.HARDMAX_STE.name: "argmax",
        OpCode.FROZEN_EXPAND.name: "expand",
        OpCode.ROW_CONCAT.name: "concat",
        OpCode.MATRIX_TRANSPOSE.name: "transpose",
        OpCode.MATRIX_RESHAPE.name: "reshape",
        OpCode.MATRIX_MATMUL.name: "matmul",
        OpCode.GROUPED_MATRIX_MATMUL.name: "grouped_matmul",
    }
    assert set(published["runtime_opcodes"]) == set(expected_kinds)
    for operation in published["operations"]:
        sample = operation["sample"]
        assert sample["proof"]["kind"] == expected_kinds[operation["opcode"]]
        assert sample["cards"]
        assert sample["output"]["value"] == sample["proof"]["result"]

    matmul = published["operations"][8]
    assert matmul["opcode"] == "MATRIX_MATMUL"
    assert matmul["sample"]["proof"]["formula"] == "row · column"
    assert matmul["sample"]["proof"]["omitted_zero_terms"] == (
        matmul["sample"]["proof"]["contraction"]
        - len(matmul["sample"]["proof"]["terms"])
    )
    assert sum(term["product"] for term in matmul["sample"]["proof"]["terms"]) == pytest.approx(
        matmul["sample"]["proof"]["result"], abs=1e-6
    )
    grouped = next(op for op in published["operations"] if op["opcode"] == "GROUPED_MATRIX_MATMUL")
    assert grouped["sample"]["proof"]["group"] >= 0
    hardmax = next(op for op in published["operations"] if op["opcode"] == "HARDMAX_STE")
    assert hardmax["sample"]["proof"]["operator"] == "ARGMAX"


def test_primary_names_and_frozen_descriptions_explain_meaning_not_only_ids(published):
    request_match = published["operations"][8]
    assert request_match["semantics"]["ru"]["name"] == (
        "Оценка совпадения запроса с каждым ходом LEGAL_SET"
    )
    hardmax = published["operations"][12]
    assert hardmax["semantics"]["ru"]["name"] == (
        "Hardmax/STE: выбор максимума в каждой строке"
    )
    grouped = next(op for op in published["operations"] if op["opcode"] == "GROUPED_MATRIX_MATMUL")
    assert grouped["semantics"]["ru"]["name"] == (
        "Число блокеров на en-passant луче для каждой группы"
    )
    frozen_operation = next(op for op in published["operations"] if op["frozen"])
    frozen = frozen_operation["frozen"][0]["semantics"]["ru"]
    assert frozen_operation["semantics"]["ru"]["name"] in frozen["purpose"]
    assert "Неизменяемый" in frozen["purpose"]


def test_fixture_is_a_real_recurrent_transition_with_explicit_feedback(published):
    fixture = published["fixture"]
    assert fixture["request"] == [52, 54, 96]
    assert fixture["request_uci"] == "e2e4"
    assert fixture["prior"]["status"] == "OK"
    assert fixture["output"]["status"] == "OK"
    assert fixture["output"]["history_plies"] == 1
    assert fixture["output"]["legal_count"] == 20
    assert fixture["feedback"]["output_context_sha256"] == fixture["output"]["context_sha256"]
    assert fixture["feedback"]["next_input_context_sha256"] == fixture["output"]["context_sha256"]
    assert published["provenance"]["numeric_executor"] == "vm_compiler.reference_executor"
    assert published["provenance"]["native_intermediate_capture"] is False
    native = published["provenance"]["native_acceptance"]
    assert native["contexts"] == 47
    assert native["feedback_edges"] == 40
    assert native["context_0_bindings"] == 5
