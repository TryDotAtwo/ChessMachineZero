"""Build a compact, operation-complete trace for the recurrent VM website.

The export records one genuine reference execution of the exact serialized
artifact. It keeps a small matrix window and a scalar arithmetic proof for all
operations instead of publishing the roughly 3 GiB retained SSA payload.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .context import build_context_0
from .graph import OpCode
from .protocol import PIECE_CHANNELS, STATUS_CHANNELS, VOCAB_SIZE, encode_move_one_hot
from .recurrent_circuit import build_recurrent_artifact
from .reference_executor import execute_artifact_reference_values


STAGES = (
    ("request", 1, 93,
     "Запрос и добавление в историю", "Request validation and history append",
     "Сопоставление тройки хода с предыдущим LEGAL_SET и тензорная запись в первую свободную тройку истории.",
     "Match the requested triple against the previous LEGAL_SET and tensor-write it into the first free history slot."),
    ("position", 94, 140,
     "Восстановление новой позиции", "New-position reconstruction",
     "Последнее событие для каждой клетки выбирается frozen 2D-attention после принятого хода.",
     "Frozen 2D attention selects the latest event for every square after the accepted move."),
    ("legal", 141, 644,
     "Следующий LEGAL_SET", "Next LEGAL_SET",
     "7780 фиксированных геометрических кандидатов проходят occupancy, path, special-move и king-safety gates, затем stable compaction.",
     "7,780 fixed geometric candidates pass occupancy, path, special-move and king-safety gates, then stable compaction."),
    ("repetition_key", 645, 1584,
     "Ключи повторения", "Repetition keys",
     "Префиксные позиции, сторона, права рокировки и только реально доступная en-passant клетка образуют ключ троекратного повторения.",
     "Prefix positions, side, castling rights and only an actually legal en-passant square form the threefold key."),
    ("future_boards", 1585, 1694,
     "Доски после возможных ходов", "Possible post-move boards",
     "Все 256 compact legal slots применяются параллельно, включая rook relocation и en-passant victim clearing.",
     "All 256 compact legal slots are applied in parallel, including rook relocation and en-passant victim clearing."),
    ("reply_exists", 1695, 2410,
     "Существование ответа", "Legal-reply existence",
     "Factorized [256,candidate] проверки доказывают, оставляет ли каждый не обнуляющий clock ход хотя бы один легальный ответ.",
     "Factorized [256,candidate] checks prove whether each non-zeroing move leaves at least one legal reply."),
    ("future_repetition", 2411, 2475,
     "Повторение следующим ходом", "Next-move repetition",
     "Будущие board/turn/rights/EP ключи сравниваются со всеми присутствующими префиксами.",
     "Future board/turn/rights/EP keys are compared with every present prefix."),
    ("draws", 2476, 2693,
     "Автоматические ничьи", "Automatic draws",
     "Halfmove clock, точное правило 50 ходов, троекратное повторение и недостаточный материал сводятся в DRAW gate.",
     "Halfmove clock, the exact fifty-move rule, threefold repetition and insufficient material reduce into the DRAW gate."),
    ("status", 2694, 2877,
     "Мат, пат, статус и выход", "Mate, stalemate, status and output",
     "Пустой LEGAL_SET различает мат/пат, terminal contexts поглощаются, затем собирается новый [2045,128] context.",
     "An empty LEGAL_SET distinguishes mate/stalemate, terminal contexts absorb, then the next [2045,128] context is packed."),
)


OPERATOR_NAMES = {
    OpCode.ROW_ROUTE: ("маршрутизация строк", "row routing"),
    OpCode.TOKEN_PROJECT: ("матричная проекция", "matrix projection"),
    OpCode.OUTPUT_PROJECT: ("выходная проекция", "output projection"),
    OpCode.POSITION_ADD: ("сложение с frozen-матрицей", "add frozen matrix"),
    OpCode.HULL_ATTN_2D: ("2D-attention с hardmax", "2D hardmax attention"),
    OpCode.RESIDUAL_ADD: ("поэлементное сложение", "elementwise addition"),
    OpCode.HARDMAX_STE: ("ARGMAX в one-hot", "ARGMAX to one-hot"),
    OpCode.FROZEN_EXPAND: ("расширение frozen-матрицы по batch", "frozen batch expansion"),
    OpCode.ROW_CONCAT: ("конкатенация строк", "row concatenation"),
    OpCode.MATRIX_TRANSPOSE: ("транспонирование матрицы", "matrix transpose"),
    OpCode.MATRIX_RESHAPE: ("изменение формы без перестановки", "shape-only reshape"),
    OpCode.MATRIX_MATMUL: ("строка на столбец", "row-by-column multiplication"),
    OpCode.GROUPED_MATRIX_MATMUL: ("независимые групповые GEMM", "independent grouped GEMMs"),
}


SPECIAL_NAMES = {
    "row_route": ("Маршрутизация строк", "Row routing"),
    "token_project": ("Матричная проекция", "Matrix projection"),
    "output_project": ("Выходная матричная проекция", "Output matrix projection"),
    "position_add": ("Сложение с frozen-матрицей", "Addition with a frozen matrix"),
    "residual_add": ("Поэлементное сложение", "Elementwise addition"),
    "hardmax_ste": ("Hardmax/STE: выбор максимума в каждой строке",
                    "Hardmax/STE: select each row maximum"),
    "frozen_expand": ("Расширение frozen-матрицы по batch",
                      "Expand a frozen matrix across the batch"),
    "row_concat": ("Конкатенация строк", "Row concatenation"),
    "matrix_transpose": ("Транспонирование матрицы", "Matrix transpose"),
    "matrix_reshape": ("Изменение формы без перестановки", "Shape-only reshape"),
    "matrix_matmul": ("Умножение строки на столбец", "Row-by-column multiplication"),
    "grouped_matrix_matmul": ("Независимые групповые GEMM",
                              "Independent grouped GEMMs"),
    "hull_attn_2d": ("2D-attention с hardmax", "2D hardmax attention"),
    "request_and_previous_context": ("Запрос + предыдущий контекст", "Request + previous context"),
    "requested_move": ("Три one-hot строки запрошенного хода", "Three one-hot requested-move rows"),
    "previous_history": ("Предыдущая хронологическая история", "Previous chronological history"),
    "previous_legal_set": ("LEGAL_SET предыдущего состояния", "Previous-state LEGAL_SET"),
    "previous_status": ("Статус предыдущего состояния", "Previous-state status"),
    "request_is_legal": ("Флаг принадлежности запросу LEGAL_SET", "Request membership in LEGAL_SET"),
    "request_is_accepted": ("Флаг принятого хода", "Accepted-request flag"),
    "request_legal_match_score": (
        "Оценка совпадения запроса с каждым ходом LEGAL_SET",
        "Request match score for every LEGAL_SET move",
    ),
    "next_history": ("История после принятого хода", "History after the accepted move"),
    "legal_set": ("Отсортированный LEGAL_SET следующей стороны", "Sorted next-side LEGAL_SET"),
    "future_has_legal_reply": ("Для каждого хода: существует ли ответ", "Per move: whether a legal reply exists"),
    "effective_ep_ray_blocker_counts": (
        "Число блокеров на en-passant луче для каждой группы",
        "Per-group blocker count on effective en-passant rays",
    ),
    "halfmove_clock": ("Счётчик полуходов без pawn/capture", "Halfmoves since pawn move/capture"),
    "automatic_fifty_move_claim": ("Автоматическое правило 50 ходов", "Automatic fifty-move rule"),
    "current_threefold_repetition": ("Троекратное повторение уже сейчас", "Current threefold repetition"),
    "next_move_threefold_repetition": ("Троекратное повторение доступно ходом", "Threefold reachable by one move"),
    "insufficient_material": ("Недостаточный материал", "Insufficient material"),
    "next_status": ("Итоговый технический статус", "Final technical status"),
    "next_context": ("Новый рекуррентный контекст", "Next recurrent context"),
}


RU_WORDS = {
    "future": "будущий", "reply": "ответ", "source": "источник", "sources": "источники",
    "target": "цель", "targets": "цели", "piece": "фигура", "pieces": "фигуры",
    "legal": "легальный", "candidate": "кандидат", "candidates": "кандидаты",
    "current": "текущий", "previous": "предыдущий", "next": "следующий",
    "history": "история", "status": "статус", "position": "позиция", "board": "доска",
    "boards": "доски", "move": "ход", "moves": "ходы", "row": "строка",
    "rows": "строки", "column": "столбец", "count": "счётчик", "mask": "маска",
    "match": "совпадение", "score": "оценка", "scores": "оценки", "kind": "тип",
    "occupancy": "занятость", "attack": "атака", "attacked": "атакованные",
    "king": "король", "safe": "безопасность", "rights": "права", "castling": "рокировка",
    "checker": "шахующая фигура", "checkers": "шахующие фигуры", "pin": "связка",
    "between": "между", "blocker": "блокер", "blockers": "блокеры", "prefix": "префикс",
    "effective": "эффективный", "ep": "en-passant", "automatic": "автоматический",
    "fifty": "пятьдесят", "material": "материал", "draw": "ничья", "output": "выход",
    "input": "вход", "selected": "выбранный", "accepted": "принятый", "activity": "активность",
    "present": "присутствие", "empty": "пустой", "padding": "padding", "result": "результат",
    "repeated": "повторенный", "repeat": "повтор", "geometry": "геометрия", "path": "траектория",
    "clear": "свободный", "invalid": "недопустимый", "evasion": "уход от шаха",
    "request": "запрос", "token": "токен", "project": "проекция", "add": "сложение",
    "matrix": "матрица", "reshape": "изменение формы", "residual": "остаточное",
    "matmul": "матричное умножение", "transpose": "транспонирование",
    "route": "маршрутизация", "concat": "конкатенация", "plies": "полуходы",
    "after": "после", "enemy": "противник", "pawn": "пешка", "pawns": "пешки",
    "slot": "слот", "squares": "клетки", "square": "клетка", "castle": "рокировка",
    "rook": "ладья", "black": "чёрные", "white": "белые", "side": "сторона",
    "counts": "числа", "ray": "луч", "zeroing": "обнуляющий счётчик",
    "capture": "взятие", "attacker": "атакующая фигура", "own": "свой",
    "terminal": "терминальный", "initial": "начальный", "bias": "смещение",
    "key": "ключ", "rank": "ранг", "color": "цвет", "victim": "снимаемая фигура",
    "overflow": "переполнение", "decoder": "декодер", "occurrences": "число вхождений",
    "halfmove": "полуход", "clock": "счётчик", "queen": "ферзь",
    "knight": "конь", "bishop": "слон", "stalemate": "пат", "win": "победа",
}


def _number(value) -> int | float:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _human_name(key: str, language: str) -> str:
    if key in SPECIAL_NAMES:
        return SPECIAL_NAMES[key][0 if language == "ru" else 1]
    words = key.split("_")
    if language == "en":
        return " ".join(words)
    return " ".join(RU_WORDS.get(word, word) for word in words)


def _stage_for(index: int):
    return next(stage for stage in STAGES if stage[1] <= index <= stage[2])


def _shape(tensor: torch.Tensor):
    return list(tensor.shape)


def _window(tensor: torch.Tensor, row: int, column: int, *, role: str, identifier: str,
            name: dict[str, str], transpose: bool = False) -> dict[str, object]:
    matrix = tensor.detach()
    if matrix.ndim == 3:
        matrix = matrix[0]
    if transpose:
        matrix = matrix.transpose(0, 1)
    rows, columns = matrix.shape
    row = max(0, min(int(row), rows - 1))
    column = max(0, min(int(column), columns - 1))
    row_start = max(0, min(row - 2, max(0, rows - 6)))
    column_start = max(0, min(column - 3, max(0, columns - 8)))
    view = matrix[row_start:row_start + 6, column_start:column_start + 8]
    return {
        "role": role,
        "id": identifier,
        "name": name,
        "shape": [rows, columns],
        "row_start": row_start,
        "column_start": column_start,
        "selected": [row, column],
        "values": [[_number(value) for value in line] for line in view.tolist()],
    }


def _value_name(circuit, value_id: int) -> dict[str, str]:
    key = circuit.semantics[value_id]["name"]
    return {"ru": _human_name(key, "ru"), "en": _human_name(key, "en")}


def _weight_name(circuit, index: int) -> dict[str, str]:
    key = circuit.tensors[index].name
    return {
        "ru": "Frozen-матрица: " + _human_name(key, "ru"),
        "en": "Frozen matrix: " + _human_name(key, "en"),
    }


def _input_coordinate(tensor: torch.Tensor, row: int, column: int):
    return (0 if tensor.shape[1] == 1 else row,
            0 if tensor.shape[2] == 1 else column)


def _weight_coordinate(tensor: torch.Tensor, row: int, column: int):
    return (0 if tensor.shape[0] == 1 else row,
            0 if tensor.shape[1] == 1 else column)


def _first_column(tensor: torch.Tensor, row: int = 0) -> int:
    vector = tensor.detach()[0, row]
    nonzero = torch.nonzero(vector, as_tuple=False)
    return int(nonzero[0, 0]) if len(nonzero) else 0


def _matmul_terms(left: torch.Tensor, right: torch.Tensor, row: int, column: int,
                   *, right_row_offset: int = 0):
    left_vector = left.detach()[0, row]
    right_vector = right.detach()[0, right_row_offset:right_row_offset + left_vector.numel(), column]
    products = left_vector * right_vector
    indices = torch.nonzero(products, as_tuple=False).flatten().tolist()
    return [[int(index), _number(left_vector[index]), _number(right_vector[index]),
             _number(products[index])] for index in indices]


def _projection_terms(left: torch.Tensor, right: torch.Tensor, row: int, column: int):
    left_vector = left.detach()[0, row]
    right_vector = right.detach()[:, column]
    products = left_vector * right_vector
    indices = torch.nonzero(products, as_tuple=False).flatten().tolist()
    return [[int(index), _number(left_vector[index]), _number(right_vector[index]),
             _number(products[index])] for index in indices]


def _equation(operation) -> str:
    output = f"v{operation.outputs[0]}"
    inputs = [f"v{value}" for value in operation.inputs]
    if operation.opcode == OpCode.ROW_ROUTE:
        start, end, *stride = operation.attributes
        return f"{output} = {inputs[0]}[:, {start}:{end}:{stride[0] if stride else 1}, :]"
    if operation.opcode in (OpCode.TOKEN_PROJECT, OpCode.OUTPUT_PROJECT):
        return f"{output} = {inputs[0]} @ w{operation.attributes[0]}"
    if operation.opcode == OpCode.POSITION_ADD:
        return f"{output} = {inputs[0]} + w{operation.attributes[0]}"
    if operation.opcode == OpCode.RESIDUAL_ADD:
        return f"{output} = {inputs[0]} + {inputs[1]}"
    if operation.opcode == OpCode.FROZEN_EXPAND:
        return f"{output} = expand_batch(w{operation.attributes[0]})"
    if operation.opcode == OpCode.ROW_CONCAT:
        return f"{output} = concat_rows({', '.join(inputs)})"
    if operation.opcode == OpCode.HARDMAX_STE:
        return f"{output} = one_hot(ARGMAX({inputs[0]}))"
    if operation.opcode == OpCode.HULL_ATTN_2D:
        return f"{output} = hardmax({inputs[0]} @ transpose({inputs[1]})) @ {inputs[2]}"
    if operation.opcode == OpCode.MATRIX_TRANSPOSE:
        return f"{output} = transpose({inputs[0]})"
    if operation.opcode == OpCode.MATRIX_RESHAPE:
        return f"{output} = reshape({inputs[0]}, {operation.attributes[0]}, {operation.attributes[1]})"
    if operation.opcode == OpCode.MATRIX_MATMUL:
        return f"{output} = {inputs[0]} @ {inputs[1]}"
    if operation.opcode == OpCode.GROUPED_MATRIX_MATMUL:
        return f"{output} = grouped_matmul({inputs[0]}, {inputs[1]}, groups={operation.attributes[0]})"
    raise ValueError(f"unsupported recurrent trace opcode {operation.opcode.name}")


def _sample_operation(circuit, operation, values, tensors):
    output_id = operation.outputs[0]
    output = values[output_id]
    row = 0
    column = _first_column(output, row)
    result = _number(output[0, row, column])
    output_card = _window(output, row, column, role="output", identifier=f"v{output_id}",
                          name=_value_name(circuit, output_id))
    cards = []
    proof: dict[str, object]

    if operation.opcode == OpCode.ROW_ROUTE:
        source = values[operation.inputs[0]]
        start, _, *stride = operation.attributes
        source_row = start + row * (stride[0] if stride else 1)
        cards.append(_window(source, source_row, column, role="input",
                             identifier=f"v{operation.inputs[0]}",
                             name=_value_name(circuit, operation.inputs[0])))
        proof = {"kind": "route", "source_index": [0, source_row, column], "result": result}
    elif operation.opcode in (OpCode.TOKEN_PROJECT, OpCode.OUTPUT_PROJECT):
        source = values[operation.inputs[0]]
        weight_index = operation.attributes[0]
        weight = tensors[weight_index]
        terms = _projection_terms(source, weight, row, column)
        k = terms[0][0] if terms else 0
        cards.extend((
            _window(source, row, k, role="left_row", identifier=f"v{operation.inputs[0]}",
                    name=_value_name(circuit, operation.inputs[0])),
            _window(weight, k, column, role="right_column", identifier=f"w{weight_index}",
                    name=_weight_name(circuit, weight_index)),
        ))
        proof = {"kind": "matmul", "formula": "row · column", "terms": [
            {"k": term[0], "left": term[1], "right": term[2], "product": term[3]}
            for term in terms
        ], "contraction": int(source.shape[2]),
            "omitted_zero_terms": int(source.shape[2]) - len(terms), "result": result}
    elif operation.opcode == OpCode.MATRIX_MATMUL:
        left, right = (values[value] for value in operation.inputs)
        terms = _matmul_terms(left, right, row, column)
        k = terms[0][0] if terms else 0
        cards.extend((
            _window(left, row, k, role="left_row", identifier=f"v{operation.inputs[0]}",
                    name=_value_name(circuit, operation.inputs[0])),
            _window(right, k, column, role="right_column", identifier=f"v{operation.inputs[1]}",
                    name=_value_name(circuit, operation.inputs[1])),
        ))
        proof = {"kind": "matmul", "formula": "row · column", "terms": [
            {"k": term[0], "left": term[1], "right": term[2], "product": term[3]}
            for term in terms
        ], "contraction": int(left.shape[2]),
            "omitted_zero_terms": int(left.shape[2]) - len(terms), "result": result}
    elif operation.opcode == OpCode.GROUPED_MATRIX_MATMUL:
        left, right = (values[value] for value in operation.inputs)
        groups = operation.attributes[0]
        rows_per_group = left.shape[1] // groups
        group = row // rows_per_group
        offset = group * left.shape[2]
        terms = _matmul_terms(left, right, row, column, right_row_offset=offset)
        k = terms[0][0] if terms else 0
        cards.extend((
            _window(left, row, k, role="left_row", identifier=f"v{operation.inputs[0]}",
                    name=_value_name(circuit, operation.inputs[0])),
            _window(right, offset + k, column, role="right_column",
                    identifier=f"v{operation.inputs[1]}",
                    name=_value_name(circuit, operation.inputs[1])),
        ))
        proof = {"kind": "grouped_matmul", "formula": "group row · group column",
                 "group": int(group), "groups": int(groups), "terms": [
                     {"k": term[0], "left": term[1], "right": term[2], "product": term[3]}
                     for term in terms
                 ], "contraction": int(left.shape[2]),
                 "omitted_zero_terms": int(left.shape[2]) - len(terms), "result": result}
    elif operation.opcode in (OpCode.POSITION_ADD, OpCode.RESIDUAL_ADD):
        left = values[operation.inputs[0]]
        left_row, left_column = _input_coordinate(left, row, column)
        left_value = _number(left[0, left_row, left_column])
        cards.append(_window(left, left_row, left_column, role="left",
                             identifier=f"v{operation.inputs[0]}",
                             name=_value_name(circuit, operation.inputs[0])))
        if operation.opcode == OpCode.POSITION_ADD:
            right_index = operation.attributes[0]
            right = tensors[right_index]
            right_row, right_column = _weight_coordinate(right, row, column)
            right_value = _number(right[right_row, right_column])
            cards.append(_window(right, right_row, right_column, role="right_frozen",
                                 identifier=f"w{right_index}", name=_weight_name(circuit, right_index)))
        else:
            right = values[operation.inputs[1]]
            right_row, right_column = _input_coordinate(right, row, column)
            right_value = _number(right[0, right_row, right_column])
            cards.append(_window(right, right_row, right_column, role="right",
                                 identifier=f"v{operation.inputs[1]}",
                                 name=_value_name(circuit, operation.inputs[1])))
        proof = {"kind": "add", "left": left_value, "right": right_value, "result": result}
    elif operation.opcode == OpCode.FROZEN_EXPAND:
        weight_index = operation.attributes[0]
        weight = tensors[weight_index]
        weight_row, weight_column = _weight_coordinate(weight, row, column)
        cards.append(_window(weight, weight_row, weight_column, role="frozen",
                             identifier=f"w{weight_index}", name=_weight_name(circuit, weight_index)))
        proof = {"kind": "expand", "source_index": [weight_row, weight_column], "result": result}
    elif operation.opcode == OpCode.ROW_CONCAT:
        remaining = row
        selected_input = 0
        source_row = 0
        for input_index, input_id in enumerate(operation.inputs):
            extent = values[input_id].shape[1]
            if remaining < extent:
                selected_input, source_row = input_index, remaining
                break
            remaining -= extent
        input_id = operation.inputs[selected_input]
        cards.append(_window(values[input_id], source_row, column, role="selected_input",
                             identifier=f"v{input_id}", name=_value_name(circuit, input_id)))
        proof = {"kind": "concat", "input": selected_input, "source_row": int(source_row),
                 "result": result}
    elif operation.opcode == OpCode.MATRIX_TRANSPOSE:
        source = values[operation.inputs[0]]
        cards.append(_window(source, column, row, role="input", identifier=f"v{operation.inputs[0]}",
                             name=_value_name(circuit, operation.inputs[0])))
        proof = {"kind": "transpose", "source_index": [0, column, row], "result": result}
    elif operation.opcode == OpCode.MATRIX_RESHAPE:
        source = values[operation.inputs[0]]
        flat_index = row * output.shape[2] + column
        source_row, source_column = divmod(flat_index, source.shape[2])
        cards.append(_window(source, source_row, source_column, role="input",
                             identifier=f"v{operation.inputs[0]}",
                             name=_value_name(circuit, operation.inputs[0])))
        proof = {"kind": "reshape", "flat_index": int(flat_index),
                 "source_index": [0, int(source_row), int(source_column)], "result": result}
    elif operation.opcode == OpCode.HARDMAX_STE:
        logits = values[operation.inputs[0]]
        winner = int(torch.argmax(logits[0, row]).item())
        cards.append(_window(logits, row, winner, role="logits_row",
                             identifier=f"v{operation.inputs[0]}",
                             name=_value_name(circuit, operation.inputs[0])))
        proof = {"kind": "argmax", "operator": "ARGMAX", "winner": winner,
                 "winner_value": _number(logits[0, row, winner]),
                 "selected_column": column, "result": result}
    elif operation.opcode == OpCode.HULL_ATTN_2D:
        query, keys, payload = (values[value] for value in operation.inputs)
        candidates = list(operation.attributes[2:])
        candidate_tensor = torch.tensor(candidates, dtype=torch.long)
        selected_keys = keys[0].index_select(0, candidate_tensor)
        scores = query[0, row] @ selected_keys.transpose(0, 1)
        winner_position = int(torch.argmax(scores).item())
        winner_row = candidates[winner_position]
        ranking = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))
        top = ranking[:min(8, len(ranking))]
        cards.extend((
            _window(query, row, 0, role="query_row", identifier=f"v{operation.inputs[0]}",
                    name=_value_name(circuit, operation.inputs[0])),
            _window(keys, 0, winner_row, role="key_column_transposed",
                    identifier=f"v{operation.inputs[1]}ᵀ",
                    name={"ru": "Транспонированные ключи Kᵀ", "en": "Transposed keys Kᵀ"},
                    transpose=True),
            _window(payload, winner_row, column, role="selected_value_row",
                    identifier=f"v{operation.inputs[2]}", name=_value_name(circuit, operation.inputs[2])),
        ))
        proof = {"kind": "attention", "operator": "Q × Kᵀ → ARGMAX → V",
                 "winner_candidate_position": winner_position, "winner_value_row": winner_row,
                 "score_terms": [
                     {"d": d, "query": _number(query[0, row, d]),
                      "key": _number(keys[0, winner_row, d]),
                      "product": _number(query[0, row, d] * keys[0, winner_row, d])}
                     for d in range(query.shape[2])
                 ],
                 "winner_score": _number(scores[winner_position]),
                 "top_candidates": [{"candidate_position": int(index),
                                     "value_row": int(candidates[index]),
                                     "score": _number(scores[index])} for index in top],
                 "result": result}
    else:  # pragma: no cover - guarded by the explicit runtime opcode test.
        raise ValueError(f"no website proof for {operation.opcode.name}")

    cards.append(output_card)
    return {
        "output": {"index": [0, row, column], "value": result, "window": output_card},
        "cards": cards,
        "proof": proof,
    }


def _context_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype="<f4").tobytes()).hexdigest()


def _status_name(context: np.ndarray) -> str:
    channel = int(np.argmax(context[1968]))
    return next(name for name, value in STATUS_CHANNELS.items() if value == channel)


def _context_summary(context: np.ndarray) -> dict[str, object]:
    history = context[:1200].reshape(400, 3, VOCAB_SIZE)
    legal = context[1200:1968].reshape(256, 3, VOCAB_SIZE)
    history_triples = np.argmax(history, axis=2)
    history_present = history_triples[:, 0] != 0
    legal_triples = np.argmax(legal, axis=2)
    legal_present = legal_triples[:, 0] != 0
    status_channel = int(np.argmax(context[1968]))
    return {
        "context_sha256": _context_hash(context),
        "status": _status_name(context),
        "status_channel": status_channel,
        "history_plies": int(history_present.sum()),
        "history_moves": history_triples[history_present].astype(int).tolist(),
        "legal_count": int(legal_present.sum()),
        "legal_moves": legal_triples[legal_present].astype(int).tolist(),
    }


def _source_digest() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("recurrent_circuit.py", "circuit.py", "legal_circuit.py",
                 "reference_executor.py", "recurrent_site_trace.py"):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_recurrent_site_trace() -> dict[str, object]:
    circuit, output_id = build_recurrent_artifact()
    artifact = circuit.artifact(output_id)
    prior = build_context_0()
    request = encode_move_one_hot(52, 54, PIECE_CHANNELS["WHITE_PAWN"])
    source = torch.from_numpy(np.concatenate((request, prior)).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        output, values, tensors = execute_artifact_reference_values(artifact, source)
    output_context = output[0].numpy()

    stage_records = [{
        "id": stage_id,
        "first_operation": first,
        "last_operation": last,
        "operation_count": last - first + 1,
        "ru": {"name": ru_name, "purpose": ru_purpose},
        "en": {"name": en_name, "purpose": en_purpose},
    } for stage_id, first, last, ru_name, en_name, ru_purpose, en_purpose in STAGES]

    operations = []
    for index, operation in enumerate(artifact.operations, 1):
        value_id = operation.outputs[0]
        semantic_key = circuit.semantics[value_id]["name"]
        stage = _stage_for(index)
        operator_ru, operator_en = OPERATOR_NAMES[operation.opcode]
        semantic_ru = _human_name(semantic_key, "ru")
        semantic_en = _human_name(semantic_key, "en")
        frozen_indices = []
        if operation.opcode in (OpCode.TOKEN_PROJECT, OpCode.OUTPUT_PROJECT,
                                OpCode.POSITION_ADD, OpCode.FROZEN_EXPAND):
            frozen_indices.append(operation.attributes[0])
        semantics = {
            "technical_id": f"v{value_id}",
            "semantic_key": semantic_key,
            "ru": {
                "name": semantic_ru,
                "purpose": f"{operator_ru.capitalize()} внутри этапа «{stage[3]}». "
                           f"Это tensor-операция, а не шахматная ветка C++.",
            },
            "en": {
                "name": semantic_en,
                "purpose": f"{operator_en.capitalize()} inside “{stage[4]}”. "
                           f"This is a tensor operation, not a C++ chess branch.",
            },
        }
        operations.append({
            "index": index,
            "opcode": operation.opcode.name,
            "stage": stage[0],
            "inputs": [f"v{value}" for value in operation.inputs],
            "input_shapes": [["B", *circuit.shapes[value]] for value in operation.inputs],
            "output": f"v{value_id}",
            "output_shape": ["B", *circuit.shapes[value_id]],
            "attributes": list(operation.attributes),
            "frozen": [{
                "id": f"w{weight}",
                "name": circuit.tensors[weight].name,
                "shape": list(circuit.tensors[weight].shape),
                "semantics": {
                    "ru": {"name": _weight_name(circuit, weight)["ru"],
                           "purpose": f"Неизменяемый операнд операции «{semantic_ru}»: {operator_ru}."},
                    "en": {"name": _weight_name(circuit, weight)["en"],
                           "purpose": f"Immutable operand of “{semantic_en}”: {operator_en}."},
                },
            } for weight in frozen_indices],
            "equation": _equation(operation),
            "semantics": semantics,
            "sample": _sample_operation(circuit, operation, values, tensors),
        })

    named_outputs = {}
    for name, value_id in circuit.named_outputs.items():
        tensor = values[value_id]
        named_outputs[name] = {
            "value": f"v{value_id}",
            "shape": ["B", *circuit.shapes[value_id]],
            "sample_0_0": _number(tensor[0, 0, 0]),
            "producer_operation": circuit.semantics[value_id]["producer"] + 1,
        }

    output_summary = _context_summary(output_context)
    first_attention = next(op for op in artifact.operations if op.opcode == OpCode.HULL_ATTN_2D)
    board = values[first_attention.outputs[0]][0]
    output_summary["board_piece_tokens"] = torch.argmax(board, dim=1).tolist()
    prior_summary = _context_summary(prior)

    return {
        "schema_version": 1,
        "artifact": "recurrent_frozen_vm_v1",
        "operation_count": len(artifact.operations),
        "tensor_count": len(artifact.tensors),
        "runtime_opcodes": sorted({operation.opcode.name for operation in artifact.operations}),
        "input": {
            "shape": ["B", 2048, 128],
            "sections": [
                {"rows": [0, 3], "meaning": "requested move [FROM, TO, RESULT_PIECE]"},
                {"rows": [3, 1203], "meaning": "previous history: 400 move triples"},
                {"rows": [1203, 1971], "meaning": "previous LEGAL_SET: 256 move triples"},
                {"rows": [1971, 1972], "meaning": "previous technical status"},
                {"rows": [1972, 2048], "meaning": "reserved service rows"},
            ],
        },
        "output": {
            "shape": ["B", 2045, 128],
            "sections": [
                {"rows": [0, 1200], "meaning": "next history"},
                {"rows": [1200, 1968], "meaning": "next LEGAL_SET"},
                {"rows": [1968, 1969], "meaning": "next technical status"},
                {"rows": [1969, 2045], "meaning": "reserved service rows"},
            ],
        },
        "final_output": f"v{output_id}",
        "fixture": {
            "request": [52, 54, PIECE_CHANNELS["WHITE_PAWN"]],
            "request_uci": "e2e4",
            "prior": prior_summary,
            "output": output_summary,
            "feedback": {
                "output_context_sha256": output_summary["context_sha256"],
                "next_input_context_sha256": output_summary["context_sha256"],
                "native_sequence_acceptance": (
                    "47 exact contexts; 40 output-to-input feedback edges; "
                    "5 context_0 bindings"
                ),
            },
        },
        "stages": stage_records,
        "named_outputs": named_outputs,
        "operations": operations,
        "provenance": {
            "numeric_executor": "vm_compiler.reference_executor",
            "native_intermediate_capture": False,
            "native_acceptance": {
                "hardware": "NVIDIA GeForce RTX 3070 Laptop GPU",
                "precision": "FP32; TF32 disabled",
                "contexts": 47,
                "feedback_edges": 40,
                "context_0_bindings": 5,
                "request_gradient_nonzero": 144,
                "sampled_global_peak_mib_with_backward": 3736,
                "elapsed_ms_with_backward": 56284,
            },
            "artifact_sha256": hashlib.sha256(artifact.to_bytes()).hexdigest(),
            "source_sha256": _source_digest(),
            "format_note": "Small exact windows/proofs for all operations; not all retained SSA cells.",
        },
    }


def main() -> None:
    destination = Path(__file__).parents[1] / "site" / "recurrent_trace.json"
    destination.write_text(
        json.dumps(build_recurrent_site_trace(), ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
