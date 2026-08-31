"""Bilingual, source-derived labels for the static artifact inspector."""

from __future__ import annotations

from .protocol import PIECE_CHANNELS
from .state_circuit import CASTLING_EVENTS, EMPTY, build_en_passant_event_patterns


def _square(channel: int) -> str:
    return f"{chr(ord('a') + channel // 10 - 1)}{channel % 10}"


def _piece(channel: int, language: str) -> str:
    en_names = {value: key.replace("_", " ").lower() for key, value in PIECE_CHANNELS.items()}
    ru_names = {
        96: "белая пешка", 97: "белый конь", 98: "белый слон", 99: "белая ладья",
        100: "белый ферзь", 101: "белый король", 102: "чёрная пешка", 103: "чёрный конь",
        104: "чёрный слон", 105: "чёрная ладья", 106: "чёрный ферзь", 107: "чёрный король",
    }
    names = ru_names if language == "ru" else en_names
    return names.get(channel, "EMPTY" if channel == EMPTY else ("padding token 0" if language == "en" else "padding token 0"))


def _text(en_name: str, ru_name: str, en_purpose: str, ru_purpose: str, rows: str, columns: str) -> dict[str, object]:
    return {
        "en": {"name": en_name, "purpose": en_purpose, "rows": rows, "columns": columns},
        "ru": {"name": ru_name, "purpose": ru_purpose, "rows": rows, "columns": columns},
    }


def _patterns() -> dict[str, list[dict[str, object]]]:
    castle = []
    for index, pattern in enumerate(CASTLING_EVENTS):
        source, destination, king, rook_source, _, rook_destination, rook = pattern
        castle.append({
            "column": index,
            "en": f"{_piece(king, 'en')} {_square(source)}→{_square(destination)}; rook {_square(rook_source)}→{_square(rook_destination)} ({_piece(rook, 'en')})",
            "ru": f"{_piece(king, 'ru')} {_square(source)}→{_square(destination)}; ладья {_square(rook_source)}→{_square(rook_destination)} ({_piece(rook, 'ru')})",
        })
    castle.append({"column": 4, "en": "no match / padding", "ru": "нет совпадения / padding"})

    ep = []
    for index, pattern in enumerate(build_en_passant_event_patterns()):
        source, destination, piece, previous_source, previous_destination, previous_piece, clear = pattern
        ep.append({
            "column": index,
            "en": f"{_piece(piece, 'en')} {_square(source)}→{_square(destination)}; previous {_piece(previous_piece, 'en')} {_square(previous_source)}→{_square(previous_destination)}; clear {_square(clear)}",
            "ru": f"{_piece(piece, 'ru')} {_square(source)}→{_square(destination)}; предыдущий ход: {_piece(previous_piece, 'ru')} {_square(previous_source)}→{_square(previous_destination)}; очистить {_square(clear)}",
        })
    ep.append({"column": 28, "en": "no match / padding", "ru": "нет совпадения / padding"})
    return {"castle": castle, "en_passant": ep}


def _records(items: list[tuple[str, str, str, str, str, str]]) -> dict[str, dict[str, object]]:
    return {key: {"technical_id": key, **_text(*texts)} for key, *texts in items}


def _localize_axes(records: dict[str, dict[str, object]]) -> None:
    translated = {
        "vocabulary token": "токен словаря", "vocabulary target token": "токен цели словаря",
        "piece/EMPTY vocabulary token": "токен фигуры/EMPTY", "piece vocabulary token": "токен фигуры",
        "2D key feature": "2D-признак ключа", "2D query feature": "2D-признак запроса",
        "compact square": "компактная клетка", "compact output square": "компактная выходная клетка",
        "history ply": "полуход истории", "previous ply": "предыдущий полуход", "current ply": "текущий полуход",
        "candidate event": "событие-кандидат", "initial event": "начальное событие",
        "input row": "строка входа", "aligned prehistory": "выровненная предыстория",
        "single aligned prehistory row": "одна выровненная строка предыстории",
        "castle pattern class": "класс паттерна рокировки", "en-passant pattern class": "класс паттерна en-passant",
        "output compact square (Q row)": "выходная компактная клетка (строка Q)",
        "candidate event (K transposed)": "событие-кандидат (K транспонирован)"}
    for record in records.values():
        record["ru"]["rows"] = translated.get(record["ru"]["rows"], record["ru"]["rows"])
        record["ru"]["columns"] = translated.get(record["ru"]["columns"], record["ru"]["columns"])


def build_site_semantics(artifact) -> dict[str, object]:
    """Return metadata for the exact current position artifact, never a UI copy."""
    if len(artifact.tensors) != 24 or len(artifact.operations) != 45:
        raise ValueError("site semantics require the 24-tensor, 45-operation position artifact")
    tensors = _records([
        ("w0", "latest square key projection", "Проекция ключа клетки", "For s=8*(file-1)+(rank-1), x=1+s/64: square token becomes (x,x²); non-square/padding becomes (0,0).", "Для s=8*(file-1)+(rank-1), x=1+s/64: токен клетки становится (x,x²); не-клетка/padding становится (0,0).", "vocabulary token", "2D key feature"),
        ("w1", "latest square queries", "Запросы клеток", "For compact s and x=1+s/64, each output-square query is (2x,-1).", "Для компактной s и x=1+s/64 запрос каждой выходной клетки равен (2x,-1).", "compact output square", "2D query feature"),
        ("w2", "initial square target tokens", "Начальные токены целей клеток", "Initial event targets; not initial board payloads.", "Цели начальных событий; не payload начальной доски.", "compact square", "vocabulary target token"),
        ("w3", "initial piece-state payload", "Начальный payload состояния фигур", "Initial one-hot piece or EMPTY payload per compact square.", "Начальный one-hot payload фигуры или EMPTY на компактную клетку.", "compact square", "piece/EMPTY vocabulary token"),
        ("w4", "source-clear payloads", "Payload очистки источника", "400 EMPTY payloads for routed source events.", "400 payload EMPTY для событий источника.", "history ply", "piece/EMPTY vocabulary token"),
        ("w5", "prehistory padding row", "Строка padding до истории", "One token-0 row prepended to previous-move streams; not a square or EMPTY event.", "Одна строка token-0 перед потоками предыдущего хода; не клетка и не EMPTY-событие.", "single aligned prehistory row", "vocabulary token"),
        ("w6", "castle FROM matcher", "Сопоставитель рокировки FROM", "Current FROM token projected to four castle classes plus no-match.", "Текущий токен FROM проецируется в четыре класса рокировки и no-match.", "vocabulary token", "castle pattern class"),
        ("w7", "castle TO matcher", "Сопоставитель рокировки TO", "Current TO token projected to castle pattern class.", "Текущий токен TO проецируется в класс паттерна рокировки.", "vocabulary token", "castle pattern class"),
        ("w8", "castle RESULT matcher", "Сопоставитель рокировки RESULT", "Current result-piece token projected to castle pattern class.", "Текущий токен RESULT_PIECE проецируется в класс паттерна рокировки.", "vocabulary token", "castle pattern class"),
        ("w9", "castle no-match bias", "Смещение no-match рокировки", "Fixed +2.5 only in no-match column 4: above two partial matches, below a complete three-slot match; not learned.", "Фиксированное +2.5 только в no-match столбце 4: выше двух частичных совпадений, ниже полного трёхслотового; не обучается.", "history ply", "castle pattern class"),
        ("w10", "castle rook source targets", "Цели очистки ладьи при рокировке", "Selected class decodes rook-source target; no-match decodes token 0.", "Выбранный класс декодирует цель источника ладьи; no-match декодирует token 0.", "castle pattern class", "vocabulary target token"),
        ("w11", "castle rook source payloads", "Payload очистки ладьи при рокировке", "Selected class decodes EMPTY at rook source; no-match is token 0.", "Выбранный класс декодирует EMPTY в источнике ладьи; no-match это token 0.", "castle pattern class", "piece/EMPTY vocabulary token"),
        ("w12", "castle rook destination targets", "Цели размещения ладьи при рокировке", "Selected class decodes rook destination; no-match decodes token 0.", "Выбранный класс декодирует назначение ладьи; no-match декодирует token 0.", "castle pattern class", "vocabulary target token"),
        ("w13", "castle rook destination payloads", "Payload размещения ладьи при рокировке", "Selected class decodes rook piece; no-match is token 0.", "Выбранный класс декодирует фигуру ладьи; no-match это token 0.", "castle pattern class", "piece vocabulary token"),
        ("w14", "en-passant current FROM matcher", "EP-сопоставитель текущего FROM", "Current FROM projected to 28 en-passant patterns plus no-match.", "Текущий FROM проецируется в 28 EP-паттернов и no-match.", "vocabulary token", "en-passant pattern class"),
        ("w15", "en-passant current TO matcher", "EP-сопоставитель текущего TO", "Current TO projected to en-passant pattern class.", "Текущий TO проецируется в класс EP-паттерна.", "vocabulary token", "en-passant pattern class"),
        ("w16", "en-passant current RESULT matcher", "EP-сопоставитель текущего RESULT", "Current RESULT_PIECE projected to en-passant pattern class.", "Текущий RESULT_PIECE проецируется в класс EP-паттерна.", "vocabulary token", "en-passant pattern class"),
        ("w17", "en-passant previous FROM matcher", "EP-сопоставитель предыдущего FROM", "Aligned previous FROM projected to en-passant pattern class.", "Выровненный предыдущий FROM проецируется в класс EP-паттерна.", "vocabulary token", "en-passant pattern class"),
        ("w18", "en-passant previous TO matcher", "EP-сопоставитель предыдущего TO", "Aligned previous TO projected to en-passant pattern class.", "Выровненный предыдущий TO проецируется в класс EP-паттерна.", "vocabulary token", "en-passant pattern class"),
        ("w19", "en-passant previous RESULT matcher", "EP-сопоставитель предыдущего RESULT", "Aligned previous RESULT_PIECE projected to en-passant pattern class.", "Выровненный предыдущий RESULT_PIECE проецируется в класс EP-паттерна.", "vocabulary token", "en-passant pattern class"),
        ("w20", "en-passant no-match bias", "Смещение EP no-match", "Fixed +5.5 only in no-match column 28: above five partial matches, below all six; not learned.", "Фиксированное +5.5 только в no-match столбце 28: выше пяти частичных совпадений, ниже всех шести; не обучается.", "history ply", "en-passant pattern class"),
        ("w21", "en-passant clear targets", "Цели очистки EP", "Selected pattern decodes captured-pawn target; no-match decodes token 0.", "Выбранный паттерн декодирует цель взятой пешки; no-match декодирует token 0.", "en-passant pattern class", "vocabulary target token"),
        ("w22", "en-passant clear payloads", "Payload очистки EP", "Selected pattern decodes EMPTY; no-match decodes token 0.", "Выбранный паттерн декодирует EMPTY; no-match декодирует token 0.", "en-passant pattern class", "piece/EMPTY vocabulary token"),
        ("w23", "latest event time bias", "Временное смещение последнего события", "(0,-epsilon*t), epsilon=2^-21: initial t=0, then t=1..400 in each of five streams; fixed, not learned.", "(0,-epsilon*t), epsilon=2^-21: начальное t=0, затем t=1..400 в каждом из пяти потоков; фиксировано, не обучается.", "candidate event", "2D key feature"),
    ])
    value_specs = [
        ("input token tensor", "Входной тензор токенов", "Request rows 0..2 are unused; history rows 3..1202 feed this subgraph.", "Строки запроса 0..2 не используются; строки истории 3..1202 питают этот подграф.", "input row", "vocabulary token"),
        ("current FROM history", "Текущий FROM истории", "Strided current source-token stream.", "Поток текущих токенов источника со stride.", "history ply", "vocabulary token"),
        ("current TO history", "Текущий TO истории", "Strided current destination-token stream.", "Поток текущих токенов назначения со stride.", "history ply", "vocabulary token"),
        ("current RESULT history", "Текущий RESULT истории", "Strided current result-piece-token stream.", "Поток текущих токенов RESULT_PIECE со stride.", "history ply", "vocabulary token"),
        ("expanded initial targets", "Расширенные начальные цели", "Batch expansion of initial event targets.", "Расширение в batch начальных целей событий.", "initial event", "vocabulary target token"),
        ("castle FROM logits", "Логиты рокировки FROM", "Per-ply castle class evidence from FROM.", "Доказательство класса рокировки из FROM для каждого ply.", "history ply", "castle pattern class"),
        ("castle TO logits", "Логиты рокировки TO", "Per-ply castle class evidence from TO.", "Доказательство класса рокировки из TO для каждого ply.", "history ply", "castle pattern class"),
        ("castle RESULT logits", "Логиты рокировки RESULT", "Per-ply castle class evidence from RESULT_PIECE.", "Доказательство класса рокировки из RESULT_PIECE для каждого ply.", "history ply", "castle pattern class"),
        ("castle partial sum", "Частичная сумма рокировки", "FROM plus TO class logits.", "Логиты классов FROM плюс TO.", "history ply", "castle pattern class"),
        ("castle full sum", "Полная сумма рокировки", "FROM, TO and RESULT class logits.", "Логиты классов FROM, TO и RESULT.", "history ply", "castle pattern class"),
        ("castle biased logits", "Логиты рокировки со смещением", "Full class evidence plus fixed no-match bias.", "Полное доказательство класса плюс фиксированное смещение no-match.", "history ply", "castle pattern class"),
        ("selected castle class", "Выбранный класс рокировки", "Hard one-hot selected castle pattern or no-match.", "Жёсткий one-hot выбранного паттерна рокировки или no-match.", "history ply", "castle pattern class"),
        ("castle source targets", "Цели источника ладьи", "Decoded rook-clear target token.", "Декодированный токен цели очистки ладьи.", "history ply", "vocabulary target token"),
        ("castle source payloads", "Payload источника ладьи", "Decoded rook-clear payload.", "Декодированный payload очистки ладьи.", "history ply", "piece/EMPTY vocabulary token"),
        ("castle destination targets", "Цели назначения ладьи", "Decoded rook-place target token.", "Декодированный токен цели размещения ладьи.", "history ply", "vocabulary target token"),
        ("castle destination payloads", "Payload назначения ладьи", "Decoded rook-place payload.", "Декодированный payload размещения ладьи.", "history ply", "piece vocabulary token"),
        ("previous FROM history", "Предыдущий FROM истории", "First 399 routed FROM rows before padding alignment.", "Первые 399 строк FROM до выравнивания padding.", "previous ply", "vocabulary token"),
        ("previous TO history", "Предыдущий TO истории", "First 399 routed TO rows before padding alignment.", "Первые 399 строк TO до выравнивания padding.", "previous ply", "vocabulary token"),
        ("previous RESULT history", "Предыдущий RESULT истории", "First 399 routed RESULT rows before padding alignment.", "Первые 399 строк RESULT до выравнивания padding.", "previous ply", "vocabulary token"),
        ("prehistory padding", "Padding до истории", "One token-0 row aligned before the first current ply.", "Одна строка token-0 перед первым текущим ply.", "aligned prehistory", "vocabulary token"),
        ("aligned previous FROM", "Выровненный предыдущий FROM", "Padding followed by previous FROM, aligned to current ply.", "Padding плюс предыдущий FROM, выровненный с текущим ply.", "current ply", "vocabulary token"),
        ("aligned previous TO", "Выровненный предыдущий TO", "Padding followed by previous TO, aligned to current ply.", "Padding плюс предыдущий TO, выровненный с текущим ply.", "current ply", "vocabulary token"),
        ("aligned previous RESULT", "Выровненный предыдущий RESULT", "Padding followed by previous RESULT, aligned to current ply.", "Padding плюс предыдущий RESULT, выровненный с текущим ply.", "current ply", "vocabulary token"),
        ("EP current FROM logits", "EP-логиты текущего FROM", "Current FROM evidence for 29 EP classes.", "Доказательство текущего FROM для 29 EP-классов.", "history ply", "en-passant pattern class"),
        ("EP current TO logits", "EP-логиты текущего TO", "Current TO evidence for 29 EP classes.", "Доказательство текущего TO для 29 EP-классов.", "history ply", "en-passant pattern class"),
        ("EP current RESULT logits", "EP-логиты текущего RESULT", "Current RESULT evidence for 29 EP classes.", "Доказательство текущего RESULT для 29 EP-классов.", "history ply", "en-passant pattern class"),
        ("EP previous FROM logits", "EP-логиты предыдущего FROM", "Aligned previous FROM evidence for 29 EP classes.", "Доказательство предыдущего FROM для 29 EP-классов.", "history ply", "en-passant pattern class"),
        ("EP previous TO logits", "EP-логиты предыдущего TO", "Aligned previous TO evidence for 29 EP classes.", "Доказательство предыдущего TO для 29 EP-классов.", "history ply", "en-passant pattern class"),
        ("EP previous RESULT logits", "EP-логиты предыдущего RESULT", "Aligned previous RESULT evidence for 29 EP classes.", "Доказательство предыдущего RESULT для 29 EP-классов.", "history ply", "en-passant pattern class"),
        ("EP sum after two slots", "Сумма EP после двух слотов", "Current FROM plus TO evidence.", "Доказательство текущих FROM плюс TO.", "history ply", "en-passant pattern class"),
        ("EP sum after three slots", "Сумма EP после трёх слотов", "Current triple evidence.", "Доказательство текущей тройки.", "history ply", "en-passant pattern class"),
        ("EP sum after four slots", "Сумма EP после четырёх слотов", "Current triple plus previous FROM.", "Текущая тройка плюс предыдущий FROM.", "history ply", "en-passant pattern class"),
        ("EP sum after five slots", "Сумма EP после пяти слотов", "Previous TO added.", "Добавлен предыдущий TO.", "history ply", "en-passant pattern class"),
        ("EP full sum", "Полная сумма EP", "Previous RESULT added; all six slots.", "Добавлен предыдущий RESULT; все шесть слотов.", "history ply", "en-passant pattern class"),
        ("EP biased logits", "EP-логиты со смещением", "Full evidence plus fixed no-match bias.", "Полное доказательство плюс фиксированное no-match смещение.", "history ply", "en-passant pattern class"),
        ("selected EP class", "Выбранный EP-класс", "Hard one-hot selected EP pattern or no-match.", "Жёсткий one-hot выбранного EP-паттерна или no-match.", "history ply", "en-passant pattern class"),
        ("EP clear targets", "Цели очистки EP", "Decoded captured-pawn target token.", "Декодированный токен цели взятой пешки.", "history ply", "vocabulary target token"),
        ("EP clear payloads", "Payload очистки EP", "Decoded EMPTY payload for captured pawn.", "Декодированный EMPTY payload взятой пешки.", "history ply", "piece/EMPTY vocabulary token"),
        ("candidate event targets", "Цели событий-кандидатов", "Six concatenated target streams: initial, source, destination, castle source, castle destination, EP.", "Шесть склеенных потоков целей: начальный, источник, назначение, источник рокировки, назначение рокировки, EP.", "candidate event", "vocabulary target token"),
        ("base candidate keys", "Базовые ключи кандидатов", "2D keys from event targets before chronology bias.", "2D-ключи из целей событий до временного смещения.", "candidate event", "2D key feature"),
        ("time-biased candidate keys", "Ключи кандидатов со временем", "2D keys after fixed chronology bias.", "2D-ключи после фиксированного временного смещения.", "candidate event", "2D key feature"),
        ("output-square queries", "Запросы выходных клеток", "One 2D query per compact output square.", "Один 2D-запрос на компактную выходную клетку.", "compact output square", "2D query feature"),
        ("initial board payload", "Начальный payload доски", "Batch-expanded initial board values.", "Расширенные в batch начальные значения доски.", "compact square", "piece/EMPTY vocabulary token"),
        ("source-clear payload", "Payload очистки источника", "Batch-expanded EMPTY values for source events.", "Расширенные в batch EMPTY-значения для событий источника.", "history ply", "piece/EMPTY vocabulary token"),
        ("candidate event payloads", "Payload событий-кандидатов", "Six concatenated payload streams, exactly aligned with v38 targets.", "Шесть склеенных потоков payload, точно выровненных с целями v38.", "candidate event", "piece/EMPTY vocabulary token"),
        ("reconstructed board", "Восстановленная доска", "Hard latest-event attention output; one piece/EMPTY token per compact square.", "Выход жёсткого внимания последнего события; один токен фигуры/EMPTY на компактную клетку.", "compact output square", "piece/EMPTY vocabulary token"),
    ]
    values = _records([(f"v{index}", *spec) for index, spec in enumerate(value_specs)])
    _localize_axes(tensors)
    _localize_axes(values)
    for index, operation in enumerate(artifact.operations, start=1):
        values[f"v{operation.outputs[0]}"]["producer"] = {"operation": index, "opcode": operation.opcode.name, "inputs": [f"v{value}" for value in operation.inputs]}
    values["v0"]["producer"] = {"kind": "input"}
    derived = {
        "op45_scores": {"technical_id": "op45_scores", **_text("Q × Kᵀ scores", "Оценки Q × Kᵀ", "Stored FP32 dot products Q[row] × K-transposed[column].", "Сохранённые FP32 скалярные произведения Q[строка] × K-транспонированный[столбец].", "output compact square (Q row)", "candidate event (K transposed)")},
        "op45_attention": {"technical_id": "op45_attention", **_text("hard forward attention", "Жёсткое внимание прямого прохода", "One-hot forward argmax over candidate events; selected-top-8 surrogate is backward-only.", "One-hot argmax прямого прохода по событиям; selected-top-8 surrogate только для backward.", "output compact square", "candidate event")},
    }
    _localize_axes(derived)
    return {
        "tensors": tensors,
        "values": values,
        "derived": derived,
        "event_streams": [
            {"start": 0, "end": 64, "en": "initial", "ru": "начальные"},
            {"start": 64, "end": 464, "en": "source clear", "ru": "очистка источника"},
            {"start": 464, "end": 864, "en": "destination piece", "ru": "фигура назначения"},
            {"start": 864, "end": 1264, "en": "castle rook source", "ru": "источник ладьи рокировки"},
            {"start": 1264, "end": 1664, "en": "castle rook destination", "ru": "назначение ладьи рокировки"},
            {"start": 1664, "end": 2064, "en": "en-passant clear", "ru": "очистка en-passant"},
        ],
        "patterns": _patterns(),
    }
