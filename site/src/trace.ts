export type TraceStage = {
  id: number;
  short: string;
  title: string;
  equation: string;
  detail: string;
  activeTokens: string[];
};

export const TRACE: TraceStage[] = [
  { id: 0, short: "SRC", title: "Источник", equation: "H(QKᵀ)V", detail: "Каждый из 128 candidate-токенов забирает фигуру со своей исходной клетки.", activeTokens: ["SQUARE:e2=WP", "CANDIDATE:e2→e4"] },
  { id: 1, short: "SIDE", title: "Очередь хода", equation: "H(QKᵀ)V", detail: "Текущий side-токен маршрутизируется во все кандидаты теми же матричными операциями.", activeTokens: ["SIDE:white", "CANDIDATE.side=white"] },
  { id: 2, short: "GEO", title: "Геометрия", equation: "H(QKᵀ)V", detail: "Ключ source/side/kind выбирает frozen geometry-токен: target, промежуточную клетку и стартовую горизонталь.", activeTokens: ["KIND:double", "TARGET:e4", "MID:e3", "START=true"] },
  { id: 3, short: "DST", title: "Целевая клетка", equation: "H(QKᵀ)V", detail: "Target-address через hardmax attention читает содержимое целевой клетки.", activeTokens: ["TARGET:e4", "SQUARE:e4=EMPTY"] },
  { id: 4, short: "MID", title: "Промежуточная клетка", equation: "H(QKᵀ)V", detail: "Отдельный address-токен читает e3; для одинарного хода маршрут уходит в нейтральный offboard-токен.", activeTokens: ["MID:e3", "SQUARE:e3=EMPTY"] },
  { id: 5, short: "P₁", title: "Своя пешка", equation: "P₁=lookup(piece,side)", detail: "Relation-токены выдают MATCHING_PAWN без шахматного условия в runtime.", activeTokens: ["WP", "WHITE", "MATCHING_PAWN=true"] },
  { id: 6, short: "P₂", title: "Пустая цель", equation: "P₂=lookup(target)", detail: "Таблица relation-токенов преобразует тип фигуры на цели в TARGET_EMPTY.", activeTokens: ["TARGET=EMPTY", "TARGET_EMPTY=true"] },
  { id: 7, short: "P₃", title: "Свободный путь", equation: "P₃=lookup(kind,mid)", detail: "Для double-кандидата PATH_CLEAR истинен только при пустой промежуточной клетке.", activeTokens: ["DOUBLE", "MID=EMPTY", "PATH_CLEAR=true"] },
  { id: 8, short: "AND", title: "Предикат легальности", equation: "LEGAL=lookup(P₁…P₅)", detail: "Hardmax выбирает точную строку таблицы пяти булевых предикатов.", activeTokens: ["MATCH=true", "EMPTY=true", "CLEAR=true", "GEOMETRY=true", "START=true", "LEGAL=true"] },
  { id: 9, short: "SEL", title: "Выбор хода", equation: "one_hot(argmax(LEGAL))", detail: "Детерминированный hardmax выбирает минимальный индекс среди legal-кандидатов.", activeTokens: ["128 candidates", "winner:e2→e3"] },
  { id: 10, short: "Wₛ", title: "Запись источника", equation: "X′=X+R(Y−X)C", detail: "Selected-токен создаёт write-токен EMPTY для исходной клетки.", activeTokens: ["WRITE:e2=EMPTY", "LEGAL=true"] },
  { id: 11, short: "Wₜ", title: "Запись цели", equation: "X′=X+R(Y−X)C", detail: "Второй write-токен несёт пешку и target-address выбранного хода.", activeTokens: ["WRITE:e3=WP", "LEGAL=true"] },
  { id: 12, short: "BOARD", title: "Новая доска", equation: "H(QKᵀ)V", detail: "Каждая output-клетка выбирает unchanged input либо точный legal-write.", activeTokens: ["e2=EMPTY", "e3=WP"] },
  { id: 13, short: "TURN", title: "Новая очередь", equation: "H(QKᵀ)V", detail: "Side меняется только если выбранный токен несёт LEGAL=true.", activeTokens: ["LEGAL=true", "SIDE:black"] },
  { id: 14, short: "↻", title: "Рекуррентное состояние", equation: "Xₜ₊₁=Transition(Xₜ)", detail: "Output square/side-токены становятся входом следующего inference с теми же frozen weights.", activeTokens: ["BOARDₜ₊₁", "SIDE:black", "same weights"] },
];

export const TOKEN_ROWS = [
  { token: "SIDE", value: "WHITE", role: "очередь хода" },
  { token: "SQUARE:e2", value: "WHITE_PAWN", role: "источник" },
  { token: "CANDIDATE:e2→e4", value: "DOUBLE", role: "кандидат" },
  { token: "GEOMETRY:e2/W/DOUBLE", value: "TARGET=e4 · MID=e3", role: "геометрия" },
  { token: "MATCHING_PAWN", value: "TRUE", role: "предикат" },
  { token: "TARGET_EMPTY", value: "TRUE", role: "предикат" },
  { token: "PATH_CLEAR", value: "TRUE", role: "предикат" },
  { token: "START_RANK", value: "TRUE", role: "предикат" },
  { token: "LEGAL", value: "TRUE", role: "конъюнкция" },
  { token: "SELECTED", value: "e2→e3", role: "hardmax" },
  { token: "WRITE_SOURCE", value: "e2=EMPTY", role: "matrix write" },
  { token: "WRITE_TARGET", value: "e3=WHITE_PAWN", role: "matrix write" },
  { token: "OUTPUT_BOARD", value: "e2=EMPTY · e3=WP", role: "результат" },
  { token: "OUTPUT_SIDE", value: "BLACK", role: "результат" },
  { token: "RECURRENT", value: "Xₜ₊₁", role: "следующий inference" },
];
