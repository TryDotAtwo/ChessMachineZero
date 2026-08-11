export type TraceStage = {
  id: number;
  short: string;
  title: string;
  equation: string;
  detail: string;
  activeTokens: string[];
};

export const TRACE: TraceStage[] = [
  { id: 0, short: "X", title: "Токены состояния", equation: "X ∈ ℝᴺˣ²⁵⁶", detail: "64 клетки, 64 кандидата, side, geometry и relation-токены образуют одну последовательность.", activeTokens: ["SIDE:white", "SQUARE:e2=WP", "CANDIDATE:e2→e3"] },
  { id: 1, short: "QKV", title: "Линейные проекции", equation: "Q=XWq · K=XWk · V=XWv", detail: "Фиксированные веса переводят поля токенов в пространство запросов, ключей и значений.", activeTokens: ["Q:candidate/e2", "K:geometry/e2/white", "V:target=e3"] },
  { id: 2, short: "QKᵀ", title: "Матрица совместимости", equation: "S=QKᵀ+M", detail: "Строка кандидата сравнивается сразу со всеми разрешёнными ключами. Маска M задаёт только тип связи.", activeTokens: ["row:CANDIDATE:e2", "col:RELATION:WP/W/VALID/EMPTY"] },
  { id: 3, short: "H", title: "Deterministic hardmax", equation: "H=one_hot(argmax(S))", detail: "На строку остаётся ровно один победитель. При равенстве берётся ключ с меньшим индексом.", activeTokens: ["winner:LEGAL=true", "tie-break:lowest-index"] },
  { id: 4, short: "AV", title: "Маршрутизация значения", equation: "Y=HV", detail: "One-hot attention переносит значение relation-токена в candidate-токен без host-ветвления.", activeTokens: ["CANDIDATE:e2.legal=true", "CANDIDATE:e2.target=e3"] },
  { id: 5, short: "RΔC", title: "Матричная запись", equation: "X′=X+R(Y−X)C", detail: "Фиксированные row/feature-проекции обновляют только разрешённые строки и поля.", activeTokens: ["WRITE:e2=EMPTY", "WRITE:e3=WHITE_PAWN"] },
  { id: 6, short: "↻", title: "Рекуррентное состояние", equation: "Xₜ₊₁=Transition(Xₜ)", detail: "Выходные square/side-токены становятся входом следующего inference тем же набором весов.", activeTokens: ["SQUARE:e2=EMPTY", "SQUARE:e3=WP", "SIDE:black"] },
];

export const TOKEN_ROWS = [
  { token: "SIDE", value: "WHITE", role: "очередь хода" },
  { token: "SQUARE:e2", value: "WHITE_PAWN", role: "источник" },
  { token: "GEOMETRY:e2/W", value: "TARGET=e3", role: "направление" },
  { token: "SQUARE:e3", value: "EMPTY", role: "цель" },
  { token: "RELATION", value: "LEGAL=TRUE", role: "предикат" },
  { token: "SELECTED", value: "e2→e3", role: "hardmax" },
];
