import type { MatrixData, MatrixName, NumericValue } from "./trace-schema";
import { valueAt } from "./matrix";

export const DEBUG_STEPS: Array<{matrix:MatrixName;label:string;action:string}> = [
  {matrix:"Q",label:"Запрос",action:"X·Wq — активный токен формирует вопрос"},
  {matrix:"K",label:"Ключи",action:"X·Wk — токены формируют адреса для сравнения"},
  {matrix:"S",label:"Scores",action:"Q·Kᵀ+M — считаются разрешённые совпадения"},
  {matrix:"A",label:"Hardmax",action:"Победивший score превращается в one-hot маршрут"},
  {matrix:"V",label:"Значения",action:"X·Wv — подготавливаются переносимые данные"},
  {matrix:"XPrime",label:"Запись",action:"A·V и R(Y−X)C формируют новое состояние"},
];

const explanations = [
 "Кандидат читает фигуру со своей исходной клетки.", "Текущая сторона передаётся всем кандидатам.",
 "Источник, сторона и тип хода выбирают фиксированный токен геометрии.", "Адрес цели читает содержимое целевой клетки.",
 "Для двойного хода отдельно читается промежуточная клетка.", "Relation-токены проверяют совпадение пешки и стороны.",
 "Relation-токены проверяют допустимость содержимого цели.", "Relation-токены проверяют свободный промежуточный путь.",
 "Пять предикатов сходятся в один токен LEGAL.", "Hardmax выбирает один легальный кандидат.",
 "Выбранный кандидат создаёт запись EMPTY для источника.", "Выбранный кандидат создаёт запись фигуры для цели.",
 "Каждая выходная клетка выбирает исходное значение либо legal-write.", "Сторона меняется только через выбранный LEGAL-токен.",
 "Выходные токены становятся входом следующего инференса с теми же весами.",
];
export const stageExplanation = (stage:number) => explanations[stage] ?? "Неизвестная стадия";
export const boardForStage = (stage:number) => ({pawn: stage >= 12 ? 20 : 12, from:12, to:20});

export type HeatCell={row:number;col:number;value:NumericValue;intensity:number};
export function heatmapWindow(matrix:MatrixData,row:number,col:number,size=12):HeatCell[]{
 const r0=Math.max(0,Math.min(row-Math.floor(size/2),matrix.rows-size));
 const c0=Math.max(0,Math.min(col-Math.floor(size/2),matrix.cols-size));
 const cells:HeatCell[]=[]; let max=0;
 for(let r=r0;r<Math.min(matrix.rows,r0+size);r++)for(let c=c0;c<Math.min(matrix.cols,c0+size);c++){
   const value=valueAt(matrix,r,c); if(typeof value==="number")max=Math.max(max,Math.abs(value)); cells.push({row:r,col:c,value,intensity:0});
 }
 return cells.map(cell=>({...cell,intensity:typeof cell.value==="number"&&max?Math.abs(cell.value)/max:0}));
}
