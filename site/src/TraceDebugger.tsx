import { useEffect, useMemo, useState } from "react";
import { multiplyCell, scoreCell, valueAt } from "./matrix";
import type { InferenceTrace, MatrixData, MatrixName, NumericValue } from "./trace-schema";

const steps: { matrix: MatrixName; title: string; note: string }[] = [
  { matrix: "X", title: "Токены состояния", note: "Строки X — единственный вход текущей стадии." },
  { matrix: "Q", title: "Запрос Q = X × Wq", note: "Фиксированные веса проецируют токены в запросы." },
  { matrix: "K", title: "Ключи K = X × Wk", note: "Ключи определяют, с чем сравнивается запрос." },
  { matrix: "S", title: "Scores = Q × Kᵀ + M", note: "Каждый score — точное скалярное произведение плюс маска." },
  { matrix: "A", title: "Hardmax", note: "В каждой строке остаётся один победивший столбец." },
  { matrix: "Y", title: "Перенос AV", note: "Выбранная строка V переносится в выход Y." },
  { matrix: "XPrime", title: "Matrix-write", note: "Выход записывается обратно в токены состояния." },
];

const fmt = (v: NumericValue | number) => v === "-inf" || v === Number.NEGATIVE_INFINITY ? "−∞" : Number.isInteger(v) ? String(v) : Number(v).toPrecision(3);

function Board({after}:{after:boolean}) {
  return <div className="lab-board" aria-label={after ? "Доска после записи" : "Доска до записи"}>
    {Array.from({length:64},(_,visual)=>{const rank=7-Math.floor(visual/8), file=visual%8, square=rank*8+file, pawn=after?20:12; return <div key={square} className={`${(rank+file)%2?"dark":"light"} ${square===12?"from":""} ${square===20?"to":""}`}><span>{square===pawn?"♙":""}</span><small>{"abcdefgh"[file]}{rank+1}</small></div>})}
  </div>;
}

function rowValues(matrix: MatrixData, row: number, limit=8) {
  const safe=Math.min(row,matrix.rows-1); return Array.from({length:Math.min(limit,matrix.cols)},(_,col)=>valueAt(matrix,safe,col));
}

function MiniMatrix({name,matrix,row,active=false}:{name:string;matrix:MatrixData;row:number;active?:boolean}) {
  return <div className={`numeric-matrix ${active?"active":""}`}><header><b>{name}</b><span>{matrix.rows}×{matrix.cols}</span></header><div className="bracket-row">{rowValues(matrix,row).map((v,i)=><i key={i} className={v!==0&&v!=="-inf"?"hot":""}>{fmt(v)}</i>)}</div><small>строка {Math.min(row,matrix.rows-1)} · первые {Math.min(8,matrix.cols)} столбцов</small></div>;
}

function FullMatrix({matrix,row,col,onCell}:{matrix:MatrixData;row:number;col:number;onCell(r:number,c:number):void}) {
  const rs=Array.from({length:Math.min(10,matrix.rows)},(_,i)=>Math.max(0,Math.min(matrix.rows-10,row-4))+i);
  const cs=Array.from({length:Math.min(10,matrix.cols)},(_,i)=>Math.max(0,Math.min(matrix.cols-10,col-4))+i);
  return <div className="full-matrix"><header><b>{matrix.name}</b><span>{matrix.rows}×{matrix.cols} · {matrix.storage}</span></header><div className="matrix-window">{rs.flatMap(r=>cs.map(c=>{const v=valueAt(matrix,r,c);return <button key={`${r}-${c}`} onClick={()=>onCell(r,c)} className={`${r===row&&c===col?"selected":""} ${v!==0&&v!=="-inf"?"nonzero":""}`} title={`${matrix.name}[${r},${c}] = ${fmt(v)}`}>{fmt(v)}</button>}))}</div></div>;
}

export function NativeTraceLaboratory({trace,inspectorTitle}:{trace:InferenceTrace;inspectorTitle:string}) {
  const [stage,setStage]=useState(0),[micro,setMicro]=useState(3),[playing,setPlaying]=useState(false),[mode,setMode]=useState<"simple"|"numbers"|"full">("numbers"),[row,setRow]=useState(13),[col,setCol]=useState(21);
  useEffect(()=>{if(!playing)return;const id=setInterval(()=>setMicro(value=>{if(value<steps.length-1)return value+1;if(stage<trace.stages.length-1){setStage(stage+1);return 0}setPlaying(false);return value}),850);return()=>clearInterval(id)},[playing,stage,trace.stages.length]);
  const s=trace.stages[stage], step=steps[micro], matrix=s.matrices[step.matrix], safeRow=Math.min(row,matrix.rows-1), safeCol=Math.min(col,matrix.cols-1), winner=s.winners[Math.min(row,s.winners.length-1)]??0;
  const calc=useMemo(()=>step.matrix==="S"?scoreCell(s.matrices.Q,s.matrices.K,s.matrices.M,Math.min(row,s.matrices.S.rows-1),Math.min(col,s.matrices.S.cols-1)):step.matrix==="Q"?multiplyCell(s.matrices.X,s.matrices.Wq,Math.min(row,s.matrices.Q.rows-1),Math.min(col,s.matrices.Q.cols-1)):null,[s,step.matrix,row,col]);
  const next=()=>setMicro(v=>Math.min(steps.length-1,v+1)), prev=()=>setMicro(v=>Math.max(0,v-1));
  return <main className="laboratory">
    <header className="topbar"><div className="brand"><span>♞</span><div><b>ChessMachineZero</b><small>Сделано на идее Percepta</small></div></div><div className="claim"><strong>Шахматы как чистый инференс</strong><span>проверяемая native-трасса · без логики правил во frontend</span></div><a href="https://github.com/TryDotAtwo/ChessMachineZero/tree/codex/percepta-transformer-vm">GitHub ↗</a></header>
    <div className="truthbar"><b>Честная граница:</b> показан реализованный pawn-rule slice e2→e3; полные шахматы ещё не реализованы. UI только читает экспортированные X/W/Q/K/S/A/V/Y/X′.</div>
    <section className="toolbar"><div className="modes">{([['simple','Просто'],['numbers','Числа'],['full','Полная трасса']] as const).map(([id,label])=><button key={id} className={mode===id?"active":""} onClick={()=>setMode(id)}>{label}</button>)}</div><div className="transport"><button onClick={prev}>←</button><button className="play" onClick={()=>setPlaying(!playing)}>{playing?"Ⅱ Пауза":"▶ Воспроизвести"}</button><button onClick={next}>Следующий микрошаг →</button><b>{stage+1}/15 · {micro+1}/7</b></div></section>
    <section className="lab-grid">
      <aside className="board-panel"><small className="eyebrow">NATIVE TRACE · STAGE {String(stage).padStart(2,"0")}</small><h1>{s.name}</h1><p>{step.note}</p><Board after={stage===14&&step.matrix==="XPrime"}/><div className="board-key"><span><i className="yellow"/>источник e2 · token row 13</span><span><i className="cyan"/>цель e3 · token row 21</span></div><div className="token-card"><b>Активный токен</b><code>row {row} · {trace.tokenLabels[row]??"state token"}</code><span>Эта же строка подсвечена во всех матрицах справа.</span></div></aside>
      <article className="trace-panel"><div className="trace-title"><div><small>МИКРОШАГ {micro+1}</small><h2>{step.title}</h2></div><span className="live">● реальные float64</span></div>
        <div className="pipeline-line">{steps.map((x,i)=><button key={x.matrix} className={i===micro?"active":i<micro?"done":""} onClick={()=>setMicro(i)}><b>{x.matrix}</b><small>{i+1}</small></button>)}</div>
        {mode==="simple"?<div className="simple-explain"><b>Что произошло</b><p>{step.note}</p><p>Ни одно шахматное условие здесь не вычисляется JavaScript-кодом сайта: результат уже находится в native-матрицах.</p></div>:<>
          <section className="matrix-equation"><MiniMatrix name="X" matrix={s.matrices.X} row={row}/><em>×</em><MiniMatrix name="Wq" matrix={s.matrices.Wq} row={Math.min(col,s.matrices.Wq.rows-1)}/><em>=</em><MiniMatrix name="Q" matrix={s.matrices.Q} row={row} active={step.matrix==="Q"}/></section>
          <section className="matrix-equation score-equation"><MiniMatrix name="Q" matrix={s.matrices.Q} row={row}/><em>×</em><MiniMatrix name="Kᵀ" matrix={s.matrices.K} row={Math.min(col,s.matrices.K.rows-1)}/><em>+</em><MiniMatrix name="M" matrix={s.matrices.M} row={row}/><em>=</em><MiniMatrix name="S" matrix={s.matrices.S} row={row} active={step.matrix==="S"}/></section>
          <section className="arithmetic"><div><small>{inspectorTitle}</small><h3>{step.matrix}[{safeRow},{safeCol}] = {fmt(valueAt(matrix,safeRow,safeCol))}</h3>{calc?<code>{calc.terms.length?calc.terms.slice(0,6).map(t=>`${fmt(t.left)}×${fmt(t.right)}`).join(" + "):"0"} = {fmt(calc.sum)}</code>:<code>значение прочитано из native trace artifact</code>}</div><div className="hardmax"><small>HARDMAX ROW {Math.min(row,s.winners.length-1)}</small><b>winning key = {winner}</b><span>остальные scores отбрасываются</span></div></section>
          <section className="matrix-equation output"><MiniMatrix name="A" matrix={s.matrices.A} row={row}/><em>×</em><MiniMatrix name="V" matrix={s.matrices.V} row={winner}/><em>=</em><MiniMatrix name="AV = Y" matrix={s.matrices.Y} row={row}/><em>→</em><MiniMatrix name="X′" matrix={s.matrices.XPrime} row={row} active={step.matrix==="XPrime"}/></section>
        </>}
        {mode==="full"&&<section className="full-explorer"><div className="full-head"><div><small>ВСЕ КООРДИНАТЫ ДОСТУПНЫ</small><h3>Полная матрица {step.matrix}</h3></div><label>row <input type="number" min="0" max={matrix.rows-1} value={safeRow} onChange={e=>setRow(+e.target.value)}/></label><label>col <input type="number" min="0" max={matrix.cols-1} value={safeCol} onChange={e=>setCol(+e.target.value)}/></label></div><FullMatrix matrix={matrix} row={safeRow} col={safeCol} onCell={(r,c)=>{setRow(r);setCol(c)}}/></section>}
      </article>
    </section>
    <section className="stage-strip">{trace.stages.map(x=><button key={x.index} className={stage===x.index?"active":""} onClick={()=>{setStage(x.index);setMicro(0)}}><b>{String(x.index).padStart(2,"0")}</b><span>{x.name}</span></button>)}</section>
    <footer><span>schema {trace.schema}</span><span>commit {trace.provenance.commit.slice(0,12)}</span><span>{trace.provenance.generator}</span></footer>
  </main>;
}
