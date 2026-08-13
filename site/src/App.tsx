import { useEffect, useMemo, useState } from "react";
import { parseInferenceTrace, REQUIRED_MATRICES, type InferenceTrace, type MatrixName } from "./trace-schema";
import { multiplyCell, scoreCell, valueAt } from "./matrix";

const operations: MatrixName[] = [...REQUIRED_MATRICES];
const fmt = (v: number | "-inf") => v === "-inf" ? "−∞" : Number.isInteger(v) ? String(v) : v.toPrecision(5);

function MatrixGrid({ trace, stage, name, row, col, onCell }: { trace: InferenceTrace; stage: number; name: MatrixName; row: number; col: number; onCell(r:number,c:number):void }) {
  const m = trace.stages[stage].matrices[name];
  const rowStart = Math.max(0, Math.min(row - 7, m.rows - 16));
  const colStart = Math.max(0, Math.min(col - 7, m.cols - 16));
  const rows = Array.from({length: Math.min(16, m.rows - rowStart)}, (_,i)=>rowStart+i);
  const cols = Array.from({length: Math.min(16, m.cols - colStart)}, (_,i)=>colStart+i);
  return <div className="matrix-pane">
    <div className="matrix-meta"><strong>{name}</strong><span>{m.rows} × {m.cols}</span><span>{m.storage.toUpperCase()}</span></div>
    <div className="matrix-scroll"><table className="matrix"><thead><tr><th>r\c</th>{cols.map(c=><th key={c}>{c}</th>)}</tr></thead><tbody>
      {rows.map(r=><tr key={r}><th>{r}</th>{cols.map(c=>{const v=valueAt(m,r,c); return <td key={c}><button className={`${r===row&&c===col?"selected":""} ${v!==0&&v!=="-inf"?"nonzero":""}`} onClick={()=>onCell(r,c)}>{fmt(v)}</button></td>})}</tr>)}
    </tbody></table></div>
    <div className="matrix-nav"><label>row <input type="number" min="0" max={m.rows-1} value={row} onChange={e=>onCell(Math.min(m.rows-1,+e.target.value),col)}/></label><label>col <input type="number" min="0" max={m.cols-1} value={col} onChange={e=>onCell(row,Math.min(m.cols-1,+e.target.value))}/></label><span>Окно 16×16; координатами доступна любая из {m.rows*m.cols} ячеек.</span></div>
  </div>;
}

function Inspector({trace,stage,name,row,col}:{trace:InferenceTrace;stage:number;name:MatrixName;row:number;col:number}) {
  const s=trace.stages[stage], m=s.matrices[name];
  const r=Math.min(row,m.rows-1), c=Math.min(col,m.cols-1);
  let formula="Значение взято непосредственно из native trace artifact.";
  let terms: ReturnType<typeof multiplyCell>["terms"] = [];
  let sum=valueAt(m,r,c);
  if(name==="Q"||name==="K"||name==="V") { const left=s.matrices.X; const right=s.matrices[name==="Q"?"Wq":name==="K"?"Wk":"Wv"]; const x=multiplyCell(left,right,r,c); terms=x.terms; sum=x.sum; formula=`${name}[${r},${c}] = Σ X[${r},k] × W${name.toLowerCase()}[k,${c}]`; }
  if(name==="S") { const x=scoreCell(s.matrices.Q,s.matrices.K,s.matrices.M,r,c); terms=x.terms; sum=x.sum; formula=`S[${r},${c}] = Σ Q[${r},k] × K[${c},k] + M[${r},${c}]`; }
  const winner=s.winners[Math.min(r,s.winners.length-1)];
  return <aside className="inspector"><h2>Арифметика ячейки</h2><code>{formula}</code><div className="result"><span>trace</span><b>{fmt(valueAt(m,r,c))}</b><span>recomputed</span><b>{typeof sum==="number"?fmt(sum):fmt(sum)}</b></div>{terms.length>0?<ol>{terms.map(t=><li key={t.k}><code>k={t.k}: {fmt(t.left)} × {fmt(t.right)} = {fmt(t.product)}</code></li>)}</ol>:<p>Ненулевых произведений нет.</p>}<div className="winner"><span>hardmax row {r}</span><strong>winning key = {winner}</strong><small>При равенстве выбирается минимальный индекс.</small></div></aside>
}

export default function App(){
 const [trace,setTrace]=useState<InferenceTrace|null>(null),[error,setError]=useState(""); const [stage,setStage]=useState(0),[name,setName]=useState<MatrixName>("S"),[row,setRow]=useState(0),[col,setCol]=useState(0);
 useEffect(()=>{fetch(`${import.meta.env.BASE_URL}traces/pawn-e2-e3.json`).then(r=>r.json()).then(x=>setTrace(parseInferenceTrace(x))).catch(e=>setError(String(e)))},[]);
 const current=trace?.stages[stage]; const changed=useMemo(()=>current?.changed.length??0,[current]);
 if(error)return <main className="loading">Ошибка трассы: {error}</main>; if(!trace)return <main className="loading">Загрузка 17 MB реальной native-трассы…</main>;
 return <main className="workspace"><header><div><strong>ChessMachineZero / Matrix Trace</strong><small>pawn-rule slice · полные шахматы ещё не реализованы</small></div><a href="https://github.com/TryDotAtwo/ChessMachineZero/tree/codex/percepta-transformer-vm">исходный код ↗</a></header>
 <section className="provenance"><span>schema {trace.schema}</span><span>commit {trace.provenance.commit.slice(0,12)}</span><span>float64</span><span>native exporter</span></section>
 <nav className="stages">{trace.stages.map(s=><button className={s.index===stage?"active":""} onClick={()=>{setStage(s.index);setRow(0);setCol(0)}} key={s.index}><b>{String(s.index).padStart(2,"0")}</b><span>{s.name}</span></button>)}</nav>
 <section className="pipeline">{operations.map((op,i)=><button className={op===name?"active":""} onClick={()=>{setName(op);setRow(0);setCol(0)}} key={op}><span>{op}</span><small>{trace.stages[stage].matrices[op].rows}×{trace.stages[stage].matrices[op].cols}</small>{i<operations.length-1&&<i>→</i>}</button>)}</section>
 <section className="status"><div><small>STAGE</small><b>{stage} / 14</b></div><div><small>MATRIX</small><b>{name}</b></div><div><small>SELECTED CELL</small><b>[{row},{col}] = {fmt(valueAt(current!.matrices[name],Math.min(row,current!.matrices[name].rows-1),Math.min(col,current!.matrices[name].cols-1)))}</b></div><div><small>CHANGED CELLS</small><b>{changed}</b></div></section>
 <section className="main-grid"><MatrixGrid trace={trace} stage={stage} name={name} row={row} col={col} onCell={(r,c)=>{setRow(r);setCol(c)}}/><Inspector trace={trace} stage={stage} name={name} row={row} col={col}/></section>
 <section className="writes"><h2>Последовательная matrix-write трасса</h2>{current!.writes.map(w=><div key={w.component}><b>component {w.component}</b><code>X{w.component+1} = X{w.component} + R{w.component}(Y − X{w.component})C{w.component}</code><span>R {w.R.rows}×{w.R.cols}</span><span>C {w.C.rows}×{w.C.cols}</span></div>)}</section>
 </main>
}
