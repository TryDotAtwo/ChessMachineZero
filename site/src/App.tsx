import { useEffect, useMemo, useState } from "react";
import { ChevronRight, GitBranch, Pause, Play, RotateCcw, StepForward } from "lucide-react";
import { TOKEN_ROWS, TRACE } from "./trace";

const files = ["a", "b", "c", "d", "e", "f", "g", "h"];

function Board({ moved, active }: { moved: boolean; active: boolean }) {
  const pawn = moved ? 20 : 12;
  return <div className={`board-wrap ${active ? "board-active" : ""}`}>
    <div className="board" aria-label={moved ? "Доска после e2-e3" : "Доска до e2-e3"}>
      {Array.from({ length: 64 }, (_, visual) => {
        const rankFromTop = Math.floor(visual / 8);
        const file = visual % 8;
        const logical = (7 - rankFromTop) * 8 + file;
        return <div className={`square ${(rankFromTop + file) % 2 ? "dark" : "light"}`} key={logical}>
          {logical === pawn && <span className="pawn">♙</span>}
          {rankFromTop === 7 && <small className="file">{files[file]}</small>}
          {file === 0 && <small className="rank">{8 - rankFromTop}</small>}
        </div>;
      })}
    </div>
    <div className="board-label">{moved ? "Xₜ₊₁ · e3" : "Xₜ · e2"}</div>
  </div>;
}

function MiniMatrix({ name, stage, offset }: { name: string; stage: number; offset: number }) {
  return <div className="matrix-block active">
    <div className="matrix-title"><span>{name}</span><small>64 × d</small></div>
    <div className="mini-matrix">
      {Array.from({ length: 48 }, (_, i) => <i key={i} style={{
        opacity: .12 + (((i * 7 + offset) % 11) / 15),
        animationDelay: `${(i % 8) * 45}ms`,
      }} />)}
    </div>
  </div>;
}

function ScoreMatrix({ stage }: { stage: number }) {
  const winner = 27;
  return <div className="score-block active">
    <div className="matrix-title"><span>QKᵀ</span><small>N × N</small></div>
    <div className="score-matrix">
      {Array.from({ length: 64 }, (_, i) => <i key={i} className={i === winner ? "winner" : ""}
        style={{ opacity: i === winner ? 1 : .12 + (((i * 13) % 17) / 28) }} />)}
      <span className="scan-line" />
    </div>
  </div>;
}

function InferenceCanvas({ stage }: { stage: number }) {
  return <div className="inference-canvas">
    <Board moved={false} active={stage === 0} />
    <div className={`token-stream ${stage === 0 ? "active" : ""}`}>
      {["SIDE", "e2", "e3", "REL"].map((x, i) => <span key={x} style={{ animationDelay: `${i * 180}ms` }}>{x}</span>)}
      <ChevronRight />
    </div>
    <div className="matrix-cluster">
      <MiniMatrix name="Q" stage={stage} offset={1} />
      <MiniMatrix name="K" stage={stage} offset={4} />
      <MiniMatrix name="V" stage={stage} offset={8} />
    </div>
    <div className="flow-arrow"><span /></div>
    <ScoreMatrix stage={stage} />
    <div className="hardmax active">
      <small>hardmax</small>{Array.from({ length: 8 }, (_, i) => <i className={i === 3 ? "on" : ""} key={i} />)}
    </div>
    <div className="av-node active">AV</div>
    <Board moved={stage >= 12} active={stage >= 12} />
  </div>;
}

function App() {
  const [stage, setStage] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [traceMode, setTraceMode] = useState<"tokens" | "matrix">("tokens");
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => setStage(s => (s + 1) % TRACE.length), 1500);
    return () => window.clearInterval(id);
  }, [playing]);
  const current = TRACE[stage];
  const selectedCells = useMemo(() => new Set([9, 18, 27, 36]), []);

  return <>
    <header>
      <a className="brand" href="#top" aria-label="Percepta Chess"><span>♞</span> Percepta Chess</a>
      <nav><a href="#idea">Идея</a><a href="#matrices">Матрицы</a><a href="#trace">Трасса</a><a href="#proof">Доказательства</a></nav>
      <a className="github-link" href="https://github.com/TryDotAtwo/ChessMachineZero/tree/codex/percepta-transformer-vm"><GitBranch size={17}/> GitHub</a>
    </header>
    <main id="top">
      <section className="hero" id="idea">
        <div className="hero-copy">
          <h1>Шахматы как<br/><em>чистый инференс</em></h1>
          <p>Токены <b>→</b> Q/K/V <b>→</b> QKᵀ <b>→</b> hardmax <b>→</b> AV <b>→</b> новое состояние доски</p>
          <div className="actions">
            <button onClick={() => setPlaying(v => !v)}>{playing ? <Pause/> : <Play/>}{playing ? "Пауза" : "Запустить ход"}</button>
            <button className="secondary" onClick={() => { setStage(s => (s + 1) % TRACE.length); setPlaying(false); }}><StepForward/> Один шаг</button>
            <a href="https://github.com/TryDotAtwo/ChessMachineZero/tree/codex/percepta-transformer-vm/native/vm2">Открыть код <ChevronRight/></a>
          </div>
        </div>
        <div className="stage-readout"><span>0{stage + 1}</span><div><small>АКТИВНАЯ СТАДИЯ</small><strong>{current.title}</strong></div></div>
        <InferenceCanvas stage={stage}/>
        <div className="stage-rail" role="tablist" aria-label="Стадии инференса">
          {TRACE.map((item, i) => <button key={item.id} className={i === stage ? "active" : ""} onClick={() => { setStage(i); setPlaying(false); }}><span>{item.short}</span><small>{item.title}</small></button>)}
        </div>
      </section>

      <section className="matrix-explainer" id="matrices">
        <div className="section-heading"><h2>Какие матрицы<br/>перемножаются</h2><p>Runtime знает только этот граф. Шахматные значения находятся в токенах, масках и замороженных весах.</p></div>
        <div className="equation-line">
          {["X", "Wq", "Q", "Kᵀ", "QKᵀ", "H", "V", "AV"].map((x, i) => <div key={x} className={x === current.short || (stage === 3 && x === "H") ? "hot" : ""}><span>{x}</span>{i < 7 && <b>{["×", "=", "×", "=", "→ hardmax", "×", "="][i]}</b>}</div>)}
        </div>
        <div className="matrix-note"><code>{current.equation}</code><p>{current.detail}</p></div>
      </section>

      <section className="trace-section" id="trace">
        <div className="trace-top"><div><h2>Трасса токенов</h2><p>Как в Percepta: можно остановить inference и увидеть, какой токен запросил какой ключ и какое значение было перенесено.</p></div>
          <div className="mode-switch"><button className={traceMode === "tokens" ? "active" : ""} onClick={() => setTraceMode("tokens")}>Токены</button><button className={traceMode === "matrix" ? "active" : ""} onClick={() => setTraceMode("matrix")}>Sparse QKᵀ</button></div>
        </div>
        {traceMode === "tokens" ? <div className="token-table">
          <div className="table-head"><span>ROW</span><span>ЗНАЧЕНИЕ</span><span>РОЛЬ</span><span>МАРШРУТ</span></div>
          {TOKEN_ROWS.map((row, i) => <div className={i === Math.min(stage, TOKEN_ROWS.length - 1) ? "active" : ""} key={row.token}><span>{String(i).padStart(3, "0")} · {row.token}</span><strong>{row.value}</strong><span>{row.role}</span><i><b style={{ width: `${20 + i * 13}%` }}/></i></div>)}
        </div> : <div className="sparse-view">
          <div className="axis y">query rows</div><div className="sparse-grid">{Array.from({ length: 64 }, (_, i) => <i key={i} className={selectedCells.has(i) ? "selected" : ""}><span>{selectedCells.has(i) ? "1" : "·"}</span></i>)}</div><div className="axis x">key columns →</div>
        </div>}
        <div className="trace-detail"><div><small>QUERY</small><code>{current.activeTokens[0]}</code></div><ChevronRight/><div><small>WINNING KEY / VALUE</small><code>{current.activeTokens.slice(1).join(" · ")}</code></div></div>
      </section>

      <section className="proof" id="proof">
        <div><h2>Что доказано сейчас</h2><p>Сайт намеренно не выдаёт текущий rule-slice за готовые шахматы.</p></div>
        <ul><li><strong>✓</strong><span>Pawn push + diagonal capture</span><small>256 кандидатов · 15 стадий</small></li><li><strong>✓</strong><span>Color-aware target predicate</span><small>side × kind × piece → allowed</small></li><li><strong>✓</strong><span>Predicate-token legality</span><small>5 предикатов → hardmax AND</small></li><li><strong>✓</strong><span>Hardmax player circuit</span><small>recurrent inference-ходы</small></li><li><strong>✓</strong><span>Runtime purity</span><small>CTest 5/5 · mutation 14/14</small></li><li className="pending"><strong>→</strong><span>Полные шахматы</span><small>pieces/check/promotion впереди</small></li></ul>
      </section>
    </main>
    <footer><span>Percepta Chess · эксперимент с transformer-native вычислением</span><button onClick={() => { setStage(0); setPlaying(true); }}><RotateCcw/> Повторить трассу</button></footer>
  </>;
}

export default App;
