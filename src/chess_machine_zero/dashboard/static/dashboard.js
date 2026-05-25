const state = {
  snapshot: null,
  selectedSquare: null,
  playing: false,
  busy: false,
  timer: null,
};

const els = {
  board: document.getElementById("board"),
  engineLine: document.getElementById("engineLine"),
  sideToMove: document.getElementById("sideToMove"),
  plyCount: document.getElementById("plyCount"),
  legalCount: document.getElementById("legalCount"),
  illegalCount: document.getElementById("illegalCount"),
  lastEmitter: document.getElementById("lastEmitter"),
  traceVerified: document.getElementById("traceVerified"),
  packetCount: document.getElementById("packetCount"),
  candidateCount: document.getElementById("candidateCount"),
  legalSetCount: document.getElementById("legalSetCount"),
  commitCount: document.getElementById("commitCount"),
  traceLog: document.getElementById("traceLog"),
  legalMoves: document.getElementById("legalMoves"),
  history: document.getElementById("history"),
  mode: document.getElementById("mode"),
  promotion: document.getElementById("promotion"),
  playPause: document.getElementById("playPause"),
  step: document.getElementById("step"),
  reset: document.getElementById("reset"),
  statusLine: document.getElementById("statusLine"),
};

async function api(path, payload) {
  const options = payload === undefined
    ? { method: "GET" }
    : {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    renderSnapshot(data.snapshot);
    throw new Error(data.error ? data.error.message : `request failed: ${response.status}`);
  }
  return data;
}

async function loadSnapshot() {
  renderSnapshot(await api("/api/snapshot"));
}

function renderSnapshot(snapshot) {
  if (!snapshot) {
    return;
  }
  state.snapshot = snapshot;
  state.selectedSquare = null;
  els.engineLine.textContent =
    `${snapshot.transformers.active} | ${snapshot.engine.rules_module} mode=${snapshot.engine.rule_execution_mode} stream=${snapshot.engine.token_streaming} mlp=${snapshot.engine.uses_mlp} prompts=${snapshot.engine.compiled_prompt_count} compiled=${snapshot.engine.compiled_rule_parameters}`;
  els.sideToMove.textContent = snapshot.side_to_move;
  els.plyCount.textContent = String(snapshot.ply);
  els.legalCount.textContent = String(snapshot.legal_count);
  els.illegalCount.textContent = String(snapshot.illegal_attempt_count + snapshot.illegal_commit_count);
  els.lastEmitter.textContent = snapshot.last_trace_actor || snapshot.transformers.active;
  els.traceVerified.textContent = snapshot.trace_legal_verification.selected_move_in_legal_set === null
    ? "-"
    : String(snapshot.trace_legal_verification.selected_move_in_legal_set);
  els.packetCount.textContent = String(snapshot.last_trace.packet_count);
  els.candidateCount.textContent = String(snapshot.last_trace.op_counts.CANDIDATE || 0);
  els.legalSetCount.textContent = String(snapshot.last_trace.op_counts.LEGAL_SET || 0);
  els.commitCount.textContent = String(snapshot.last_trace.op_counts.COMMIT_MOVE || 0);
  els.statusLine.textContent = statusText(snapshot);
  renderBoard(snapshot);
  renderLegalMoves(snapshot);
  renderHistory(snapshot);
  renderTrace(snapshot);
}

function statusText(snapshot) {
  if (snapshot.terminal.is_terminal) {
    return `terminal result=${snapshot.terminal.result} reason=${snapshot.terminal.reason}`;
  }
  if (isHumanTurn(snapshot)) {
    return "human turn";
  }
  return "transformer turn";
}

function renderBoard(snapshot) {
  const byIndex = new Map(snapshot.board.squares.map((square) => [square.index, square]));
  const targetSquares = targetSet();
  els.board.replaceChildren();
  for (let rank = 7; rank >= 0; rank -= 1) {
    for (let file = 0; file < 8; file += 1) {
      const index = rank * 8 + file;
      const square = byIndex.get(index);
      const button = document.createElement("button");
      button.type = "button";
      button.className = squareClass(index, file, rank, square, targetSquares);
      button.dataset.square = square.name;
      button.setAttribute("role", "gridcell");
      button.setAttribute("aria-label", `${square.name} ${square.piece || "empty"}`);
      const piece = document.createElement("span");
      piece.className = "piece";
      piece.textContent = square.unicode;
      button.appendChild(piece);
      button.addEventListener("click", () => onSquareClick(square.name));
      els.board.appendChild(button);
    }
  }
}

function squareClass(index, file, rank, square, targetSquares) {
  const classes = ["square"];
  if ((file + rank) % 2 === 1) {
    classes.push("dark");
  }
  if (square.color === "black") {
    classes.push("black-piece");
  }
  if (state.selectedSquare === square.name) {
    classes.push("selected");
  }
  if (targetSquares.has(square.name)) {
    classes.push(square.piece ? "capture-target" : "target");
  }
  return classes.join(" ");
}

function renderLegalMoves(snapshot) {
  els.legalMoves.replaceChildren();
  for (const move of snapshot.legal_moves) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = move;
    button.addEventListener("click", () => submitHumanMove(move));
    els.legalMoves.appendChild(button);
  }
}

function renderHistory(snapshot) {
  els.history.replaceChildren();
  for (const item of snapshot.history.slice(-80)) {
    const row = document.createElement("li");
    row.innerHTML = `<span>${item.actor} ${item.side_to_move}</span> ${item.move_uci} <span>tokens=${item.emitted_token_count} legal=${item.trace_verified_legal}</span>`;
    els.history.appendChild(row);
  }
}

function renderTrace(snapshot) {
  const last = snapshot.history[snapshot.history.length - 1];
  if (last && last.emitted_tokens) {
    const offset = Math.max(0, last.emitted_tokens.length - 120);
    els.traceLog.textContent = last.emitted_tokens
      .slice(offset)
      .map((tokens, index) => `${last.actor} token[${offset + index + 1}]=[${tokens.join(",")}]`)
      .join("\n");
    return;
  }
  const packets = snapshot.last_trace.packets.slice(-90);
  els.traceLog.textContent = packets
    .map((packet, index) =>
      `${snapshot.transformers.active} token[${index + 1}]=[${packet.tokens.join(",")}] op=${packet.op} tag=${packet.tag}`
    )
    .join("\n");
}

function targetSet() {
  const targets = new Set();
  if (!state.snapshot || !state.selectedSquare) {
    return targets;
  }
  for (const move of state.snapshot.legal_moves) {
    if (move.slice(0, 2) === state.selectedSquare) {
      targets.add(move.slice(2, 4));
    }
  }
  return targets;
}

function onSquareClick(squareName) {
  const snapshot = state.snapshot;
  if (!snapshot || !isHumanTurn(snapshot)) {
    return;
  }
  if (!state.selectedSquare) {
    if (hasLegalFrom(squareName)) {
      state.selectedSquare = squareName;
      renderBoard(snapshot);
    }
    return;
  }
  if (state.selectedSquare === squareName) {
    state.selectedSquare = null;
    renderBoard(snapshot);
    return;
  }
  const move = chooseMove(state.selectedSquare, squareName);
  if (!move) {
    state.selectedSquare = hasLegalFrom(squareName) ? squareName : null;
    renderBoard(snapshot);
    return;
  }
  submitHumanMove(move);
}

function hasLegalFrom(squareName) {
  return state.snapshot.legal_moves.some((move) => move.slice(0, 2) === squareName);
}

function chooseMove(from, to) {
  const matches = state.snapshot.legal_moves.filter((move) => move.slice(0, 2) === from && move.slice(2, 4) === to);
  if (matches.length === 0) {
    return null;
  }
  if (matches.length === 1) {
    return matches[0];
  }
  const promo = els.promotion.value;
  return matches.find((move) => move.endsWith(promo)) || matches.find((move) => move.endsWith("q")) || matches[0];
}

function isHumanTurn(snapshot) {
  const mode = els.mode.value;
  return (mode === "human_white" && snapshot.side_to_move === "w") ||
    (mode === "human_black" && snapshot.side_to_move === "b");
}

async function submitHumanMove(move) {
  try {
    stopAutoPlay();
    renderSnapshot(await api("/api/move", { move, auto_reply: els.mode.value !== "selfplay" }));
  } catch (error) {
    els.statusLine.textContent = error.message;
  }
}

async function stepOnce() {
  if (state.busy) {
    return;
  }
  try {
    if (!state.snapshot || state.snapshot.terminal.is_terminal) {
      return;
    }
    if (els.mode.value === "selfplay" || !isHumanTurn(state.snapshot)) {
      state.busy = true;
      els.step.disabled = true;
      els.statusLine.textContent = `${state.snapshot.transformers.active} computing move`;
      renderSnapshot(await api("/api/step", { count: 1 }));
    }
  } catch (error) {
    stopAutoPlay();
    els.statusLine.textContent = error.message;
  } finally {
    state.busy = false;
    els.step.disabled = false;
  }
}

async function autoPlayLoop() {
  if (!state.playing) {
    return;
  }
  await stepOnce();
  if (state.playing && state.snapshot && !state.snapshot.terminal.is_terminal && !isHumanTurn(state.snapshot)) {
    state.timer = window.setTimeout(autoPlayLoop, 250);
  }
}

function startAutoPlay() {
  if (state.playing) {
    return;
  }
  state.playing = true;
  els.playPause.textContent = "Pause";
  autoPlayLoop();
}

function stopAutoPlay() {
  state.playing = false;
  els.playPause.textContent = "Play";
  if (state.timer !== null) {
    window.clearTimeout(state.timer);
    state.timer = null;
  }
}

els.playPause.addEventListener("click", () => {
  if (state.playing) {
    stopAutoPlay();
  } else {
    startAutoPlay();
  }
});

els.step.addEventListener("click", () => {
  stopAutoPlay();
  stepOnce();
});

els.reset.addEventListener("click", async () => {
  stopAutoPlay();
  renderSnapshot(await api("/api/reset", {}));
});

els.mode.addEventListener("change", () => {
  if (els.mode.value !== "selfplay") {
    stopAutoPlay();
  }
  state.selectedSquare = null;
  renderSnapshot(state.snapshot);
});

loadSnapshot().then(() => {
  if (els.mode.value === "selfplay") {
    startAutoPlay();
  }
});
