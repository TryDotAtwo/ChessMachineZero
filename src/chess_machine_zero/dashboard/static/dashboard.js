const state = {
  snapshot: null,
  selectedSquare: null,
  playing: false,
  busy: false,
  timer: null,
  sideTraceMode: {
    white: "readable",
    black: "readable",
  },
  traceReplayTimers: {
    white: null,
    black: null,
  },
  traceReplayCounts: {
    white: 0,
    black: 0,
  },
  traceReplayKeys: {
    white: "",
    black: "",
  },
  traceStreams: {
    white: null,
    black: null,
  },
  traceScrollState: {
    white: makeTraceScrollState(),
    black: makeTraceScrollState(),
  },
  traceScrollSuppress: {
    white: false,
    black: false,
  },
};

function makeTraceScrollState() {
  return {
    readable: {
      locked: false,
      top: 0,
    },
    token: {
      locked: false,
      top: 0,
    },
  };
}

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
  tokenMeter: document.getElementById("tokenMeter"),
  whiteJournalMeter: document.getElementById("whiteJournalMeter"),
  blackJournalMeter: document.getElementById("blackJournalMeter"),
  whiteTraceOutput: document.getElementById("whiteTraceOutput"),
  blackTraceOutput: document.getElementById("blackTraceOutput"),
  whiteReadableTab: document.getElementById("whiteReadableTab"),
  whiteTokenTab: document.getElementById("whiteTokenTab"),
  blackReadableTab: document.getElementById("blackReadableTab"),
  blackTokenTab: document.getElementById("blackTokenTab"),
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
  renderSideTraceJournals(snapshot);
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

function renderSideTraceJournals(snapshot) {
  const streams = snapshot.transformer_token_streams || {};
  const white = streams.white || emptyTraceStream();
  const black = streams.black || emptyTraceStream();
  const totalPackets = white.packet_count + black.packet_count;
  const visiblePackets = white.packets.length + black.packets.length;
  els.tokenMeter.textContent = `tokens=${totalPackets * 7} packets=${totalPackets} visible=${visiblePackets}`;
  scheduleSideJournal("white", white);
  scheduleSideJournal("black", black);
}

function emptyTraceStream() {
  return {
    packet_count: 0,
    visible_packet_count: 0,
    truncated_count: 0,
    packets: [],
  };
}

function scheduleSideJournal(side, stream) {
  const key = traceStreamKey(stream);
  const previousKey = state.traceReplayKeys[side];
  const targetCount = stream.packet_count;
  state.traceStreams[side] = stream;
  state.traceReplayKeys[side] = key;
  if (state.traceReplayTimers[side] !== null) {
    window.clearTimeout(state.traceReplayTimers[side]);
    state.traceReplayTimers[side] = null;
  }
  if (key !== previousKey || targetCount < state.traceReplayCounts[side]) {
    state.traceReplayCounts[side] = 0;
  }
  appendTraceLinesGradually(side);
}

function traceStreamKey(stream) {
  const first = stream.packets[0];
  const last = stream.packets[stream.packets.length - 1];
  const firstToken = first ? first.tokens.join(".") : "none";
  const lastToken = last ? last.tokens.join(".") : "none";
  return `${stream.packet_count}:${firstToken}:${lastToken}`;
}

function appendTraceLinesGradually(side) {
  const stream = state.traceStreams[side] || emptyTraceStream();
  const targetCount = stream.packet_count;
  const current = state.traceReplayCounts[side];
  if (targetCount === 0) {
    renderSideJournalFrame(side, stream, 0, 0);
    return;
  }
  if (current >= targetCount) {
    renderSideJournalFrame(side, stream, targetCount, targetCount);
    return;
  }
  const next = Math.min(targetCount, current + 4);
  state.traceReplayCounts[side] = next;
  renderSideJournalFrame(side, stream, next, current + 1);
  const tick = () => {
    const activeStream = state.traceStreams[side] || emptyTraceStream();
    const activeTarget = activeStream.packet_count;
    const before = state.traceReplayCounts[side];
    const after = Math.min(activeTarget, before + 4);
    state.traceReplayCounts[side] = after;
    renderSideJournalFrame(side, activeStream, after, before + 1);
    if (after >= activeTarget) {
      state.traceReplayTimers[side] = null;
      return;
    }
    state.traceReplayTimers[side] = window.setTimeout(tick, 18);
  };
  state.traceReplayTimers[side] = window.setTimeout(tick, 18);
}

function renderSideJournalFrame(side, stream, displayCount, newStart) {
  const meter = side === "white" ? els.whiteJournalMeter : els.blackJournalMeter;
  const output = side === "white" ? els.whiteTraceOutput : els.blackTraceOutput;
  const readableTab = side === "white" ? els.whiteReadableTab : els.blackReadableTab;
  const tokenTab = side === "white" ? els.whiteTokenTab : els.blackTokenTab;
  const mode = state.sideTraceMode[side];
  const scrollState = state.traceScrollState[side][mode];
  const sameRenderedMode = output.dataset.traceMode === mode;
  const previousTop = sameRenderedMode ? output.scrollTop : scrollState.top;
  const shouldFollow = !scrollState.locked && (!sameRenderedMode || isTraceNearBottom(output));
  const packets = stream.packets.slice(-160);
  const visibleCount = Math.min(displayCount, packets.length);
  const visiblePackets = packets.slice(0, visibleCount);
  const offset = Math.max(0, stream.packet_count - packets.length);
  meter.textContent = `packets=${stream.packet_count} tokens=${stream.packet_count * 7}`;
  readableTab.classList.toggle("active", mode === "readable");
  tokenTab.classList.toggle("active", mode === "token");
  readableTab.setAttribute("aria-selected", String(mode === "readable"));
  tokenTab.setAttribute("aria-selected", String(mode === "token"));
  output.classList.toggle("readable-log", mode === "readable");
  output.classList.toggle("token-log", mode === "token");
  output.dataset.traceMode = mode;
  if (visiblePackets.length === 0) {
    output.replaceChildren(emptyLogLine(side, mode));
    restoreTraceScrollPosition(side, output, scrollState, true, previousTop);
    return;
  }
  const fragment = document.createDocumentFragment();
  visiblePackets.forEach((packet, index) => {
    const absoluteIndex = offset + index + 1;
    const line = document.createElement("div");
    line.className = traceLineClass(packet, absoluteIndex >= offset + newStart);
    line.textContent = mode === "token"
      ? formatRawTokenLine(packet, absoluteIndex, side)
      : formatReadableTraceLine(packet, absoluteIndex, side);
    fragment.appendChild(line);
  });
  output.replaceChildren(fragment);
  restoreTraceScrollPosition(side, output, scrollState, shouldFollow, previousTop);
}

function recordTraceScrollPosition(side) {
  if (state.traceScrollSuppress[side]) {
    return;
  }
  const output = side === "white" ? els.whiteTraceOutput : els.blackTraceOutput;
  const mode = state.sideTraceMode[side];
  const scrollState = state.traceScrollState[side][mode];
  scrollState.top = output.scrollTop;
  scrollState.locked = !isTraceNearBottom(output);
}

function isTraceNearBottom(output) {
  const remaining = output.scrollHeight - output.clientHeight - output.scrollTop;
  return remaining <= 8;
}

function restoreTraceScrollPosition(side, output, scrollState, shouldFollow, previousTop) {
  state.traceScrollSuppress[side] = true;
  if (shouldFollow) {
    output.scrollTop = output.scrollHeight;
    scrollState.top = output.scrollTop;
    scrollState.locked = false;
    window.setTimeout(() => {
      state.traceScrollSuppress[side] = false;
    }, 0);
    return;
  }
  output.scrollTop = Math.min(previousTop, Math.max(0, output.scrollHeight - output.clientHeight));
  scrollState.top = previousTop;
  window.setTimeout(() => {
    state.traceScrollSuppress[side] = false;
  }, 0);
}

function emptyLogLine(side, mode) {
  const line = document.createElement("div");
  line.className = "trace-line trace-line-muted";
  line.textContent = mode === "token"
    ? `waiting for ${side} token trace`
    : `waiting for ${side} trace packets`;
  return line;
}

function traceLineClass(packet, fresh) {
  const classes = ["trace-line", `trace-op-${packet.op.toLowerCase()}`];
  if (fresh) {
    classes.push("trace-line-new");
  }
  return classes.join(" ");
}

function formatReadableTraceLine(packet, index, side) {
  const actor = `${side} shared_transformer`;
  if (packet.op === "CANDIDATE") {
    return `${actor} step=${index} try ${squareName(packet.a1)}${squareName(packet.a2)} promo=${packet.a3} flags=${packet.commit}`;
  }
  if (packet.op === "LEGAL_SET") {
    return `${actor} step=${index} legal move_id=${packet.a0} ok=${packet.a1}`;
  }
  if (packet.op === "COMMIT_MOVE") {
    return `${actor} step=${index} commit ${squareName(packet.a1)}${squareName(packet.a2)} promo=${packet.a3} flags=${packet.commit}`;
  }
  if (packet.op === "WRITE_SQ") {
    return `${actor} step=${index} write ${squareName(packet.a0)} piece=${packet.a1} ply=${packet.a2}`;
  }
  if (packet.op === "TERMINAL_SET") {
    return `${actor} step=${index} terminal result=${packet.a0} reason=${packet.a1} ply=${packet.a2}`;
  }
  if (packet.op === "PROGRAM_HALT") {
    return `${actor} step=${index} halt tag=${packet.tag}`;
  }
  return `${actor} step=${index} ${packet.op} a0=${packet.a0} a1=${packet.a1} a2=${packet.a2} a3=${packet.a3} tag=${packet.tag} commit=${packet.commit}`;
}

function formatRawTokenLine(packet, index, side) {
  const hexToken = packet.tokens.map((value) => Number(value).toString(16).padStart(2, "0")).join(" ");
  return `${hexToken} ${side}.${packet.op.toLowerCase()}(${packetDescription(packet)},idx=${index})`;
}

function packetDescription(packet) {
  if (packet.op === "CANDIDATE") {
    return `move_id=${packet.a0},from=${squareName(packet.a1)},to=${squareName(packet.a2)},promo=${packet.a3},flags=${packet.commit}`;
  }
  if (packet.op === "LEGAL_SET") {
    return `move_id=${packet.a0},sts=${packet.a1},bt=0`;
  }
  if (packet.op === "COMMIT_MOVE") {
    return `from=${squareName(packet.a1)},to=${squareName(packet.a2)},promo=${packet.a3},flags=${packet.commit}`;
  }
  if (packet.op === "WRITE_SQ") {
    return `sq=${squareName(packet.a0)},piece=${packet.a1},ply=${packet.a2}`;
  }
  if (packet.op === "WRITE_REG" || packet.op === "WRITE_CASTLE" || packet.op === "WRITE_EP" || packet.op === "WRITE_CLOCK") {
    return `a0=${packet.a0},a1=${packet.a1},a2=${packet.a2},sts=1,bt=0`;
  }
  if (packet.op === "TERMINAL_SET") {
    return `result=${packet.a0},reason=${packet.a1},ply=${packet.a2}`;
  }
  if (packet.op === "PROGRAM_HALT") {
    return `tag=${packet.tag},sts=1,bt=0`;
  }
  return `a0=${packet.a0},a1=${packet.a1},a2=${packet.a2},a3=${packet.a3},tag=${packet.tag},commit=${packet.commit}`;
}

function squareName(index) {
  if (!Number.isInteger(index) || index < 0 || index > 63) {
    return String(index);
  }
  const files = "abcdefgh";
  return `${files[index % 8]}${Math.floor(index / 8) + 1}`;
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
  resetTraceReplayState();
  renderSnapshot(await api("/api/reset", {}));
});

function resetTraceReplayState() {
  for (const side of ["white", "black"]) {
    if (state.traceReplayTimers[side] !== null) {
      window.clearTimeout(state.traceReplayTimers[side]);
      state.traceReplayTimers[side] = null;
    }
    state.traceReplayCounts[side] = 0;
    state.traceReplayKeys[side] = "";
  }
  resetTraceScrollState();
}

function resetTraceScrollState() {
  state.traceScrollState.white = makeTraceScrollState();
  state.traceScrollState.black = makeTraceScrollState();
}

function setSideTraceMode(side, mode) {
  state.sideTraceMode[side] = mode;
  const stream = state.traceStreams[side] || emptyTraceStream();
  renderSideJournalFrame(side, stream, state.traceReplayCounts[side], state.traceReplayCounts[side] + 1);
}

els.whiteReadableTab.addEventListener("click", () => setSideTraceMode("white", "readable"));
els.whiteTokenTab.addEventListener("click", () => setSideTraceMode("white", "token"));
els.blackReadableTab.addEventListener("click", () => setSideTraceMode("black", "readable"));
els.blackTokenTab.addEventListener("click", () => setSideTraceMode("black", "token"));
els.whiteTraceOutput.addEventListener("scroll", () => recordTraceScrollPosition("white"));
els.blackTraceOutput.addEventListener("scroll", () => recordTraceScrollPosition("black"));

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
