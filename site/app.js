const fixtures = [
  { move: [52, 54, 96], from: "e2", to: "e4", piece: "♙" },
  { move: [57, 55, 102], from: "e7", to: "e5", piece: "♟" },
  { move: [71, 63, 97], from: "g1", to: "f3", piece: "♘" },
  { move: [28, 36, 103], from: "b8", to: "c6", piece: "♞" },
  { move: [61, 34, 98], from: "f1", to: "c4", piece: "♗" },
  { move: [86, 75, 103], from: "g8", to: "f6", piece: "♞" },
  { move: [51, 53, 96], from: "d2", to: "d4", piece: "♙" },
];

const initialPosition = {
  a8: "♜", b8: "♞", c8: "♝", d8: "♛", e8: "♚", f8: "♝", g8: "♞", h8: "♜",
  a7: "♟", b7: "♟", c7: "♟", d7: "♟", e7: "♟", f7: "♟", g7: "♟", h7: "♟",
  a2: "♙", b2: "♙", c2: "♙", d2: "♙", e2: "♙", f2: "♙", g2: "♙", h2: "♙",
  a1: "♖", b1: "♘", c1: "♗", d1: "♕", e1: "♔", f1: "♗", g1: "♘", h1: "♖",
};

const files = "abcdefgh";
const board = document.querySelector("#board");
const tokenList = document.querySelector("#tokenList");
const capacityText = document.querySelector("#capacityText");
const capacityBar = document.querySelector("#capacityBar");
const verification = document.querySelector("#verification");
let timer = null;
let visiblePlies = fixtures.length;

function positionAfter(plies) {
  const position = { ...initialPosition };
  fixtures.slice(0, plies).forEach(({ from, to, piece }) => {
    delete position[from];
    position[to] = piece;
  });
  return position;
}

function renderBoard(changedSquare = "") {
  const position = positionAfter(visiblePlies);
  board.replaceChildren();
  for (let rank = 8; rank >= 1; rank -= 1) {
    for (let file = 0; file < 8; file += 1) {
      const squareName = `${files[file]}${rank}`;
      const square = document.createElement("div");
      square.className = `square ${(file + rank) % 2 === 0 ? "dark" : ""}`;
      if (squareName === changedSquare) square.classList.add("changed");
      square.setAttribute("role", "gridcell");
      square.setAttribute("aria-label", `${squareName}${position[squareName] ? ` ${position[squareName]}` : " empty"}`);
      square.textContent = position[squareName] || "";
      board.append(square);
    }
  }
}

function renderHistory(activeIndex = visiblePlies - 1) {
  tokenList.replaceChildren();
  fixtures.forEach(({ move }, index) => {
    const row = document.createElement("li");
    if (index === activeIndex && index < visiblePlies) row.className = "active";
    row.innerHTML = `<span>[${move[0]}]</span><span>[${move[1]}]</span><span>[${move[2]}]</span>`;
    row.style.opacity = index < visiblePlies ? "1" : ".22";
    tokenList.append(row);
  });
  capacityText.textContent = `${visiblePlies} / 400 plies`;
  capacityBar.style.width = `${Math.max(2, visiblePlies / 4)}%`;
}

function stopReplay() {
  if (timer !== null) window.clearInterval(timer);
  timer = null;
  verification.classList.remove("running");
}

document.querySelector("#runArtifact").addEventListener("click", () => {
  stopReplay();
  visiblePlies = 0;
  verification.classList.add("running");
  renderHistory(-1);
  renderBoard();
  timer = window.setInterval(() => {
    visiblePlies += 1;
    const current = fixtures[visiblePlies - 1];
    renderHistory(visiblePlies - 1);
    renderBoard(current.to);
    if (visiblePlies === fixtures.length) stopReplay();
  }, 360);
});

document.querySelector("#resetTrace").addEventListener("click", () => {
  stopReplay();
  visiblePlies = 0;
  renderHistory(-1);
  renderBoard();
});

renderHistory();
renderBoard();

function compactAttributes(attributes) {
  if (attributes.length <= 10) return `[${attributes.join(", ")}]`;
  return `[${attributes.slice(0, 4).join(", ")}, … ${attributes.length - 4} routing indices]`;
}

async function renderArtifactTrace() {
  const response = await fetch("./artifact_trace.json");
  if (!response.ok) throw new Error(`artifact trace request failed: ${response.status}`);
  const trace = await response.json();

  const opcodeContainer = document.querySelector("#runtimeOpcodes");
  opcodeContainer.replaceChildren(...trace.runtime_opcodes.map((opcode) => {
    const item = document.createElement("code");
    item.textContent = opcode;
    return item;
  }));

  const operationContainer = document.querySelector("#ssaOperations");
  operationContainer.replaceChildren(...trace.operations.map((operation) => {
    const row = document.createElement("tr");
    const values = [
      operation.index,
      operation.opcode,
      operation.inputs.map((value) => `v${value}`).join(", "),
      operation.outputs.map((value) => `v${value}`).join(", "),
      compactAttributes(operation.attributes),
    ];
    values.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    return row;
  }));
}

renderArtifactTrace().catch((error) => {
  const operationContainer = document.querySelector("#ssaOperations");
  operationContainer.innerHTML = `<tr><td colspan="5">${error.message}</td></tr>`;
});

function formatShape(shape) { return `[${shape.join(" × ")}]`; }

const MATRIX_PAGE_SIZE = 200;

function matrixView(name, matrix) {
  const article = document.createElement("article");
  article.className = "numeric-matrix";
  const heading = document.createElement("h4");
  heading.textContent = `${name} ${formatShape(matrix.shape)} · nnz=${matrix.nnz}`;
  const scroller = document.createElement("div");
  scroller.className = "matrix-values";
  const table = document.createElement("table");
  table.innerHTML = "<thead><tr><th>index</th><th>value</th></tr></thead>";
  const body = document.createElement("tbody");
  let page = 0;
  const pager = document.createElement("div");
  pager.className = "matrix-pager";
  const previous = document.createElement("button");
  previous.type = "button";
  previous.textContent = "←";
  const status = document.createElement("span");
  const next = document.createElement("button");
  next.type = "button";
  next.textContent = "→";
  function renderPage() {
    const start = page * MATRIX_PAGE_SIZE;
    const end = Math.min(start + MATRIX_PAGE_SIZE, matrix.entries.length);
    body.replaceChildren(...matrix.entries.slice(start, end).map((entry) => {
      const row = document.createElement("tr");
      const index = document.createElement("td");
      const value = document.createElement("td");
      index.textContent = `[${entry.slice(0, -1).join(",")}]`;
      value.textContent = String(entry.at(-1));
      row.append(index, value);
      return row;
    }));
    status.textContent = matrix.entries.length ? `${start + 1}–${end} / ${matrix.entries.length}` : "0 / 0";
    previous.disabled = page === 0;
    next.disabled = end === matrix.entries.length;
  }
  previous.addEventListener("click", () => { page -= 1; renderPage(); });
  next.addEventListener("click", () => { page += 1; renderPage(); });
  pager.append(previous, status, next);
  renderPage();
  table.append(body);
  scroller.append(table);
  article.append(heading, pager, scroller);
  return article;
}

function renderCalculation(sample) {
  const lines = [`output[${sample.output_index.join(",")}] = ${sample.output_value}`];
  if (sample.identity) lines.push(sample.identity);
  if (sample.terms && Array.isArray(sample.terms[0])) {
    sample.terms.forEach(([k, left, right, product]) => lines.push(`k=${k}: ${left} × ${right} = ${product}`));
    lines.push(`sum = ${sample.output_value}`);
  } else if (sample.terms) lines.push(`${sample.terms.join(" + ")} = ${sample.output_value}`);
  if (sample.score_terms) {
    sample.score_terms.forEach(([d, q, k]) => lines.push(`d=${d}: Q=${q}, K=${k}, Q×K=${q * k}`));
    lines.push(`winner=${sample.winner}, score=${sample.winner_score}`);
    lines.push(`V[${sample.winner},${sample.output_index.at(-1)}] = ${sample.output_value}`);
  }
  return lines.join("\n");
}

async function renderNumericTrace() {
  const response = await fetch("./numeric_trace.json");
  if (!response.ok) throw new Error(`numeric trace request failed: ${response.status}`);
  const trace = await response.json();
  const picker = document.querySelector("#numericOperations");
  trace.operations.forEach((operation, arrayIndex) => {
    const option = document.createElement("option");
    option.value = String(arrayIndex);
    option.textContent = `${String(operation.index).padStart(2, "0")} · ${operation.opcode} · ${operation.equation}`;
    picker.append(option);
  });
  function show(index) {
    const bounded = Math.max(0, Math.min(trace.operations.length - 1, index));
    picker.value = String(bounded);
    const operation = trace.operations[bounded];
    document.querySelector("#numericEquation").textContent = operation.equation;
    const inputs = document.querySelector("#matrixInputs");
    const weights = document.querySelector("#matrixWeights");
    const intermediates = document.querySelector("#matrixIntermediates");
    const output = document.querySelector("#matrixOutput");
    inputs.replaceChildren(...operation.inputs.map((name) => matrixView(name, trace.values[name])));
    weights.replaceChildren(...operation.frozen.map((name) => matrixView(`${name} · ${trace.tensors[name].name}`, trace.tensors[name])));
    if (!operation.frozen.length) weights.textContent = "No frozen matrix operand for this operation.";
    intermediates.replaceChildren(...operation.derived.map((name) => matrixView(name, trace.derived[name])));
    if (!operation.derived.length) intermediates.textContent = "No hidden intermediate matrix for this operation.";
    output.replaceChildren(matrixView(operation.output, trace.values[operation.output]));
    document.querySelector("#matrixCalculation").textContent = renderCalculation(operation.sample);
  }
  picker.addEventListener("change", () => show(Number(picker.value)));
  document.querySelector("#previousNumericOperation").addEventListener("click", () => show(Number(picker.value) - 1));
  document.querySelector("#nextNumericOperation").addEventListener("click", () => show(Number(picker.value) + 1));
  show(0);
}

renderNumericTrace().catch((error) => { document.querySelector("#numericEquation").textContent = error.message; });
