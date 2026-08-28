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
