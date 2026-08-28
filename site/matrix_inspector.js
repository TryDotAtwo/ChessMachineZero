(function matrixInspectorModule() {
  const MODELS = {
    ROW_ROUTE: "route",
    TOKEN_PROJECT: "matmul",
    POSITION_ADD: "add",
    RESIDUAL_ADD: "add",
    FROZEN_EXPAND: "expand",
    ROW_CONCAT: "concat",
    HARDMAX_STE: "hardmax",
    HULL_ATTN_2D: "attention",
  };

  const PIECES = {
    96: ["белая пешка", "white pawn"], 97: ["белый конь", "white knight"],
    98: ["белый слон", "white bishop"], 99: ["белая ладья", "white rook"],
    100: ["белый ферзь", "white queen"], 101: ["белый король", "white king"],
    102: ["чёрная пешка", "black pawn"], 103: ["чёрный конь", "black knight"],
    104: ["чёрный слон", "black bishop"], 105: ["чёрная ладья", "black rook"],
    106: ["чёрный ферзь", "black queen"], 107: ["чёрный король", "black king"],
  };

  const COPY = {
    ru: {
      title: "Матричное исполнение",
      subtitle: "Выбери любую из 45 операций, затем наведи или нажми на выходную ячейку.",
      operation: "Операция", row: "Строка выхода", column: "Столбец выхода",
      dimension: "Окно измерения k", explanation: "Как получилась выбранная ячейка",
      input: "Вход", weight: "Frozen-веса", output: "Выход", scores: "Оценки QKᵀ", attention: "Hardmax attention A",
      rows: "строки", columns: "столбцы", technical: "SSA",
      matmulTitle: "Строка матрицы × столбец весов",
      matmulDesc: "Цветом выделены ровно та строка и тот столбец, которые дают выбранное число.",
      genericDesc: "Показано точное преобразование входных ячеек в выходную.",
      selectedRow: "выбранная строка", selectedColumn: "выбранный столбец", value: "значение",
      token: "токен", batch: "партия в batch", move: "строка / ход", pattern: "столбец / признак",
    },
    en: {
      title: "Matrix execution", subtitle: "Select any of 45 operations, then hover or click any output cell.",
      operation: "Operation", row: "Output row", column: "Output column", dimension: "Dot-product k window", explanation: "How the selected cell was produced",
      input: "Input", weight: "Frozen weights", output: "Output", scores: "QKᵀ scores", attention: "Hardmax attention A",
      rows: "rows", columns: "columns", technical: "SSA", matmulTitle: "Matrix row × weight column",
      matmulDesc: "The exact row and column producing the selected number are highlighted together.", genericDesc: "The exact mapping from input cells to the selected output cell is shown.",
      selectedRow: "selected row", selectedColumn: "selected column", value: "value", token: "token", batch: "batch item", move: "row / move", pattern: "column / feature",
    },
  };

  const matrixCache = new WeakMap();
  function matrixIndex(matrix) {
    if (!matrixCache.has(matrix)) {
      const map = new Map();
      matrix.entries.forEach((entry) => map.set(entry.slice(0, -1).join(","), entry.at(-1)));
      matrixCache.set(matrix, map);
    }
    return matrixCache.get(matrix);
  }
  function matrixValue(matrix, indices) { return matrixIndex(matrix).get(indices.join(",")) || 0; }
  function tokenMeaning(token, language) {
    if (token === 0) return language === "ru" ? "padding / пусто" : "padding / empty";
    const file = Math.floor(token / 10); const rank = token % 10;
    if (file >= 1 && file <= 8 && rank >= 1 && rank <= 8) return `${language === "ru" ? "клетка" : "square"} ${"abcdefgh"[file - 1]}${rank}`;
    if (PIECES[token]) return PIECES[token][language === "ru" ? 0 : 1];
    return `${language === "ru" ? "служебный токен" : "service token"} ${token}`;
  }
  function operationModel(opcode) { return MODELS[opcode] || null; }
  function dotProduct(left, weight, batch, row, column) {
    const size = left.shape.at(-1); const terms = [];
    for (let k = 0; k < size; k += 1) {
      const a = matrixValue(left, left.shape.length === 3 ? [batch, row, k] : [row, k]);
      const b = matrixValue(weight, [k, column]);
      terms.push({ k, left: a, right: b, product: a * b });
    }
    return { terms, total: terms.reduce((sum, term) => sum + term.product, 0) };
  }
  function concatSource(matrices, outputRow) {
    let offset = 0;
    for (let input = 0; input < matrices.length; input += 1) {
      const rows = matrices[input].shape.at(-2);
      if (outputRow < offset + rows) return { input, row: outputRow - offset };
      offset += rows;
    }
    return null;
  }
  function hardmaxWinner(matrix, batch, row) {
    for (let column = 0; column < matrix.shape.at(-1); column += 1) {
      if (matrixValue(matrix, [batch, row, column]) === 1) return column;
    }
    return 0;
  }

  const api = { matrixValue, tokenMeaning, operationModel, dotProduct, concatSource, hardmaxWinner };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof document === "undefined") return;

  let trace; let language = "ru"; let operationIndex = 0; let selectedRow = 0; let selectedColumn = 0; let kStart = 0;
  const byId = (id) => document.getElementById(id);
  const shapeText = (matrix) => `[${matrix.shape.join(" × ")}]`;
  function lastRows(matrix) { return matrix.shape.at(-2); }
  function lastColumns(matrix) { return matrix.shape.at(-1); }
  function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
  function displayNumber(value) { return Number.isInteger(value) ? String(value) : Number(value.toFixed(7)).toString(); }

  function semanticFrozenName(name) {
    const words = name.replaceAll("_", " ");
    if (language === "en") return words;
    return words
      .replace("castle", "рокировка")
      .replace("en passant", "взятие на проходе")
      .replace("latest square key projection", "проекция ключей клеток")
      .replace("latest square queries", "запросы клеток")
      .replace("initial square tokens", "токены исходных клеток")
      .replace("initial piece state", "исходные фигуры")
      .replace("match slot", "совпадение слота")
      .replace("no match bias", "bias нет совпадения")
      .replace("targets", "цели").replace("values", "значения")
      .replace("padding row", "padding-строка").replace("time bias", "временной bias");
  }

  function operationTitle(operation) {
    const titles = language === "ru" ? {
      ROW_ROUTE: "Выбор строк из истории", TOKEN_PROJECT: COPY.ru.matmulTitle,
      POSITION_ADD: "Поэлементное сложение с frozen-bias", RESIDUAL_ADD: "Поэлементное residual-сложение",
      FROZEN_EXPAND: "Добавление batch-измерения", ROW_CONCAT: "Склейка матриц по строкам",
      HARDMAX_STE: "Hardmax: выбор максимума в каждой строке", HULL_ATTN_2D: "Attention: QKᵀ → hardmax → A×V",
    } : {
      ROW_ROUTE: "Select history rows", TOKEN_PROJECT: COPY.en.matmulTitle, POSITION_ADD: "Elementwise frozen-bias addition",
      RESIDUAL_ADD: "Elementwise residual addition", FROZEN_EXPAND: "Expand across batch", ROW_CONCAT: "Concatenate matrices by rows",
      HARDMAX_STE: "Hardmax: select each row maximum", HULL_ATTN_2D: "Attention: QKᵀ → hardmax → A×V",
    };
    return titles[operation.opcode];
  }

  function matrixCard(name, technicalName, matrix, options = {}) {
    const rows = lastRows(matrix); const columns = lastColumns(matrix); const row = clamp(options.row || 0, 0, rows - 1); const column = clamp(options.column || 0, 0, columns - 1);
    const rowStart = clamp(options.rowStart ?? row - 2, 0, Math.max(0, rows - 6)); const columnStart = clamp(options.columnStart ?? column - 3, 0, Math.max(0, columns - 8));
    const rowEnd = Math.min(rows, rowStart + 6); const columnEnd = Math.min(columns, columnStart + 8);
    const article = document.createElement("article"); article.className = "matrix-card";
    const header = document.createElement("header"); header.innerHTML = `<div><strong>${name}</strong><small>${COPY[language].technical}: ${technicalName}</small></div><span>${shapeText(matrix)}</span>`; article.append(header);
    const rowAxis = options.rowAxis || COPY[language].rows; const columnAxis = options.columnAxis || COPY[language].columns;
    const axes = document.createElement("p"); axes.className = "matrix-axes"; axes.textContent = `↓ ${rowAxis} ${rowStart}…${rowEnd - 1} · → ${columnAxis} ${columnStart}…${columnEnd - 1}`; article.append(axes);
    const table = document.createElement("table"); const head = document.createElement("thead"); const hr = document.createElement("tr"); hr.append(document.createElement("th"));
    for (let col = columnStart; col < columnEnd; col += 1) { const th = document.createElement("th"); th.textContent = col; hr.append(th); } head.append(hr); table.append(head);
    const body = document.createElement("tbody");
    for (let r = rowStart; r < rowEnd; r += 1) {
      const tr = document.createElement("tr"); const th = document.createElement("th"); th.textContent = r; tr.append(th);
      for (let col = columnStart; col < columnEnd; col += 1) {
        const index = matrix.shape.length === 3 ? [0, r, col] : [r, col]; const value = matrixValue(matrix, index); const td = document.createElement("td"); td.textContent = displayNumber(value);
        if (options.highlightRow === r) td.classList.add("matrix-row-highlight");
        if (options.highlightColumn === col) td.classList.add("matrix-column-highlight");
        if (options.selectedRow === r && options.selectedColumn === col) td.classList.add("matrix-cell-selected");
        if (value !== 0) td.classList.add(value > 0 ? "matrix-positive" : "matrix-negative");
        const meaningIndex = options.tokenAxis === "row" ? r : options.tokenAxis === "column" ? col : null;
        const meaning = meaningIndex === null ? "" : ` · ${COPY[language].token} ${meaningIndex} = ${tokenMeaning(meaningIndex, language)}`;
        td.title = `${technicalName}[${index.join(", ")}] = ${displayNumber(value)} · ${rowAxis} ${r} · ${columnAxis} ${col}${meaning}`;
        if (options.onSelect) { td.tabIndex = 0; td.addEventListener("mouseenter", () => options.onSelect(r, col)); td.addEventListener("click", () => options.onSelect(r, col)); }
        tr.append(td);
      } body.append(tr);
    }
    table.append(body); const viewport = document.createElement("div"); viewport.className = "matrix-table-viewport"; viewport.append(table); article.append(viewport); return article;
  }

  function sign(symbol) { const element = document.createElement("div"); element.className = "matrix-sign"; element.textContent = symbol; return element; }
  function valueName(id) { return `${COPY[language].input} (${id})`; }
  function selectedOutput(operation) { return trace.values[operation.output]; }
  function normalizeSelection(operation) {
    const output = selectedOutput(operation); selectedRow = clamp(selectedRow, 0, lastRows(output) - 1); selectedColumn = clamp(selectedColumn, 0, lastColumns(output) - 1);
    byId("outputRow").max = String(lastRows(output) - 1); byId("outputColumn").max = String(lastColumns(output) - 1); byId("outputRow").value = String(selectedRow); byId("outputColumn").value = String(selectedColumn);
  }

  function renderMatmul(operation, flow) {
    const left = trace.values[operation.inputs[0]]; const weightId = operation.frozen[0]; const weight = trace.tensors[weightId]; const output = selectedOutput(operation);
    const product = dotProduct(left, weight, 0, selectedRow, selectedColumn); const firstActive = product.terms.find((term) => term.left !== 0 || term.right !== 0)?.k || 0;
    if (kStart === 0 && firstActive > 7) kStart = Math.floor(firstActive / 8) * 8; kStart = clamp(kStart, 0, Math.max(0, product.terms.length - 8));
    byId("dotDimension").max = String(Math.max(0, product.terms.length - 8)); byId("dotDimension").value = String(kStart); byId("dotDimensionValue").textContent = `${kStart}–${Math.min(product.terms.length - 1, kStart + 7)}`;
    flow.append(
      matrixCard(COPY[language].input, operation.inputs[0], left, { row: selectedRow, rowStart: selectedRow - 2, columnStart: kStart, highlightRow: selectedRow, rowAxis: COPY[language].move, columnAxis: COPY[language].token, tokenAxis: "column" }), sign("×"),
      matrixCard(`${COPY[language].weight}: ${semanticFrozenName(weight.name)}`, weightId, weight, { rowStart: kStart, column: selectedColumn, highlightColumn: selectedColumn, rowAxis: COPY[language].token, columnAxis: COPY[language].pattern, tokenAxis: "row" }), sign("="),
      matrixCard(COPY[language].output, operation.output, output, { row: selectedRow, column: selectedColumn, selectedRow, selectedColumn, onSelect: selectCell, rowAxis: COPY[language].move, columnAxis: COPY[language].pattern })
    );
    const visibleTerms = product.terms.slice(kStart, kStart + 8); const formula = visibleTerms.map((term) => `<span class="dot-term"><i>k=${term.k}</i>${displayNumber(term.left)} × ${displayNumber(term.right)} = <b>${displayNumber(term.product)}</b></span>`).join("");
    byId("cellExplanation").innerHTML = `<p><strong>${operation.output}[0, ${selectedRow}, ${selectedColumn}] = ${displayNumber(matrixValue(output, [0, selectedRow, selectedColumn]))}</strong></p><div class="dot-terms">${formula}</div><p>Σ k=0…${product.terms.length - 1} = <strong>${displayNumber(product.total)}</strong></p>`;
  }

  function renderAttention(operation, flow) {
    const q = trace.values[operation.inputs[0]]; const k = trace.values[operation.inputs[1]]; const v = trace.values[operation.inputs[2]]; const scores = trace.derived[operation.derived[0]]; const attention = trace.derived[operation.derived[1]]; const output = selectedOutput(operation);
    let winner = 0; for (let candidate = 0; candidate < lastColumns(attention); candidate += 1) if (matrixValue(attention, [0, selectedRow, candidate]) === 1) { winner = candidate; break; }
    flow.classList.add("attention-flow");
    flow.append(
      matrixCard("Queries Q", operation.inputs[0], q, { row: selectedRow, highlightRow: selectedRow }), sign("×"),
      matrixCard("Keys K", operation.inputs[1], k, { row: winner, highlightRow: winner }), sign("→"),
      matrixCard(COPY[language].scores, operation.derived[0], scores, { row: selectedRow, column: winner, rowStart: selectedRow - 2, columnStart: winner - 3, selectedRow, selectedColumn: winner }), sign("→"),
      matrixCard(COPY[language].attention, operation.derived[1], attention, { row: selectedRow, column: winner, rowStart: selectedRow - 2, columnStart: winner - 3, selectedRow, selectedColumn: winner }), sign("×"),
      matrixCard("Values V", operation.inputs[2], v, { row: winner, column: selectedColumn, rowStart: winner - 2, columnStart: selectedColumn - 3, highlightRow: winner, highlightColumn: selectedColumn }), sign("="),
      matrixCard(COPY[language].output, operation.output, output, { row: selectedRow, column: selectedColumn, selectedRow, selectedColumn, onSelect: selectCell })
    );
    const qTerms = [0, 1].map((d) => `${displayNumber(matrixValue(q, [0, selectedRow, d]))}×${displayNumber(matrixValue(k, [0, winner, d]))}`).join(" + ");
    byId("cellExplanation").innerHTML = `<p>Q[${selectedRow},:] × K[${winner},:] = ${qTerms} = <strong>${displayNumber(matrixValue(scores, [0, selectedRow, winner]))}</strong></p><p>hardmax → ${COPY[language].selectedColumn} ${winner}; A[${selectedRow},${winner}] = 1</p><p>A[${selectedRow},:] × V[:,${selectedColumn}] = V[${winner},${selectedColumn}] = <strong>${displayNumber(matrixValue(output, [0, selectedRow, selectedColumn]))}</strong></p>`;
  }

  function renderGeneric(operation, flow) {
    const matrices = operation.inputs.map((id) => [valueName(id), id, trace.values[id]]);
    operation.frozen.forEach((id) => matrices.push([`${COPY[language].weight}: ${semanticFrozenName(trace.tensors[id].name)}`, id, trace.tensors[id]]));
    operation.derived.forEach((id) => matrices.push([id.includes("attention") ? COPY[language].attention : COPY[language].scores, id, trace.derived[id]]));
    const concat = operation.opcode === "ROW_CONCAT" ? concatSource(operation.inputs.map((id) => trace.values[id]), selectedRow) : null;
    const winner = operation.opcode === "HARDMAX_STE" ? hardmaxWinner(selectedOutput(operation), 0, selectedRow) : selectedColumn;
    matrices.forEach(([name, id, matrix], index) => {
      if (index) flow.append(sign(operation.opcode.includes("ADD") ? "+" : "→"));
      const inputIndex = operation.inputs.indexOf(id);
      const contributing = !concat || inputIndex === concat.input;
      const matrixRow = concat && contributing ? concat.row : selectedRow;
      flow.append(matrixCard(name, id, matrix, {
        row: matrixRow,
        column: operation.opcode === "HARDMAX_STE" ? winner : selectedColumn,
        highlightRow: contributing ? matrixRow : -1,
        highlightColumn: contributing ? (operation.opcode === "HARDMAX_STE" ? winner : selectedColumn) : -1,
      }));
    });
    flow.append(sign("="), matrixCard(COPY[language].output, operation.output, selectedOutput(operation), { row: selectedRow, column: selectedColumn, selectedRow, selectedColumn, onSelect: selectCell }));
    const output = selectedOutput(operation); const outputValue = matrixValue(output, [0, selectedRow, selectedColumn]);
    let explanation = `${operation.output}[0, ${selectedRow}, ${selectedColumn}] = <strong>${displayNumber(outputValue)}</strong>`;
    if (operation.opcode === "RESIDUAL_ADD") explanation += `<br>${displayNumber(matrixValue(trace.values[operation.inputs[0]], [0, selectedRow, selectedColumn]))} + ${displayNumber(matrixValue(trace.values[operation.inputs[1]], [0, selectedRow, selectedColumn]))} = ${displayNumber(outputValue)}`;
    if (operation.opcode === "POSITION_ADD") explanation += `<br>${displayNumber(matrixValue(trace.values[operation.inputs[0]], [0, selectedRow, selectedColumn]))} + ${displayNumber(matrixValue(trace.tensors[operation.frozen[0]], [selectedRow, selectedColumn]))} = ${displayNumber(outputValue)}`;
    if (operation.opcode === "HARDMAX_STE") explanation += `<br>argmax(${operation.inputs[0]}[0, ${selectedRow}, :]) = ${winner}; ${operation.output}[0,${selectedRow},${selectedColumn}] = ${selectedColumn === winner ? 1 : 0}`;
    if (operation.opcode === "ROW_CONCAT" && concat) explanation += `<br>${operation.output}[0,${selectedRow},${selectedColumn}] ← ${operation.inputs[concat.input]}[0,${concat.row},${selectedColumn}]`;
    if (operation.opcode === "ROW_ROUTE") { const raw = trace.operations[operationIndex]; const match = raw.equation.match(/[:, ](\d+):(\d+):(\d+)/); if (match) explanation += `<br>${operation.output}[0,${selectedRow},${selectedColumn}] ← ${operation.inputs[0]}[0,${Number(match[1]) + selectedRow * Number(match[3])},${selectedColumn}]`; }
    byId("cellExplanation").innerHTML = `<p>${explanation}</p>`;
  }

  function selectCell(row, column) { selectedRow = row; selectedColumn = column; renderOperation(); }
  function renderOperation() {
    const operation = trace.operations[operationIndex]; normalizeSelection(operation); const copy = COPY[language];
    byId("operationTitle").textContent = operationTitle(operation); byId("operationDescription").textContent = operationModel(operation.opcode) === "matmul" ? copy.matmulDesc : copy.genericDesc;
    byId("operationCounter").textContent = `${String(operation.index).padStart(2, "0")} / ${trace.operations.length} · ${operation.opcode}`;
    byId("numericEquation").textContent = operation.equation; byId("numericOperations").value = String(operationIndex);
    const flow = byId("matrixFlow"); flow.className = "matrix-flow"; flow.replaceChildren();
    byId("dotDimension").disabled = operationModel(operation.opcode) !== "matmul";
    if (operationModel(operation.opcode) === "matmul") renderMatmul(operation, flow);
    else if (operationModel(operation.opcode) === "attention") renderAttention(operation, flow);
    else renderGeneric(operation, flow);
  }

  function applyLanguage() {
    const copy = COPY[language]; document.documentElement.lang = language; byId("numeric-title").textContent = copy.title; byId("traceSubtitle").textContent = copy.subtitle;
    byId("operationLabel").textContent = copy.operation; byId("rowLabel").textContent = copy.row; byId("columnLabel").textContent = copy.column; byId("dimensionLabel").textContent = copy.dimension; byId("explanationTitle").textContent = copy.explanation;
    byId("languageRu").classList.toggle("active", language === "ru"); byId("languageEn").classList.toggle("active", language === "en");
    const picker = byId("numericOperations"); picker.replaceChildren(...trace.operations.map((operation, index) => { const option = document.createElement("option"); option.value = String(index); option.textContent = `${String(operation.index).padStart(2, "0")} · ${operationTitle(operation)} · ${operation.opcode}`; return option; })); renderOperation();
  }

  async function initialize() {
    const response = await fetch("./numeric_trace.json"); if (!response.ok) throw new Error(`numeric trace request failed: ${response.status}`); trace = await response.json();
    byId("numericOperations").addEventListener("change", (event) => { operationIndex = Number(event.target.value); selectedRow = 0; selectedColumn = 0; kStart = 0; renderOperation(); });
    byId("previousNumericOperation").addEventListener("click", () => { operationIndex = clamp(operationIndex - 1, 0, trace.operations.length - 1); selectedRow = 0; selectedColumn = 0; kStart = 0; renderOperation(); });
    byId("nextNumericOperation").addEventListener("click", () => { operationIndex = clamp(operationIndex + 1, 0, trace.operations.length - 1); selectedRow = 0; selectedColumn = 0; kStart = 0; renderOperation(); });
    byId("outputRow").addEventListener("input", (event) => { selectedRow = Number(event.target.value); renderOperation(); }); byId("outputColumn").addEventListener("input", (event) => { selectedColumn = Number(event.target.value); renderOperation(); });
    byId("dotDimension").addEventListener("input", (event) => { kStart = Number(event.target.value); renderOperation(); }); byId("languageRu").addEventListener("click", () => { language = "ru"; applyLanguage(); }); byId("languageEn").addEventListener("click", () => { language = "en"; applyLanguage(); }); applyLanguage();
  }
  initialize().catch((error) => { byId("numericEquation").textContent = error.message; });
}());
