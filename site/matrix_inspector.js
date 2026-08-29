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

  function highlightMode(opcode) {
    if (opcode === "TOKEN_PROJECT") return "row-column";
    if (opcode === "HARDMAX_STE") return "row-argmax";
    if (opcode === "HULL_ATTN_2D") return "attention";
    return "cell";
  }

  function flowClass(opcode) {
    return ["ROW_ROUTE", "FROZEN_EXPAND", "HARDMAX_STE"].includes(opcode) ? "matrix-flow word-operator-flow" : "matrix-flow";
  }

  function operationOptionText(index, title) { return `${String(index).padStart(2, "0")} · ${title}`; }
  function outputCellText(requestedLanguage, row, column, value) {
    return requestedLanguage === "ru"
      ? `Выход[batch 0, строка ${row}, столбец ${column}] = ${value}`
      : `Output[batch 0, row ${row}, column ${column}] = ${value}`;
  }
  function routeSourceRow(equation, outputRow) {
    const match = equation.match(/(?:rows=|[:, ]\s*)(\d+):(\d+):(\d+)/);
    return match ? Number(match[1]) + outputRow * Number(match[3]) : outputRow;
  }
  function tensorPurpose(opcode, kind, requestedLanguage) {
    const presentation = operationPresentation(opcode, requestedLanguage);
    if (kind === "input") return presentation?.input || "";
    if (kind === "frozen" && opcode === "FROZEN_EXPAND") return presentation?.input || "";
    if (kind === "frozen") return requestedLanguage === "ru"
      ? "Frozen-матрица из артефакта: неизменяемое смещение для каждой строки и каждого признака"
      : "Frozen artifact matrix: an immutable offset for every row and feature";
    return "";
  }

  function operationPresentation(opcode, requestedLanguage) {
    const presentations = {
      ru: {
        ROW_ROUTE: { operator: "ВЫБОР СТРОК", input: "История ходов и служебный контекст: строка = событие, столбец = токен", output: "Выбранные строки без изменения значений", equation: "Каждая выходная ячейка копируется из указанной строки входного тензора" },
        TOKEN_PROJECT: { operator: "МАТРИЧНОЕ УМНОЖЕНИЕ", input: "Входные активации: строка = обрабатываемый объект, столбец = входной признак", output: "Спроецированные активации: строка = тот же объект, столбец = выходной признак", equation: "Каждая выходная ячейка — скалярное произведение одной строки входа и одного столбца frozen-весов" },
        POSITION_ADD: { operator: "ПОЭЛЕМЕНТНОЕ СЛОЖЕНИЕ", input: "Динамические активации плюс frozen-смещение той же формы", output: "Сумма соответствующих ячеек", equation: "Каждая выходная ячейка равна входной ячейке плюс соответствующая ячейка frozen-матрицы" },
        RESIDUAL_ADD: { operator: "RESIDUAL-СЛОЖЕНИЕ", input: "Два динамических тензора одинаковой формы", output: "Поэлементная сумма двух путей", equation: "Каждая выходная ячейка равна сумме ячеек с теми же индексами в двух входах" },
        FROZEN_EXPAND: { operator: "КОПИРОВАНИЕ В BATCH", input: "Frozen-матрица исходных клеток: строка = клетка, столбец = токен содержимого", output: "Та же матрица для каждой партии в batch", equation: "Frozen-матрица не вычисляется из входа: она хранится в артефакте и копируется в batch без изменения значений" },
        ROW_CONCAT: { operator: "СКЛЕЙКА ПО СТРОКАМ", input: "Несколько последовательностей с одинаковыми столбцами признаков", output: "Одна последовательность: входы поставлены друг за другом", equation: "Выходная ячейка копируется из одного входа; меняется только номер строки из-за смещения" },
        HARDMAX_STE: { operator: "ARGMAX ПО СТРОКЕ", input: "Оценки вариантов в каждой строке", output: "One-hot: 1 только у максимального варианта", equation: "Для каждой строки: найти максимальную оценку → поставить 1 в её столбце, остальные значения обнулить" },
      },
      en: {
        ROW_ROUTE: { operator: "SELECT ROWS", input: "Move history and service context: row = event, column = token", output: "Selected rows with values unchanged", equation: "Each output cell is copied from the specified input-tensor row" },
        TOKEN_PROJECT: { operator: "MATRIX MULTIPLICATION", input: "Input activations: row = processed item, column = input feature", output: "Projected activations: row = the same item, column = output feature", equation: "Each output cell is the dot product of one input row and one frozen-weight column" },
        POSITION_ADD: { operator: "ELEMENTWISE ADD", input: "Dynamic activations plus a same-shaped frozen offset", output: "Sum of matching cells", equation: "Each output cell is the input cell plus the matching frozen-matrix cell" },
        RESIDUAL_ADD: { operator: "RESIDUAL ADD", input: "Two same-shaped dynamic tensors", output: "Elementwise sum of both paths", equation: "Each output cell is the sum of same-index cells from both inputs" },
        FROZEN_EXPAND: { operator: "COPY INTO BATCH", input: "Frozen initial-square matrix: row = square, column = content token", output: "The same matrix for every batch item", equation: "The frozen matrix is stored in the artifact, not computed from input; batch expansion copies its values unchanged" },
        ROW_CONCAT: { operator: "CONCATENATE ROWS", input: "Several sequences sharing the same feature columns", output: "One sequence with inputs placed one after another", equation: "An output cell is copied from one input; only its row index changes by an offset" },
        HARDMAX_STE: { operator: "ROW ARGMAX", input: "Candidate scores in each row", output: "One-hot: 1 only at the maximum candidate", equation: "For each row: find the maximum score → put 1 in its column and zero every other value" },
      },
    };
    return presentations[requestedLanguage][opcode] || null;
  }

  const api = { matrixValue, tokenMeaning, operationModel, dotProduct, concatSource, hardmaxWinner, highlightMode, flowClass, operationOptionText, outputCellText, routeSourceRow, tensorPurpose, operationPresentation };
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
    const header = document.createElement("header"); header.innerHTML = `<div><strong>${name}</strong>${options.purpose ? `<small>${options.purpose}</small>` : ""}</div><span>${shapeText(matrix)}</span>`; article.append(header);
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
        td.title = `${name}[${index.join(", ")}] = ${displayNumber(value)} · ${rowAxis} ${r} · ${columnAxis} ${col}${meaning}`;
        if (options.onSelect) { td.tabIndex = 0; td.addEventListener("mouseenter", () => options.onSelect(r, col)); td.addEventListener("click", () => options.onSelect(r, col)); }
        tr.append(td);
      } body.append(tr);
    }
    table.append(body); const viewport = document.createElement("div"); viewport.className = "matrix-table-viewport"; viewport.append(table); article.append(viewport); return article;
  }

  function sign(symbol) { const element = document.createElement("div"); element.className = "matrix-sign"; element.textContent = symbol; return element; }
  function valueName() { return COPY[language].input; }
  function selectedOutput(operation) { return trace.values[operation.output]; }
  function normalizeSelection(operation) {
    const output = selectedOutput(operation); selectedRow = clamp(selectedRow, 0, lastRows(output) - 1); selectedColumn = clamp(selectedColumn, 0, lastColumns(output) - 1);
    byId("outputRow").max = String(lastRows(output) - 1); byId("outputColumn").max = String(lastColumns(output) - 1); byId("outputRow").value = String(selectedRow); byId("outputColumn").value = String(selectedColumn);
  }

  function renderMatmul(operation, flow) {
    const left = trace.values[operation.inputs[0]]; const weightId = operation.frozen[0]; const weight = trace.tensors[weightId]; const output = selectedOutput(operation);
    const presentation = operationPresentation(operation.opcode, language);
    const product = dotProduct(left, weight, 0, selectedRow, selectedColumn); const firstActive = product.terms.find((term) => term.left !== 0 || term.right !== 0)?.k || 0;
    if (kStart === 0 && firstActive > 7) kStart = Math.floor(firstActive / 8) * 8; kStart = clamp(kStart, 0, Math.max(0, product.terms.length - 8));
    byId("dotDimension").max = String(Math.max(0, product.terms.length - 8)); byId("dotDimension").value = String(kStart); byId("dotDimensionValue").textContent = `${kStart}–${Math.min(product.terms.length - 1, kStart + 7)}`;
    flow.append(
      matrixCard(COPY[language].input, operation.inputs[0], left, { purpose: presentation.input, row: selectedRow, rowStart: selectedRow - 2, columnStart: kStart, highlightRow: selectedRow, rowAxis: COPY[language].move, columnAxis: COPY[language].token, tokenAxis: "column" }), sign("×"),
      matrixCard(`${COPY[language].weight}: ${semanticFrozenName(weight.name)}`, weightId, weight, { purpose: language === "ru" ? "Неизменяемые обученные коэффициенты: строка = входной признак, столбец = выходной признак" : "Immutable trained coefficients: row = input feature, column = output feature", rowStart: kStart, column: selectedColumn, highlightColumn: selectedColumn, rowAxis: COPY[language].token, columnAxis: COPY[language].pattern, tokenAxis: "row" }), sign("="),
      matrixCard(COPY[language].output, operation.output, output, { purpose: presentation.output, row: selectedRow, column: selectedColumn, selectedRow, selectedColumn, onSelect: selectCell, rowAxis: COPY[language].move, columnAxis: COPY[language].pattern })
    );
    const visibleTerms = product.terms.slice(kStart, kStart + 8); const formula = visibleTerms.map((term) => `<span class="dot-term"><i>k=${term.k}</i>${displayNumber(term.left)} × ${displayNumber(term.right)} = <b>${displayNumber(term.product)}</b></span>`).join("");
    byId("cellExplanation").innerHTML = `<p><strong>${outputCellText(language, selectedRow, selectedColumn, displayNumber(matrixValue(output, [0, selectedRow, selectedColumn])))}</strong></p><div class="dot-terms">${formula}</div><p>Σ k=0…${product.terms.length - 1} = <strong>${displayNumber(product.total)}</strong></p>`;
  }

  function renderAttention(operation, flow) {
    const q = trace.values[operation.inputs[0]]; const k = trace.values[operation.inputs[1]]; const v = trace.values[operation.inputs[2]]; const scores = trace.derived[operation.derived[0]]; const attention = trace.derived[operation.derived[1]]; const output = selectedOutput(operation);
    let winner = 0; for (let candidate = 0; candidate < lastColumns(attention); candidate += 1) if (matrixValue(attention, [0, selectedRow, candidate]) === 1) { winner = candidate; break; }
    flow.classList.add("attention-flow");
    flow.append(
      matrixCard("Queries Q", operation.inputs[0], q, { purpose: language === "ru" ? "Что ищет каждая выходная строка; 2 координаты на строку" : "What each output row searches for; 2 coordinates per row", row: selectedRow, highlightRow: selectedRow }), sign("×"),
      matrixCard("Keys K", operation.inputs[1], k, { purpose: language === "ru" ? "Адрес каждого кандидата; строка = кандидат, 2 столбца = координаты ключа" : "Address of each candidate; row = candidate, 2 columns = key coordinates", row: winner, highlightRow: winner }), sign("Q × Kᵀ"),
      matrixCard(COPY[language].scores, operation.derived[0], scores, { purpose: language === "ru" ? "Сходство каждого запроса с каждым кандидатом" : "Similarity of every query to every candidate", row: selectedRow, column: winner, rowStart: selectedRow - 2, columnStart: winner - 3, selectedRow, selectedColumn: winner }), sign(language === "ru" ? "ARGMAX\nПО СТРОКЕ" : "ROW\nARGMAX"),
      matrixCard(COPY[language].attention, operation.derived[1], attention, { purpose: language === "ru" ? "One-hot выбор одного кандидата для каждой строки запроса" : "One-hot selection of one candidate per query row", row: selectedRow, column: winner, rowStart: selectedRow - 2, columnStart: winner - 3, selectedRow, selectedColumn: winner }), sign("×"),
      matrixCard("Values V", operation.inputs[2], v, { purpose: language === "ru" ? "Полезные данные кандидатов, которые attention переносит в выход" : "Candidate payload copied to output by attention", row: winner, column: selectedColumn, rowStart: winner - 2, columnStart: selectedColumn - 3, highlightRow: winner }), sign("="),
      matrixCard(COPY[language].output, operation.output, output, { purpose: language === "ru" ? "Для каждой строки запроса — данные выбранного кандидата" : "Selected candidate data for every query row", row: selectedRow, column: selectedColumn, selectedRow, selectedColumn, onSelect: selectCell })
    );
    const qTerms = [0, 1].map((d) => `${displayNumber(matrixValue(q, [0, selectedRow, d]))}×${displayNumber(matrixValue(k, [0, winner, d]))}`).join(" + ");
    byId("cellExplanation").innerHTML = `<p>Q[${selectedRow},:] × K[${winner},:] = ${qTerms} = <strong>${displayNumber(matrixValue(scores, [0, selectedRow, winner]))}</strong></p><p>hardmax → ${COPY[language].selectedColumn} ${winner}; A[${selectedRow},${winner}] = 1</p><p>A[${selectedRow},:] × V[:,${selectedColumn}] = V[${winner},${selectedColumn}] = <strong>${displayNumber(matrixValue(output, [0, selectedRow, selectedColumn]))}</strong></p>`;
  }

  function renderGeneric(operation, flow) {
    const presentation = operationPresentation(operation.opcode, language);
    const matrices = operation.opcode === "FROZEN_EXPAND" ? [] : operation.inputs.map((id, index) => [operation.inputs.length > 1 ? `${valueName()} ${index + 1}` : valueName(), id, trace.values[id], "input"]);
    operation.frozen.forEach((id) => matrices.push([operation.opcode === "FROZEN_EXPAND" ? presentation.input : `${COPY[language].weight}: ${semanticFrozenName(trace.tensors[id].name)}`, id, trace.tensors[id], "frozen"]));
    operation.derived.forEach((id) => matrices.push([id.includes("attention") ? COPY[language].attention : COPY[language].scores, id, trace.derived[id], "derived"]));
    const concat = operation.opcode === "ROW_CONCAT" ? concatSource(operation.inputs.map((id) => trace.values[id]), selectedRow) : null;
    const routedRow = operation.opcode === "ROW_ROUTE" ? routeSourceRow(operation.equation, selectedRow) : selectedRow;
    const winner = operation.opcode === "HARDMAX_STE" ? hardmaxWinner(selectedOutput(operation), 0, selectedRow) : selectedColumn;
    matrices.forEach(([name, id, matrix, kind], index) => {
      if (index) flow.append(sign(operation.opcode.includes("ADD") ? "+" : "→"));
      const inputIndex = operation.inputs.indexOf(id);
      const contributing = !concat || inputIndex === concat.input;
      const matrixRow = concat && contributing ? concat.row : routedRow;
      const exactColumn = operation.opcode === "HARDMAX_STE" ? winner : selectedColumn;
      flow.append(matrixCard(name, id, matrix, {
        purpose: tensorPurpose(operation.opcode, kind, language),
        row: matrixRow,
        column: exactColumn,
        highlightRow: operation.opcode === "HARDMAX_STE" && contributing ? matrixRow : -1,
        selectedRow: contributing ? matrixRow : -1,
        selectedColumn: contributing ? exactColumn : -1,
      }));
    });
    flow.append(sign(flowClass(operation.opcode).includes("word-operator-flow") ? presentation.operator : "="), matrixCard(COPY[language].output, operation.output, selectedOutput(operation), { purpose: presentation.output, row: selectedRow, column: selectedColumn, selectedRow, selectedColumn, onSelect: selectCell }));
    const output = selectedOutput(operation); const outputValue = matrixValue(output, [0, selectedRow, selectedColumn]);
    let explanation = `${COPY[language].output}[batch 0, ${COPY[language].rows} ${selectedRow}, ${COPY[language].columns} ${selectedColumn}] = <strong>${displayNumber(outputValue)}</strong>`;
    if (operation.opcode === "RESIDUAL_ADD") explanation += `<br>${displayNumber(matrixValue(trace.values[operation.inputs[0]], [0, selectedRow, selectedColumn]))} + ${displayNumber(matrixValue(trace.values[operation.inputs[1]], [0, selectedRow, selectedColumn]))} = ${displayNumber(outputValue)}`;
    if (operation.opcode === "POSITION_ADD") explanation += `<br>${displayNumber(matrixValue(trace.values[operation.inputs[0]], [0, selectedRow, selectedColumn]))} + ${displayNumber(matrixValue(trace.tensors[operation.frozen[0]], [selectedRow, selectedColumn]))} = ${displayNumber(outputValue)}`;
    if (operation.opcode === "HARDMAX_STE") explanation += `<br>argmax(${COPY[language].rows} ${selectedRow}) = ${COPY[language].columns} ${winner}; ${COPY[language].columns} ${selectedColumn} → ${selectedColumn === winner ? 1 : 0}`;
    if (operation.opcode === "ROW_CONCAT" && concat) explanation += `<br>${COPY[language].output}: ${COPY[language].rows} ${selectedRow} ← ${COPY[language].input} ${concat.input + 1}: ${COPY[language].rows} ${concat.row}`;
    if (operation.opcode === "ROW_ROUTE") explanation += `<br>${COPY[language].output}: ${COPY[language].rows} ${selectedRow} ← ${COPY[language].input}: ${COPY[language].rows} ${routedRow}`;
    byId("cellExplanation").innerHTML = `<p>${explanation}</p>`;
  }

  function selectCell(row, column) { selectedRow = row; selectedColumn = column; renderOperation(); }
  function renderOperation() {
    const operation = trace.operations[operationIndex]; normalizeSelection(operation); const copy = COPY[language];
    byId("operationTitle").textContent = operationTitle(operation); byId("operationDescription").textContent = operationModel(operation.opcode) === "matmul" ? copy.matmulDesc : copy.genericDesc;
    const presentation = operationPresentation(operation.opcode, language);
    byId("operationCounter").textContent = `${String(operation.index).padStart(2, "0")} / ${trace.operations.length}`;
    byId("numericEquation").textContent = presentation ? presentation.equation : operationTitle(operation); byId("numericOperations").value = String(operationIndex);
    const flow = byId("matrixFlow"); flow.className = flowClass(operation.opcode); flow.replaceChildren();
    byId("dotDimension").disabled = operationModel(operation.opcode) !== "matmul";
    if (operationModel(operation.opcode) === "matmul") renderMatmul(operation, flow);
    else if (operationModel(operation.opcode) === "attention") renderAttention(operation, flow);
    else renderGeneric(operation, flow);
  }

  function applyLanguage() {
    const copy = COPY[language]; document.documentElement.lang = language; byId("numeric-title").textContent = copy.title; byId("traceSubtitle").textContent = copy.subtitle;
    byId("operationLabel").textContent = copy.operation; byId("rowLabel").textContent = copy.row; byId("columnLabel").textContent = copy.column; byId("dimensionLabel").textContent = copy.dimension; byId("explanationTitle").textContent = copy.explanation;
    byId("languageRu").classList.toggle("active", language === "ru"); byId("languageEn").classList.toggle("active", language === "en");
    const picker = byId("numericOperations"); picker.replaceChildren(...trace.operations.map((operation, index) => { const option = document.createElement("option"); option.value = String(index); option.textContent = operationOptionText(operation.index, operationTitle(operation)); return option; })); renderOperation();
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
