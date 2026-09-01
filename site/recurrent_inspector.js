(function recurrentInspectorModule() {
  "use strict";

  const MODELS = {
    ROW_ROUTE: "route",
    TOKEN_PROJECT: "matmul",
    OUTPUT_PROJECT: "matmul",
    POSITION_ADD: "add",
    HULL_ATTN_2D: "attention",
    RESIDUAL_ADD: "add",
    HARDMAX_STE: "argmax",
    FROZEN_EXPAND: "expand",
    ROW_CONCAT: "concat",
    MATRIX_TRANSPOSE: "transpose",
    MATRIX_RESHAPE: "reshape",
    MATRIX_MATMUL: "matmul",
    GROUPED_MATRIX_MATMUL: "grouped_matmul",
  };
  const PROOF_KIND = {
    route: "route", matmul: "matmul", add: "add", attention: "attention",
    argmax: "argmax", expand: "expand", concat: "concat", transpose: "transpose",
    reshape: "reshape", grouped_matmul: "grouped_matmul",
  };
  const STATUS_BY_CHANNEL = {
    89: "OK", 90: "ILLEGAL_MOVE", 91: "WHITE_WIN", 92: "BLACK_WIN",
    93: "DRAW", 94: "HISTORY_OVERFLOW",
  };
  const COPY = {
    ru: {
      loading: "Загрузка полного recurrent trace…", ready: "Полный trace проверен",
      stage: "Этап", operation: "Операция", input: "Вход", output: "Выход",
      frozen: "Frozen-матрицы этой операции", proof: "Точная арифметика выбранной ячейки",
      noFrozen: "Frozen-матрица не нужна: операция связывает промежуточные тензоры.",
      previous: "Предыдущая", next: "Следующая", request: "запрос", status: "статус",
      history: "полуходов истории", legal: "легальных ходов", feedback: "Выход context → следующий вход context",
      feedbackEdges: "output→input рёбер", contextBindings: "привязок context_0",
      native: "Нативная проверка", reference: "Числа ниже — exact Python reference того же .cmz; native проверен отдельно по полным выходам.",
      invalid: "Ошибка полного trace",
    },
    en: {
      loading: "Loading full recurrent trace…", ready: "Full trace validated",
      stage: "Stage", operation: "Operation", input: "Input", output: "Output",
      frozen: "Frozen matrices used by this operation", proof: "Exact arithmetic for the selected cell",
      noFrozen: "No frozen matrix is needed: this operation connects intermediate tensors.",
      previous: "Previous", next: "Next", request: "request", status: "status",
      history: "history plies", legal: "legal moves", feedback: "Output context → next input context",
      feedbackEdges: "output→input edges", contextBindings: "context_0 bindings",
      native: "Native acceptance", reference: "Numbers below are the exact Python reference of the same .cmz; native full outputs were verified separately.",
      invalid: "Full-trace validation error",
    },
  };

  function object(value) { return value !== null && typeof value === "object" && !Array.isArray(value); }
  function finite(value) { return typeof value === "number" && Number.isFinite(value); }
  function requireCondition(condition, message) { if (!condition) throw new Error(message); }
  function hash(value) { return typeof value === "string" && /^[0-9a-f]{64}$/.test(value); }
  function operationModel(opcode) { return MODELS[opcode] || null; }

  function validateWindow(window, label) {
    requireCondition(object(window) && Array.isArray(window.shape) && window.shape.length === 2, `${label}: window shape`);
    requireCondition(Array.isArray(window.selected) && window.selected.length === 2, `${label}: selected cell`);
    requireCondition(Number.isInteger(window.row_start) && Number.isInteger(window.column_start), `${label}: window origin`);
    requireCondition(Array.isArray(window.values) && window.values.length > 0, `${label}: window values`);
    window.values.forEach((row) => {
      requireCondition(Array.isArray(row) && row.length > 0 && row.every(finite), `${label}: finite rectangular values`);
      requireCondition(row.length === window.values[0].length, `${label}: rectangular values`);
    });
  }

  function selectedWindowValue(window, label) {
    const row = window.selected[0] - window.row_start;
    const column = window.selected[1] - window.column_start;
    requireCondition(row >= 0 && row < window.values.length, `${label}: selected row outside window`);
    requireCondition(column >= 0 && column < window.values[0].length, `${label}: selected column outside window`);
    return window.values[row][column];
  }

  function validateMoveTriples(moves, label) {
    requireCondition(Array.isArray(moves), `${label}: move list`);
    moves.forEach((move) => requireCondition(
      Array.isArray(move) && move.length === 3 && move.every((channel) => Number.isInteger(channel) && channel > 0 && channel < 128),
      `${label}: malformed move triple`,
    ));
  }

  function validateContextSummary(summary, label) {
    requireCondition(object(summary) && hash(summary.context_sha256), `${label}: context hash`);
    requireCondition(STATUS_BY_CHANNEL[summary.status_channel] === summary.status, `${label}: status/channel mismatch`);
    validateMoveTriples(summary.history_moves, `${label}.history`);
    validateMoveTriples(summary.legal_moves, `${label}.legal`);
    requireCondition(summary.history_plies === summary.history_moves.length, `${label}: history count`);
    requireCondition(summary.legal_count === summary.legal_moves.length && summary.legal_count <= 256, `${label}: legal count`);
  }

  function validateProof(proof, output, label) {
    requireCondition(object(proof) && PROOF_KIND[proof.kind], `${label}: proof kind`);
    requireCondition(finite(proof.result) && proof.result === output.value, `${label}: proof result differs from output`);
    if (proof.kind === "matmul" || proof.kind === "grouped_matmul") {
      requireCondition(Array.isArray(proof.terms) && Number.isInteger(proof.contraction), `${label}: dot-product terms`);
      requireCondition(
        Number.isInteger(proof.omitted_zero_terms) && proof.omitted_zero_terms === proof.contraction - proof.terms.length,
        `${label}: omitted zero-term count`,
      );
      const indices = new Set();
      proof.terms.forEach((term) => {
        requireCondition(object(term) && Number.isInteger(term.k) && finite(term.left) && finite(term.right) && finite(term.product), `${label}: malformed dot term`);
        requireCondition(term.k >= 0 && term.k < proof.contraction && !indices.has(term.k), `${label}: dot-term index`);
        indices.add(term.k);
        requireCondition(Math.fround(term.left * term.right) === term.product, `${label}: forged dot product`);
      });
      const sum = proof.terms.reduce((total, term) => total + term.product, 0);
      requireCondition(Math.abs(sum - proof.result) <= 1e-5 * Math.max(1, Math.abs(proof.result)), `${label}: dot-product sum`);
      if (proof.kind === "grouped_matmul") requireCondition(Number.isInteger(proof.group) && proof.group >= 0, `${label}: group`);
    }
    if (proof.kind === "argmax") requireCondition(proof.operator === "ARGMAX" && Number.isInteger(proof.winner), `${label}: ARGMAX`);
    if (proof.kind === "attention") {
      requireCondition(String(proof.operator).includes("ARGMAX") && Array.isArray(proof.score_terms), `${label}: attention proof`);
    }
  }

  function validateTrace(trace) {
    try {
      requireCondition(object(trace) && trace.schema_version === 1 && trace.artifact === "recurrent_frozen_vm_v1", "schema/version");
      requireCondition(trace.operation_count === 2877 && Array.isArray(trace.operations) && trace.operations.length === trace.operation_count, "operation coverage");
      requireCondition(trace.tensor_count === 192 && Array.isArray(trace.runtime_opcodes), "tensor/opcode metadata");
      requireCondition(Array.isArray(trace.stages) && trace.stages.length > 0, "stages");
      let nextOperation = 1;
      const stageIds = new Set();
      trace.stages.forEach((stage) => {
        requireCondition(object(stage) && typeof stage.id === "string" && !stageIds.has(stage.id), "stage identity");
        requireCondition(stage.first_operation === nextOperation && stage.last_operation >= stage.first_operation, "stage gap/overlap");
        requireCondition(stage.operation_count === stage.last_operation - stage.first_operation + 1, "stage count");
        requireCondition(object(stage.ru) && object(stage.en) && stage.ru.name && stage.en.name && stage.ru.purpose && stage.en.purpose, "stage bilingual semantics");
        stageIds.add(stage.id); nextOperation = stage.last_operation + 1;
      });
      requireCondition(nextOperation === trace.operation_count + 1, "stage tail");

      const values = new Set(["v0"]);
      trace.operations.forEach((operation, offset) => {
        const label = `operation ${offset + 1}`;
        requireCondition(object(operation) && operation.index === offset + 1, `${label}: index`);
        const model = operationModel(operation.opcode);
        requireCondition(model && trace.runtime_opcodes.includes(operation.opcode), `${label}: opcode`);
        requireCondition(stageIds.has(operation.stage), `${label}: stage`);
        requireCondition(Array.isArray(operation.inputs) && operation.inputs.every((id) => values.has(id)), `${label}: undefined input`);
        requireCondition(operation.output === `v${offset + 1}` && !values.has(operation.output), `${label}: SSA output`);
        requireCondition(Array.isArray(operation.output_shape) && operation.output_shape[0] === "B", `${label}: output shape`);
        requireCondition(Array.isArray(operation.attributes) && operation.attributes.every(Number.isInteger), `${label}: attributes`);
        requireCondition(object(operation.semantics) && object(operation.semantics.ru) && object(operation.semantics.en), `${label}: semantics`);
        requireCondition(operation.semantics.ru.name && operation.semantics.ru.purpose && operation.semantics.en.name && operation.semantics.en.purpose, `${label}: bilingual semantics`);
        requireCondition(Array.isArray(operation.frozen), `${label}: frozen list`);
        operation.frozen.forEach((record) => requireCondition(record.id && record.name && object(record.semantics), `${label}: frozen meaning`));
        requireCondition(object(operation.sample) && object(operation.sample.output) && Array.isArray(operation.sample.cards), `${label}: sample`);
        requireCondition(operation.sample.cards.length > 0 && finite(operation.sample.output.value), `${label}: sample values`);
        validateWindow(operation.sample.output.window, `${label}.output`);
        requireCondition(
          selectedWindowValue(operation.sample.output.window, `${label}.output`) === operation.sample.output.value,
          `${label}: selected output cell differs from declared value`,
        );
        operation.sample.cards.forEach((card, index) => validateWindow(card, `${label}.card${index}`));
        requireCondition(operation.sample.proof.kind === model, `${label}: opcode/proof mismatch`);
        validateProof(operation.sample.proof, operation.sample.output, label);
        values.add(operation.output);
      });
      requireCondition(trace.final_output === `v${trace.operation_count}`, "final output");
      const fixture = trace.fixture;
      requireCondition(object(fixture) && Array.isArray(fixture.request) && fixture.request.length === 3, "fixture request");
      validateMoveTriples([fixture.request], "fixture.request");
      validateContextSummary(fixture.prior, "fixture.prior");
      validateContextSummary(fixture.output, "fixture.output");
      requireCondition(
        fixture.prior.legal_moves.some((move) => move.every((channel, index) => channel === fixture.request[index])),
        "fixture request absent from prior LEGAL_SET",
      );
      requireCondition(
        fixture.output.history_plies === fixture.prior.history_plies + 1 &&
        fixture.output.history_moves.at(-1).every((channel, index) => channel === fixture.request[index]),
        "fixture history append",
      );
      requireCondition(fixture.feedback.output_context_sha256 === fixture.output.context_sha256 && fixture.feedback.next_input_context_sha256 === fixture.output.context_sha256, "feedback identity");
      requireCondition(object(trace.provenance) && hash(trace.provenance.artifact_sha256) && hash(trace.provenance.source_sha256), "provenance");
      requireCondition(trace.provenance.native_intermediate_capture === false, "native capture label");
      return {ok: true};
    } catch (error) { return {ok: false, error: String(error.message || error)}; }
  }

  function displayNumber(value) {
    if (Number.isInteger(value)) return String(value);
    return Number(value).toPrecision(7).replace(/0+$/, "").replace(/\.$/, "");
  }

  function proofEquation(proof, language = "ru") {
    if (proof.kind === "matmul") {
      const terms = proof.terms.length ? proof.terms.map((term) => `${displayNumber(term.left)}×${displayNumber(term.right)}`).join(" + ") : "0";
      const omitted = language === "ru" ? `${proof.omitted_zero_terms} нулевых произведений опущено` : `${proof.omitted_zero_terms} zero products omitted`;
      return language === "ru" ? `строка × столбец: ${terms} = ${displayNumber(proof.result)}; ${omitted}` : `row × column: ${terms} = ${displayNumber(proof.result)}; ${omitted}`;
    }
    if (proof.kind === "grouped_matmul") {
      const terms = proof.terms.length ? proof.terms.map((term) => `${displayNumber(term.left)}×${displayNumber(term.right)}`).join(" + ") : "0";
      const omitted = language === "ru" ? `${proof.omitted_zero_terms} нулевых произведений опущено` : `${proof.omitted_zero_terms} zero products omitted`;
      return language === "ru" ? `группа ${proof.group}: строка × столбец = ${terms} = ${displayNumber(proof.result)}; ${omitted}` : `group ${proof.group}: row × column = ${terms} = ${displayNumber(proof.result)}; ${omitted}`;
    }
    if (proof.kind === "argmax") return `ARGMAX → column ${proof.winner}; selected ${proof.selected_column} = ${displayNumber(proof.result)}`;
    if (proof.kind === "attention") return `Q × Kᵀ → ARGMAX(row ${proof.winner_value_row}) → V = ${displayNumber(proof.result)}`;
    if (proof.kind === "add") return `${displayNumber(proof.left)} + ${displayNumber(proof.right)} = ${displayNumber(proof.result)}`;
    if (proof.kind === "route") return `source[${proof.source_index.join(",")}] → ${displayNumber(proof.result)}`;
    if (proof.kind === "concat") return `input ${proof.input + 1}, row ${proof.source_row} → ${displayNumber(proof.result)}`;
    if (proof.kind === "transpose") return `source[${proof.source_index.join(",")}] → transpose = ${displayNumber(proof.result)}`;
    if (proof.kind === "reshape") return `flat[${proof.flat_index}] = source[${proof.source_index.join(",")}] = ${displayNumber(proof.result)}`;
    if (proof.kind === "expand") return `frozen[${proof.source_index.join(",")}] → batch = ${displayNumber(proof.result)}`;
    return String(proof.result);
  }

  function readyStatus(loadedTrace, language = "ru") {
    return `${COPY[language].ready}: ${loadedTrace.operation_count} ops / ${loadedTrace.tensor_count} frozen tensors`;
  }

  const api = {validateTrace, proofEquation, operationModel, readyStatus};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window === "undefined" || typeof document === "undefined") return;

  let trace = null; let language = document.documentElement.lang === "en" ? "en" : "ru";
  let operationOffset = 0; let stageId = "request";
  const byId = (id) => document.getElementById(id);
  const element = (tag, className, text) => { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = text; return node; };

  function matrixCard(card) {
    const article = element("article", `full-matrix-card role-${card.role}`);
    const header = element("header");
    header.append(element("strong", "", card.name[language]), element("code", "", `${card.id} [${card.shape.join(" × ")}]`));
    const viewport = element("div", "full-matrix-viewport"); const table = element("table");
    const head = element("thead"); const headRow = element("tr"); headRow.append(element("th", "", ""));
    const columns = card.values[0].map((_, index) => card.column_start + index);
    columns.forEach((column) => headRow.append(element("th", column === card.selected[1] ? "selected-axis" : "", String(column))));
    head.append(headRow); table.append(head);
    const body = element("tbody");
    card.values.forEach((values, rowIndex) => {
      const globalRow = card.row_start + rowIndex; const tr = element("tr");
      tr.append(element("th", globalRow === card.selected[0] ? "selected-axis" : "", String(globalRow)));
      values.forEach((value, columnIndex) => {
        const globalColumn = card.column_start + columnIndex;
        const classes = [];
        if (["left_row", "query_row", "logits_row"].includes(card.role) && globalRow === card.selected[0]) classes.push("row-factor");
        if (["right_column", "key_column_transposed"].includes(card.role) && globalColumn === card.selected[1]) classes.push("column-factor");
        if (globalRow === card.selected[0] && globalColumn === card.selected[1]) classes.push("selected-cell");
        tr.append(element("td", classes.join(" "), displayNumber(value)));
      });
      body.append(tr);
    });
    table.append(body); viewport.append(table); article.append(header, viewport); return article;
  }

  function flowOperator(proof, index, total) {
    if (proof.kind === "argmax") return "ARGMAX";
    if (index === total - 1) return "=";
    if (proof.kind === "matmul" || proof.kind === "grouped_matmul") return "×";
    if (proof.kind === "add") return "+";
    if (proof.kind === "attention") return index === 0 ? "×" : "ARGMAX →";
    return "→";
  }

  function renderContext() {
    const fixture = trace.fixture; const copy = COPY[language];
    byId("contextInput").replaceChildren(
      element("small", "", `${copy.input} [1 × 2048 × 128]`),
      element("strong", "", `${copy.request}: [${fixture.request.join(", ")}] · ${fixture.request_uci}`),
      element("span", "", `${copy.status}: ${fixture.prior.status} · ${fixture.prior.history_plies} ${copy.history} · ${fixture.prior.legal_count} ${copy.legal}`),
      element("code", "", fixture.prior.context_sha256.slice(0, 16)),
    );
    byId("contextOutput").replaceChildren(
      element("small", "", `${copy.output} [1 × 2045 × 128]`),
      element("strong", "", `${copy.status}: ${fixture.output.status}`),
      element("span", "", `${fixture.output.history_plies} ${copy.history} · ${fixture.output.legal_count} ${copy.legal}`),
      element("code", "", fixture.output.context_sha256.slice(0, 16)),
    );
    byId("feedbackEdge").replaceChildren(
      element("b", "", "="), element("span", "", copy.feedback),
      element("code", "", fixture.feedback.output_context_sha256.slice(0, 16)),
    );
    const native = trace.provenance.native_acceptance;
    byId("fullVmEvidence").textContent = `${copy.native}: ${native.contexts} contexts · ${native.feedback_edges} ${copy.feedbackEdges} · ${native.context_0_bindings} ${copy.contextBindings} · gradient ${native.request_gradient_nonzero} nonzero · peak ${native.sampled_global_peak_mib_with_backward} MiB. ${copy.reference}`;
  }

  function operationsInStage() { return trace.operations.filter((operation) => operation.stage === stageId); }
  function renderStageButtons() {
    const container = byId("fullVmStages"); container.replaceChildren();
    trace.stages.forEach((stage) => {
      const button = element("button", stage.id === stageId ? "active" : ""); button.type = "button";
      button.append(element("strong", "", stage[language].name), element("small", "", `${stage.first_operation}–${stage.last_operation} · ${stage.operation_count} ops`));
      button.title = stage[language].purpose;
      button.addEventListener("click", () => { stageId = stage.id; operationOffset = stage.first_operation - 1; renderStageButtons(); renderOperationPicker(); renderOperation(); });
      container.append(button);
    });
  }

  function renderOperationPicker() {
    const picker = byId("fullVmOperations"); const operations = operationsInStage();
    picker.replaceChildren(...operations.map((operation) => {
      const option = element("option", "", `${String(operation.index).padStart(4, "0")} · ${operation.semantics[language].name} · ${operation.opcode}`);
      option.value = String(operation.index - 1); return option;
    }));
    picker.value = String(operationOffset);
  }

  function renderOperation() {
    const operation = trace.operations[operationOffset]; const sample = operation.sample;
    byId("fullVmOperationTitle").textContent = operation.semantics[language].name;
    byId("fullVmOperationPurpose").textContent = operation.semantics[language].purpose;
    byId("fullVmOperationCounter").textContent = `${operation.index} / ${trace.operation_count}`;
    byId("fullVmEquation").textContent = operation.equation;
    const flow = byId("fullVmMatrixFlow"); flow.replaceChildren();
    sample.cards.forEach((card, index) => {
      if (index) flow.append(element("div", "full-flow-operator", flowOperator(sample.proof, index - 1, sample.cards.length - 1)));
      flow.append(matrixCard(card));
    });
    byId("fullVmProof").replaceChildren(
      element("h4", "", COPY[language].proof),
      element("p", "proof-equation", proofEquation(sample.proof, language)),
    );
    if (Array.isArray(sample.proof.terms) && sample.proof.terms.length) {
      const list = element("div", "proof-terms");
      sample.proof.terms.forEach((term) => list.append(element("code", "", `k=${term.k}: ${displayNumber(term.left)} × ${displayNumber(term.right)} = ${displayNumber(term.product)}`)));
      byId("fullVmProof").append(list);
    }
    const frozen = byId("fullVmFrozen"); frozen.replaceChildren(element("h4", "", COPY[language].frozen));
    if (!operation.frozen.length) frozen.append(element("p", "", COPY[language].noFrozen));
    operation.frozen.forEach((record) => {
      const item = element("div", "frozen-meaning");
      item.append(element("strong", "", record.semantics[language].name), element("code", "", `${record.id} [${record.shape.join(" × ")}]`), element("span", "", record.semantics[language].purpose));
      frozen.append(item);
    });
    byId("fullVmOperations").value = String(operationOffset);
  }

  function applyLanguage(next) {
    language = next; if (!trace) return;
    renderContext(); renderStageButtons(); renderOperationPicker(); renderOperation();
    byId("fullVmStatus").textContent = readyStatus(trace, language);
  }

  async function initialize() {
    const status = byId("fullVmStatus"); status.textContent = COPY[language].loading;
    try {
      const url = byId("downloadRecurrentTrace").href;
      const response = await fetch(url); if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const loaded = await response.json(); const validation = validateTrace(loaded);
      if (!validation.ok) throw new Error(validation.error);
      trace = loaded; stageId = trace.stages[0].id; operationOffset = 0;
      renderContext(); renderStageButtons(); renderOperationPicker(); renderOperation();
      status.textContent = readyStatus(trace, language);
      byId("fullVmOperations").addEventListener("change", (event) => { operationOffset = Number(event.target.value); renderOperation(); });
      byId("previousFullVmOperation").addEventListener("click", () => { operationOffset = Math.max(0, operationOffset - 1); stageId = trace.operations[operationOffset].stage; renderStageButtons(); renderOperationPicker(); renderOperation(); });
      byId("nextFullVmOperation").addEventListener("click", () => { operationOffset = Math.min(trace.operation_count - 1, operationOffset + 1); stageId = trace.operations[operationOffset].stage; renderStageButtons(); renderOperationPicker(); renderOperation(); });
      window.addEventListener("trace-language", (event) => applyLanguage(event.detail));
    } catch (error) {
      status.textContent = `${COPY[language].invalid}: ${String(error.message || error)}`;
      byId("fullVmVerification").classList.add("error");
    }
  }

  initialize();
}());
