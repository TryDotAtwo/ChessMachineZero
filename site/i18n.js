(function i18nModule() {
  let language = "ru"; let loadError = ""; let traceState = "loading";
  const ready = {
    ru: "Position-микроскоп загружен: v45 декодирован без JavaScript chess replay.",
    en: "Position microscope loaded: v45 decoded without JavaScript chess replay.",
  };
  const copy = {
    ru: {
      navFullVm: "Полная VM", navMatrices: "45-op микроскоп", navBoundary: "Границы",
      eyebrow: "ПОЛНЫЙ RECURRENT TENSOR TRACE",
      title: "Один ход шахмат: 2877 generic tensor-операций",
      subtitle: "Запрос [e2,e4,белая пешка] + предыдущий context → история, новый LEGAL_SET, статус OK и следующий context.",
      fullVmTitle: "Полный переход VM, без JavaScript chess logic", fullOperation: "Операция полного графа",
      ssaGlossary: "vN — результат N-й tensor-операции; wN — неизменяемая frozen-матрица. Рядом всегда указано, какие данные хранит тензор и зачем он нужен именно здесь.",
      legacyTitle: "45-op микроскоп восстановления позиции",
      legacySummary: "Полный COO и произвольная координата для вложенного position subgraph",
      loading: "Загрузка position trace…", traceError: "Ошибка проверки position trace",
      fixtureTitle: "Фиксированная история токенов",
      fixtureNote: "Токены декодируются из fixture; браузер не применяет шахматные правила.",
      outputLabel: "ВЛОЖЕННЫЙ POSITION OUTPUT · v45 [1 × 64 × 128]",
      boardNote: "Доска — декодер one-hot position output; не JavaScript replay.",
      artifactTitle: "Провенанс position-экспорта", precision: "точность", sourceHash: "source hash",
      boundaryTitle: "Реальная граница исполнения",
      boundaryBody: "Production C++ исполняет 2877 generic tensor-op: routing, frozen GEMM, add, reshape/transpose, grouped GEMM, hardmax/STE и attention. Шахматных веток, board object, oracle или Python в runtime нет.",
      numericBody: "Вес хранится как packed E2M1/FP4 + FP32 scales; текущая загрузка и вычисление — FP32, TF32 отключён. Сайт показывает компактные exact-окна Python reference; native acceptance сравнивает полные выходы.",
      browserValidationBody: "Браузер не запускает .cmz и не вычисляет шахматы: он валидирует сохранённую топологию, feedback hashes и scalar proofs. Поэтому site trace — инспектор, не второй движок.",
      attentionBody: "Hard forward точный; backward — заданный STE/selected-softmax surrogate. Проверено, что градиент доходит до request, но полезность обучения и speedup над обычным движком пока не доказаны.",
      footer: "Полная recurrent VM исполняется нативно; сайт честно разделяет reference-intermediate trace и native full-output acceptance.",
    },
    en: {
      navFullVm: "Full VM", navMatrices: "45-op microscope", navBoundary: "Boundary",
      eyebrow: "FULL RECURRENT TENSOR TRACE",
      title: "One chess move: 2,877 generic tensor operations",
      subtitle: "Request [e2,e4,white pawn] + previous context → history, next LEGAL_SET, OK status, and next context.",
      fullVmTitle: "Complete VM transition, with no JavaScript chess logic", fullOperation: "Full-graph operation",
      ssaGlossary: "vN is the result of tensor operation N; wN is an immutable frozen matrix. The adjacent label states what the tensor stores and why this operation uses it.",
      legacyTitle: "45-op position-reconstruction microscope",
      legacySummary: "Complete COO and arbitrary coordinates for the nested position subgraph",
      loading: "Loading position trace…", traceError: "Position-trace validation error",
      fixtureTitle: "Fixed token history",
      fixtureNote: "Tokens are decoded from the fixture; the browser applies no chess rules.",
      outputLabel: "NESTED POSITION OUTPUT · v45 [1 × 64 × 128]",
      boardNote: "The board only decodes the one-hot position output; it is not JavaScript replay.",
      artifactTitle: "Position-export provenance", precision: "precision", sourceHash: "source hash",
      boundaryTitle: "Actual execution boundary",
      boundaryBody: "Production C++ executes 2,877 generic tensor ops: routing, frozen GEMMs, addition, reshape/transpose, grouped GEMM, hardmax/STE, and attention. Runtime has no chess branches, board object, oracle, or Python.",
      numericBody: "Weights are stored as packed E2M1/FP4 plus FP32 scales; current loading and compute are FP32 with TF32 disabled. The site shows compact exact Python-reference windows; native acceptance compares complete outputs.",
      browserValidationBody: "The browser neither runs .cmz nor computes chess: it validates stored topology, feedback hashes, and scalar proofs. The site trace is an inspector, not a second engine.",
      attentionBody: "Hard forward is exact; backward is the declared STE/selected-softmax surrogate. Gradient delivery to the request is verified, but learning utility and speedup over a conventional engine are not established.",
      footer: "The complete recurrent VM executes natively; the site explicitly separates reference intermediate traces from native full-output acceptance.",
    },
  };
  function failureText(nextLanguage = language, detail = loadError) {
    return detail ? `${copy[nextLanguage].traceError}: ${detail}` : copy[nextLanguage].traceError;
  }
  function statusText() {
    return traceState === "ready" ? ready[language] : traceState === "error" ? failureText() : copy[language].loading;
  }
  function renderStatus() {
    const status = document.getElementById("traceStatus"); if (status) status.textContent = statusText();
  }
  function apply(nextLanguage) {
    language = nextLanguage; document.documentElement.lang = language;
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      if (node.id !== "traceStatus") node.textContent = copy[language][node.dataset.i18n] || node.textContent;
    });
    const direct = language === "ru"
      ? ["Любая ячейка position-экспорта", "Тензор / значение", "строка", "столбец"]
      : ["Any position-export cell", "Tensor / value", "row", "column"];
    ["tensorReaderTitle", "matrixLabel", "matrixRowLabel", "matrixColumnLabel"].forEach((id, index) => {
      const node = document.getElementById(id); if (node) node.textContent = direct[index];
    });
    ["languageRu", "languageEn"].forEach((id) => {
      const button = document.getElementById(id);
      if (button) button.classList.toggle("active", id === `language${language[0].toUpperCase()}${language.slice(1)}`);
    });
    renderStatus();
  }
  function setReady() { traceState = "ready"; loadError = ""; renderStatus(); }
  function setLoadError(error) { traceState = "error"; loadError = String(error?.message || error || "unknown error"); apply(language); }
  function selectLanguage(nextLanguage) { apply(nextLanguage); window.dispatchEvent(new CustomEvent("trace-language", {detail: nextLanguage})); }
  function initialize() {
    document.getElementById("languageRu")?.addEventListener("click", () => selectLanguage("ru"));
    document.getElementById("languageEn")?.addEventListener("click", () => selectLanguage("en"));
  }
  const api = {copy, apply, failureText, setReady, setLoadError, initialize};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") { window.TraceI18n = api; initialize(); }
}());
