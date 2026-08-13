import { useEffect, useState } from "react";
import { NativeTraceLaboratory } from "./TraceDebugger";
import { parseInferenceTrace, type InferenceTrace } from "./trace-schema";

export default function App() {
  const [trace, setTrace] = useState<InferenceTrace | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}traces/pawn-e2-e3.json`)
      .then(response => response.json())
      .then(value => setTrace(parseInferenceTrace(value)))
      .catch(reason => setError(String(reason)));
  }, []);

  if (error) return <main className="loading">Ошибка загрузки native-трассы: {error}</main>;
  if (!trace) return <main className="loading">Загружаю 17 MB реальной float64-трассы…</main>;

  return <NativeTraceLaboratory trace={trace} inspectorTitle="Арифметика ячейки" />;
}
