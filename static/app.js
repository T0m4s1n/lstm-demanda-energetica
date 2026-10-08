const COLUMNS = [
  ["demanda_mw", "mw"],
  ["temperatura_c", "°c"],
  ["humedad_pct", "humedad %"],
  ["viento_kmh", "viento"],
  ["radiacion_wm2", "radiación"],
  ["precipitacion_mm", "lluvia"],
  ["precio_kwh", "precio"],
  ["hora", "hora"],
  ["dia_semana", "día"],
  ["fin_semana", "fin sem."],
  ["festivo", "festivo"],
  ["mes", "mes"],
  ["periodo_dia", "periodo"],
];

const PERIODS = [
  ["dawn", "madrugada"],
  ["morning", "mañana"],
  ["afternoon", "tarde"],
  ["night", "noche"],
];

const VARIABLE_LABELS = {
  recent_demand: "demanda reciente",
  calendar: "calendario",
  weather: "clima",
  price: "precio",
  demanda_mw: "demanda",
  temperatura_c: "temperatura",
  humedad_pct: "humedad",
  viento_kmh: "viento",
  radiacion_wm2: "radiación",
  precipitacion_mm: "precipitación",
  precio_kwh: "precio",
  fin_semana: "fin de semana",
  festivo: "festivo",
  hour: "hora",
  day_of_week: "día de la semana",
  month: "mes",
  period_of_day: "periodo del día",
};

const state = {
  context: null,
  rows: [],
  observed: null,
  edited: false,
  predictFor: "",
};

const $ = (id) => document.getElementById(id);

function formatNumber(value, digits) {
  return Number(value).toLocaleString("es-CO", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function periodFromHour(hour) {
  hour = Number(hour);
  if (hour <= 5) return "dawn";
  if (hour <= 11) return "morning";
  if (hour <= 17) return "afternoon";
  return "night";
}

function selectedModel() {
  return $("model").value;
}

function windowOf(id) {
  return Number(String(id).split("_w").pop());
}

function fillModelSelect() {
  const select = $("model");
  const groups = [
    ["modelos lstm", state.context.models],
    ["referencia", state.context.baseline],
  ];
  select.innerHTML = "";
  for (const [title, rows] of groups) {
    const group = document.createElement("optgroup");
    group.label = title;
    for (const row of rows) {
      const option = document.createElement("option");
      option.value = row.id;
      const mae = row.MAE == null ? "" : ` · mae ${formatNumber(row.MAE, 1)} mw`;
      option.textContent = `${row.Model} · ${row.Window} h${mae}`.toLocaleLowerCase("es");
      if (row.id === state.context.recommended) option.selected = true;
      group.appendChild(option);
    }
    select.appendChild(group);
  }
}

function readTable() {
  return [...document.querySelectorAll("#window tbody tr")].map((tr) => {
    const row = { timestamp: tr.dataset.timestamp };
    for (const [key] of COLUMNS) {
      row[key] = tr.querySelector(`[data-col="${key}"]`).value;
    }
    return row;
  });
}

function drawSpark(rows, prediction) {
  const svg = $("spark");
  const values = rows.map((row) => Number(row.demanda_mw));
  const hasPrediction = Number.isFinite(prediction);
  if (hasPrediction) values.push(prediction);
  const width = 280;
  const height = 96;
  const pad = 8;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const xOf = (i) => pad + (i * (width - pad * 2)) / Math.max(values.length - 1, 1);
  const yOf = (v) => height - pad - ((v - min) / span) * (height - pad * 2);
  const line = values
    .slice(0, rows.length)
    .map((v, i) => `${i === 0 ? "M" : "L"}${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`)
    .join(" ");
  let extra = "";
  if (hasPrediction) {
    const i = values.length - 1;
    extra = `<line x1="${xOf(i - 1)}" y1="${yOf(values[i - 1])}" x2="${xOf(i)}" y2="${yOf(prediction)}" stroke="#1a56a0" stroke-width="1.5"/><circle cx="${xOf(i)}" cy="${yOf(prediction)}" r="3.5" fill="#1a56a0"/>`;
  }
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `<path d="${line}" fill="none" stroke="#243447" stroke-width="1.4"/>${extra}`;
}

function renderTable(rows) {
  const table = $("window");
  const head = `<tr><th>marca</th>${COLUMNS.map(([, title]) => `<th>${title}</th>`).join("")}</tr>`;
  const body = rows
    .map((row) => {
      const cells = COLUMNS.map(([key]) => {
        if (key === "periodo_dia") {
          const options = PERIODS.map(
            ([value, label]) => `<option value="${value}" ${row[key] === value ? "selected" : ""}>${label}</option>`
          ).join("");
          return `<td><select data-col="${key}">${options}</select></td>`;
        }
        const step = ["hora", "dia_semana", "mes", "fin_semana", "festivo"].includes(key) ? "1" : "any";
        return `<td><input data-col="${key}" type="number" step="${step}" value="${row[key]}"></td>`;
      }).join("");
      return `<tr data-timestamp="${row.timestamp}"><td class="stamp">${row.timestamp.slice(5, 16)}</td>${cells}</tr>`;
    })
    .join("");
  table.innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
  $("target-hour").textContent = state.predictFor ? `hora a predecir: ${state.predictFor}` : "";
  drawSpark(rows, null);
}

async function loadWindow() {
  const window = windowOf(selectedModel());
  const response = await fetch(`/api/window?window=${window}`);
  const data = await response.json();
  if (!response.ok) {
    showError((data.error || "no se pudo cargar la ventana.").toLocaleLowerCase("es"));
    return;
  }
  state.rows = data.rows;
  state.observed = data.observed_mw;
  state.predictFor = data.predict_for;
  state.edited = false;
  renderTable(data.rows);
  $("figure").textContent = "—";
  $("notice").textContent = data.observed_mw == null
    ? "ventana histórica cargada. la hora siguiente no está en la serie limpia."
    : `referencia de la serie limpia para esa hora: ${formatNumber(data.observed_mw, 1)} mw. si edita la ventana, esa referencia deja de aplicar.`;
  $("notice").classList.remove("error");
}

function showError(text) {
  $("notice").textContent = text;
  $("notice").classList.add("error");
}

function alignCalendar() {
  for (const tr of document.querySelectorAll("#window tbody tr")) {
    const hour = tr.querySelector('[data-col="hora"]').value;
    const weekday = Number(tr.querySelector('[data-col="dia_semana"]').value);
    tr.querySelector('[data-col="periodo_dia"]').value = periodFromHour(hour);
    tr.querySelector('[data-col="fin_semana"]').value = weekday >= 5 ? 1 : 0;
  }
  state.edited = true;
}

async function predict() {
  const rows = readTable();
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: selectedModel(), rows }),
  });
  const data = await response.json();
  if (!response.ok) {
    showError((data.error || "no se pudo predecir.").toLocaleLowerCase("es"));
    return;
  }
  $("figure").textContent = formatNumber(data.prediction_mw, 1);
  $("notice").classList.remove("error");
  const model = [...state.context.models, ...state.context.baseline].find((row) => row.id === data.model);
  const detail = model && model.MAE != null
    ? `mae de prueba de este modelo: ${formatNumber(model.MAE, 2)} mw · rmse ${formatNumber(model.RMSE, 2)} · r² ${formatNumber(model.R2, 3)}.`
    : "";
  const reference = !state.edited && state.observed != null
    ? ` referencia observada: ${formatNumber(state.observed, 1)} mw.`
    : "";
  const lead = data.kind === "persistence"
    ? "persistencia: repite la última demanda."
    : "predicción lstm de la hora siguiente.";
  $("notice").textContent = `${lead} ${detail}${reference}`;
  drawSpark(rows, data.prediction_mw);
}

function renderResults() {
  const body = $("results-body");
  const rows = state.context.results.filter((row) => row.key !== "persistence");
  body.innerHTML = rows
    .map((row) => {
      const mark = row.id === state.context.recommended ? "recommended" : "";
      return `<tr class="${mark}">
        <td>${String(row.Model).toLocaleLowerCase("es")}</td>
        <td>${row.Window}</td>
        <td>${formatNumber(row.MAE, 2)}</td>
        <td>${formatNumber(row.MSE, 1)}</td>
        <td>${formatNumber(row.RMSE, 2)}</td>
        <td>${formatNumber(row.MAPE, 2)}</td>
        <td>${formatNumber(row.R2, 4)}</td>
        <td>${Math.round(row.Epochs)}</td>
      </tr>`;
    })
    .join("");
  const baseline = state.context.baseline
    .map((row) => `${row.Window} h: mae ${formatNumber(row.MAE, 2)} mw, rmse ${formatNumber(row.RMSE, 2)}, r² ${formatNumber(row.R2, 3)}`)
    .join(" · ");
  $("baseline-note").textContent = `persistencia (la hora siguiente igual a la última): ${baseline}.`;
}

function renderCharts() {
  const captions = {
    "loss_train_validation.png": "mse de entrenamiento y validación, ambos sin dropout. la línea punteada vertical es la época restaurada por early stopping.",
    "actual_vs_predicted.png": "conjunto de prueba del modelo recomendado. arriba, los últimos 14 días. abajo, todas las horas de prueba.",
    "prediction_error.png": "error = predicho menos real, en mw, para el modelo recomendado.",
    "model_comparison.png": "mae y rmse de prueba de las tres arquitecturas en cada ventana.",
    "temporal_attention.png": "peso medio que la lstm con atención asigna a cada hora de una ventana de 48 h, en una muestra de prueba.",
  };
  $("gallery").innerHTML = state.context.charts
    .map(
      (name) => `<figure>
        <img src="/charts/${name}" alt="${captions[name] || name}">
        <figcaption>${captions[name] || name}</figcaption>
      </figure>`
    )
    .join("");
}

function escapeHtml(text) {
  return String(text).replace(/[&<>]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[char]);
}

function renderAnalysis() {
  const block = $("analysis");
  const analysis = state.context.analysis;
  if (!analysis) {
    block.hidden = true;
    return;
  }
  block.hidden = false;
  $("analysis-list").innerHTML = analysis.questions
    .map((item) => {
      const paragraphs = String(item.answer)
        .split(/\n+/)
        .map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`)
        .join("");
      return `<article class="question"><h3>${escapeHtml(item.number)}. ${escapeHtml(item.question)}</h3>${paragraphs}</article>`;
    })
    .join("");
}

function renderCleaning() {
  const cleaning = state.context.cleaning;
  const split = state.context.split;
  $("counts").innerHTML = [
    `${cleaning.original_rows} filas originales`,
    `${cleaning.exact_duplicates_removed} duplicados exactos`,
    `${cleaning.rows_with_missing_measurements} con faltantes`,
    `${cleaning.rows_with_any_anomaly} con anomalía`,
    `${cleaning.rows_dropped_missing_or_anomaly} filas eliminadas`,
    `${cleaning.clean_rows} horas útiles`,
    `${cleaning.gaps_longer_than_one_hour} huecos`,
  ].map((text) => `<span>${text}</span>`).join("");
  $("split").textContent =
    `entrenamiento ${split.train_start} → ${split.train_end} (${split.train_rows} h). ` +
    `validación ${split.val_start} → ${split.val_end} (${split.val_rows} h). ` +
    `prueba ${split.test_start} → ${split.test_end} (${split.test_rows} h).`;
  const titles = {
    demand_mw: "demanda",
    temperature_c: "temperatura",
    humidity_pct: "humedad",
    wind_kmh: "viento",
    price_kwh: "precio",
    radiation_and_precipitation: "radiación y precipitación",
    missing_values: "valores faltantes",
    period_of_day: "periodo del día",
  };
  $("rules").innerHTML = Object.entries(cleaning.threshold_reasons)
    .map(([key, text]) => `<article><h3>${titles[key] || key}</h3><p>${String(text).toLocaleLowerCase("es")}</p></article>`)
    .join("");
}

function importanceTable(rows) {
  return `<table class="results-table"><thead><tr><th>variable</th><th>aumento del mae (mw)</th></tr></thead><tbody>
    ${rows.map((row) => `<tr><td>${VARIABLE_LABELS[row.variable] || row.variable}</td><td>${formatNumber(row.mae_increase_mw, 2)}</td></tr>`).join("")}
    </tbody></table>`;
}

function renderImportance() {
  const container = $("importance");
  const groups = state.context.group_importance || [];
  const detail = state.context.importance || [];
  if (!groups.length && !detail.length) {
    container.hidden = true;
    return;
  }
  container.hidden = false;
  container.innerHTML = `<h3>influencia en el modelo recomendado</h3>
    <p class="meta">aumento del mae de prueba al permutar cada bloque de la ventana. un aumento mayor indica que la predicción dependía más de esas entradas. no es un efecto causal.</p>
    ${groups.length ? importanceTable(groups) : ""}
    ${detail.length ? `<h3>por variable</h3>${importanceTable(detail)}` : ""}`;
}

async function start() {
  const response = await fetch("/api/context");
  state.context = await response.json();
  fillModelSelect();
  renderResults();
  renderCharts();
  renderAnalysis();
  renderCleaning();
  renderImportance();
  await loadWindow();
  $("model").addEventListener("change", loadWindow);
  $("reload").addEventListener("click", loadWindow);
  $("align").addEventListener("click", alignCalendar);
  $("predict").addEventListener("click", predict);
  $("window").addEventListener("input", () => {
    state.edited = true;
  });
}

start();
