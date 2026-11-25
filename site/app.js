"use strict";

const metrics = {
  prop_alpha: { label: "PROP", name: "Пропорциональная доля" },
  ef_alpha: { label: "EF", name: "Отсутствие зависти" },
  ef1_alpha: { label: "EF1", name: "Приближение к EF1" },
  po_alpha: { label: "PO", name: "Приближение к Парето-оптимальности", efficiency: true },
  sw_ratio: { label: "SW", name: "Суммарная эффективность", efficiency: true },
  mms_alpha: { label: "MMS", name: "Гарантированная доля" },
  efx_alpha: { label: "EFX", name: "Приближение к EFX+" },
  sd_ef1: { label: "SD-EF1", name: "Доля выполнения SD-EF1" },
  nw_ratio: { label: "MNW", name: "Благосостояние по Нэшу", efficiency: true },
};
const paperAxes = ["prop_alpha", "ef_alpha", "ef1_alpha", "po_alpha", "sw_ratio", "mms_alpha"];
const extendedAxes = ["prop_alpha", "ef_alpha", "ef1_alpha", "efx_alpha", "sd_ef1",
  "po_alpha", "sw_ratio", "nw_ratio", "mms_alpha"];
const percent = new Intl.NumberFormat("ru", { style: "percent", maximumFractionDigits: 1 });
const number = new Intl.NumberFormat("ru", { maximumFractionDigits: 2 });
const rateLabel = value => value === null ? "н/д" : percent.format(value);
const get = id => document.getElementById(id);
const group = get("group");
const size = get("size");
const runSelect = get("run");
const refresh = get("refresh");
const syncStatus = get("sync-status");
const kinds = { goods: "Блага", chores: "Обязанности" };
const kindControls = [...document.querySelectorAll('input[name="kind"]')];
const radars = get("radars");
const rate = get("rate");
const settings = new WeakMap();
const observer = new ResizeObserver(entries => {
  for (const { target } of entries) drawRadar(target);
});

function element(tag, content, className) {
  const node = document.createElement(tag);
  if (content !== undefined) node.textContent = content;
  if (className) node.className = className;
  return node;
}

function drawRadar(canvas) {
  const { axes, values } = settings.get(canvas);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const x = width / 2;
  const y = height / 2;
  const r = Math.min(width / 2 - 54, height / 2 - 43);
  if (r <= 0) return;
  const step = 2 * Math.PI / axes.length;
  const angle = index => -Math.PI / 2 - step / 2 + step * index;
  const point = (index, fraction = 1, offset = 0) => [
    x + Math.cos(angle(index)) * (r * fraction + offset),
    y + Math.sin(angle(index)) * (r * fraction + offset),
  ];

  axes.forEach((key, i) => {
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.arc(x, y, r, angle(i) - step * 0.45, angle(i) + step * 0.45);
    ctx.closePath();
    ctx.fillStyle = values[key] === null ? "#e7ebe6" : metrics[key].efficiency ? "#f3c69f" : "#c6e2bb";
    ctx.fill();
  });

  ctx.strokeStyle = "#68817476";
  ctx.lineWidth = 0.75;
  ctx.setLineDash([2, 3]);
  for (const tick of [0.2, 0.4, 0.6, 0.8, 1]) {
    ctx.beginPath();
    ctx.arc(x, y, r * tick, 0, 2 * Math.PI);
    ctx.stroke();
  }
  axes.forEach((_, i) => {
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(...point(i));
    ctx.stroke();
  });
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.strokeStyle = "#789085";
  ctx.stroke();

  const missing = axes.findIndex(key => values[key] === null);
  const points = [];
  ctx.beginPath();
  let shouldConnect = false;
  for (let offset = 0; offset < axes.length; offset++) {
    const i = (missing + 1 + offset) % axes.length;
    const key = axes[i];
    const value = values[key];
    if (value === null) {
      shouldConnect = false;
      continue;
    }
    const [px, py] = point(i, Math.min(1, Math.max(0, value)));
    if (shouldConnect) ctx.lineTo(px, py);
    else ctx.moveTo(px, py);
    shouldConnect = true;
    points.push({ x: px, y: py, key });
  }
  if (missing === -1) {
    ctx.closePath();
    ctx.fillStyle = "#1b4e3412";
    ctx.fill();
  }
  ctx.strokeStyle = "#173d2b";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = "#173d2b";
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 2.7, 0, 2 * Math.PI);
    ctx.fill();
  }
  settings.get(canvas).points = points;

  ctx.textBaseline = "middle";
  axes.forEach((key, i) => {
    const [tx, ty] = point(i, 1, 13);
    const side = Math.cos(angle(i));
    ctx.textAlign = side < -0.22 ? "right" : side > 0.22 ? "left" : "center";
    ctx.font = "650 10.5px system-ui, sans-serif";
    ctx.fillStyle = "#1d3c2b";
    ctx.fillText(metrics[key].label, tx, ty - 5);
    ctx.font = "10.5px system-ui, sans-serif";
    ctx.fillStyle = "#405b4e";
    ctx.fillText(rateLabel(values[key]), tx, ty + 8);
  });
}

function radar(entry, axes) {
  const card = element("article", undefined, entry.is_model ? "radar-card model" : "radar-card");
  const canvas = element("canvas");
  const tip = element("div", undefined, "chart-tip");
  tip.hidden = true;
  canvas.setAttribute("role", "img");
  const labels = axes.map(key => metrics[key].label + " " + rateLabel(entry.values[key]));
  canvas.setAttribute("aria-label", entry.label + ". " + labels.join(", "));
  settings.set(canvas, { axes, values: entry.values, points: [] });
  canvas.addEventListener("pointermove", event => {
    const area = canvas.getBoundingClientRect();
    const p = settings.get(canvas).points.find(({ x, y }) =>
      Math.hypot(event.clientX - area.left - x, event.clientY - area.top - y) < 11);
    tip.hidden = !p;
    if (p) tip.textContent = metrics[p.key].name + " · " + rateLabel(entry.values[p.key]);
  });
  canvas.addEventListener("pointerleave", () => { tip.hidden = true; });
  card.append(element("h2", entry.label), canvas, tip);
  return card;
}

function bars(entries) {
  return entries.map(entry => {
    const value = entry.values[rate.value];
    const row = element("div");
    const header = element("div", undefined, "bar-head");
    header.append(element("span", entry.label), element("strong", rateLabel(value)));
    const track = element("div", undefined, "track");
    track.setAttribute("aria-hidden", "true");
    const fill = element("span", undefined, entry.is_model ? "bar-fill model" : "bar-fill");
    fill.style.width = Math.min(100, Math.max(0, value * 100)) + "%";
    track.append(fill);
    row.append(header, track);
    return row;
  });
}

function table(entries) {
  return entries.map(entry => {
    const row = element("tr", undefined, entry.is_model ? "model" : undefined);
    const method = element("th", entry.label);
    method.scope = "row";
    const cells = ["ef1", "sd_ef1", "efx"].map(key => element("td", rateLabel(entry.values[key])));
    const total = entry.values.total === null ? "н/д" : number.format(entry.values.total);
    row.append(method, ...cells, element("td", total));
    return row;
  });
}

function downloadLink(path, label) {
  const link = element("a", label);
  link.href = path;
  if (path.startsWith("https://")) {
    link.target = "_blank";
    link.rel = "noopener";
  } else {
    link.download = "";
  }
  return link;
}

const repository = "EvgeniiZhidkov/delezhii_project";
const api = "https://api.github.com/repos/" + repository;
const fileTypes = /^(results\.csv|manifest\.json|summary\.csv|radar_summary\.csv|bootstrap\.csv|(?:paper|extended)_radar_[12]\.pdf|scaling_summary\.csv|scaling_(?:paper|n|m)_(?:goods|chores)\.png|no_efx_chores\.csv)$/;
const groupLabels = {
  paper: "Общая шкала (paper)",
  uniform: "Независимые оценки (uniform)",
  "r=2": "Два уровня (1 и 2)",
  "r=5": "Два уровня (1 и 5)",
  "r=10": "Два уровня (1 и 10)",
  no_ef: "Без EF",
  no_efx: "Без EFX",
};
const fields = [...Object.keys(metrics), "ef1", "efx", "po", "total"];
const compareNames = new Intl.Collator("ru", { numeric: true }).compare;
const catalogueFile = document.currentScript.dataset.results || "data.json";

function catalog(blobs, origin = "") {
  const folders = new Map();
  for (const { path, sha } of blobs) {
    const match = /^runs\/(?:published\/)?([^/]+)\/([^/]+)$/.exec(path);
    if (!match) continue;
    const [, name, filename] = match;
    if (name === "published" || !fileTypes.test(filename)) continue;
    const id = path.slice(0, -filename.length - 1);
    if (!folders.has(id)) folders.set(id, { id, label: name.replaceAll("_", " · "), files: {}, versions: [] });
    const folder = folders.get(id);
    folder.files[filename] = origin + path.split("/").map(encodeURIComponent).join("/");
    folder.versions.push(sha ? path + ":" + sha : path);
  }
  return [...folders.values()].filter(({ files }) =>
    files["results.csv"] || files["scaling_summary.csv"] || files["no_efx_chores.csv"])
    .map(folder => ({ ...folder, version: folder.versions.sort().join("\n") }))
    .sort((a, b) => Number(Boolean(b.files["results.csv"])) - Number(Boolean(a.files["results.csv"]))
      || compareNames(a.label, b.label));
}

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error("HTTP " + response.status);
  return response;
}

function parseCsv(csv) {
  const rows = [];
  let row = [];
  let value = "";
  for (const [token, quoted, plain] of csv.replace(/^\uFEFF/, "").matchAll(/"((?:[^"]|"")*)"|([^",\r\n]+)|\r\n|[,\r\n]/g)) {
    if (token === ",") {
      row.push(value);
      value = "";
    } else if (token[0] === "\r" || token[0] === "\n") {
      if (row.length || value !== "") rows.push([...row, value]);
      row = [];
      value = "";
    } else {
      value = quoted === undefined ? plain : quoted.replaceAll('""', '"');
    }
  }
  if (row.length || value !== "") rows.push([...row, value]);
  const headers = rows.shift() || [];
  return rows.map(row => Object.fromEntries(headers.map((header, i) => [header, row[i] ?? ""])));
}

async function fetchCsv(url) {
  return parseCsv(await (await request(url)).text());
}

function numeric(value) {
  if (value === "True") return 1;
  if (value === "False") return 0;
  if (value === "" || value == null) return null;
  const result = Number(value);
  return Number.isFinite(result) ? result : null;
}

function average(rows, key) {
  let sum = 0;
  let count = 0;
  for (const row of rows) {
    const value = numeric(row[key]);
    if (value === null) continue;
    sum += value;
    count++;
  }
  return count ? sum / count : null;
}

function groupBy(rows, key) {
  const groups = new Map();
  for (const row of rows) {
    const value = key(row);
    if (!groups.has(value)) groups.set(value, []);
    groups.get(value).push(row);
  }
  return groups;
}

const averages = rows => Object.fromEntries(fields.map(key => [key, average(rows, key)]));
const instanceKey = row => JSON.stringify([row.group, row.base_id]);

function summarize(rows, modelNames) {
  return [...groupBy(rows, row => row.method)].map(([method, scores]) => {
    const instances = [...groupBy(scores, instanceKey).values()].map(repeats =>
      ({ ...averages(repeats), base_id: repeats[0].base_id }));
    // Match radar_table: average repeats, then groups sharing a base, then bases.
    const bases = [...groupBy(instances, row => row.base_id).values()].map(averages);
    const values = Object.fromEntries(fields.map(key =>
      [key, average(Object.hasOwn(metrics, key) ? bases : instances, key)]));
    const isModel = scores[0].source === "model";
    const [model, ...variant] = method.split(" · ");
    const label = isModel ? [modelNames.get(model) || model, ...variant].join(" · ") : method;
    return { method, label, is_model: isModel, values };
  });
}

function fillSelect(control, options, preferred = control.value) {
  control.replaceChildren(...options.map(([value, label]) => new Option(label, value)));
  if (options.some(([value]) => value === preferred)) control.value = preferred;
}

let runs = [];
let activeRun;
let activeData;
let visible = [];
let loadId = 0;
let revision;
let syncing = false;

function updateFilters() {
  const rows = activeData.rows;
  for (const input of kindControls) input.disabled = !rows.some(row => row.kind === input.value);
  let kind = kindControls.find(input => input.checked && !input.disabled);
  if (!kind) {
    kind = kindControls.find(input => !input.disabled);
    kind.checked = true;
  }
  const domain = rows.filter(row => row.kind === kind.value);
  const sizes = [...new Set(domain.map(row => row.setting))].sort(compareNames);
  fillSelect(size, sizes.map(value => [value, value]));
  const bySize = domain.filter(row => row.setting === size.value);
  const groups = [...new Set(bySize.map(row => row.group))];
  fillSelect(group, [["", "Все типы входов"], ...groups.map(key => [key, groupLabels[key] || key])]);
  get("size-field").hidden = sizes.length === 1;
  get("group-field").hidden = groups.length === 1;
  return bySize.filter(row => !group.value || row.group === group.value);
}

function renderInteractive() {
  if (!activeData?.rows) return;
  const { files } = activeRun;
  const rows = updateFilters();
  const kind = rows[0].kind;
  const count = new Set(rows.map(instanceKey)).size;
  const answers = rows.filter(row => row.source === "model").length;
  get("selection").textContent = "Матриц на графиках — " + number.format(count) + " · " + size.value
    + (answers ? " · Ответов модели — " + number.format(answers) : "");
  const axes = get("extended").checked ? extendedAxes : paperAxes;
  const entries = summarize(rows, activeData.modelNames);
  const hasReference = entries.some(entry => entry.method === "EF1-opt (exact)");
  get("reference").parentElement.hidden = !hasReference;
  visible = entries.filter(entry => get("reference").checked || entry.method !== "EF1-opt (exact)" || entries.length === 1);
  observer.disconnect();
  radars.replaceChildren(...visible.map(entry => radar(entry, axes)));
  radars.querySelectorAll("canvas").forEach(canvas => observer.observe(canvas));
  get("bars").replaceChildren(...bars(visible));
  get("table").replaceChildren(...table(visible));
  get("total-title").textContent = kind === "goods" ? "Польза" : "Стоимость";
  const scheme = get("extended").checked ? "extended" : "paper";
  const pdf = scheme + "_radar_" + (kind === "goods" ? 1 : 2) + ".pdf";
  const fullView = !group.value && new Set(activeData.rows.map(row => row.setting)).size === 1;
  const downloads = [
    [fullView ? files[pdf] : null, "График · PDF"],
    [files["results.csv"], "Результаты · CSV"],
    [files["summary.csv"], "Группы · CSV"],
    [files["radar_summary.csv"], "Оси радара · CSV"],
    [files["bootstrap.csv"], "Интервалы · CSV"],
    ...reportDownloads(files),
  ];
  get("downloads").replaceChildren(...downloads.filter(([url]) => url).map(([url, label]) => downloadLink(url, label)));
}

function reportDownloads(files) {
  return [
    [files["scaling_summary.csv"], "Масштабирование · CSV"],
    [files["no_efx_chores.csv"], "Без EFX · CSV"],
  ].filter(([url]) => url);
}

function figures(files, variants) {
  const grid = element("div", undefined, "scaling-charts");
  for (const [variant, kind, label] of variants) {
    const url = files["scaling_" + variant + "_" + kind + ".png"];
    if (!url) continue;
    const link = element("a");
    link.href = url;
    const image = element("img");
    image.src = url;
    image.alt = kinds[kind] + " · " + label;
    image.loading = "lazy";
    link.append(image);
    grid.append(link);
  }
  return grid;
}

function renderReports({ scaling, noEfx }, files, hasResults) {
  const report = get("report");
  report.hidden = !scaling?.length && !noEfx?.length;
  report.replaceChildren();
  if (scaling?.length) {
    const sizes = groupBy(scaling, row => row.setting);
    const count = [...sizes.values()].reduce((sum, rows) =>
      sum + Math.max(...rows.map(row => numeric(row.n_expected) ?? 0)), 0);
    report.append(element("h2", "Масштабирование по n и m"));
    const stats = (count ? number.format(count) + " матриц · " : "") + sizes.size + " размеров";
    report.append(element("p", stats, "chart-note"));
    report.append(figures(files, Object.keys(kinds).map(kind => ["paper", kind, "Зависимость от n и m"])));
    for (const [kind, label] of Object.entries(kinds)) {
      const grid = figures(files, [["n", kind, "Зависимость от n"], ["m", kind, "Зависимость от m"]]);
      if (!grid.childElementCount) continue;
      const details = element("details", undefined, "report-details");
      details.append(element("summary", label + " · все размеры"), grid);
      report.append(details);
    }
  }
  if (noEfx?.length) {
    report.append(element("h2", "Обязанности без EFX"));
    report.append(element("p", "Матриц — " + new Set(noEfx.map(row => row.setting)).size, "chart-note"));
    const scroll = element("div", undefined, "table-scroll");
    scroll.tabIndex = 0;
    scroll.setAttribute("aria-label", "Результаты на матрицах без EFX");
    const data = element("table");
    const head = data.createTHead().insertRow();
    for (const label of ["Размер", "Метод", "EF1", "SD-EF1", "EFX+", "Приближение к EFX+", "SW"]) {
      const cell = element("th", label);
      cell.scope = "col";
      head.append(cell);
    }
    const body = data.createTBody();
    for (const row of noEfx) {
      body.insertRow().append(element("td", row.setting), element("td", row.method),
        ...["ef1", "sd_ef1", "efx", "efx_alpha", "sw_ratio"].map(key => element("td", rateLabel(numeric(row[key])))));
    }
    scroll.append(data);
    report.append(scroll);
  }
  if (!hasResults) {
    get("downloads").replaceChildren(...reportDownloads(files).map(([url, label]) => downloadLink(url, label)));
  }
}

async function selectRun() {
  const run = runs.find(run => run.id === runSelect.value);
  if (run && activeRun?.id === run.id && activeRun.version === run.version) return;
  const id = ++loadId;
  activeRun = run;
  activeData = null;
  get("filters").hidden = true;
  get("interactive").hidden = true;
  get("report").hidden = true;
  get("downloads").replaceChildren();
  observer.disconnect();
  get("selection").textContent = run ? "Загрузка результатов…" : "Опубликованных прогонов пока нет.";
  if (!run) return;

  const { files } = run;
  const csv = key => files[key] ? fetchCsv(files[key]) : null;
  try {
    const [rows, manifest, scaling, noEfx] = await Promise.all([
      csv("results.csv"),
      files["manifest.json"]
        ? request(files["manifest.json"]).then(response => response.json()).catch(() => null)
        : null,
      csv("scaling_summary.csv"),
      csv("no_efx_chores.csv"),
    ]);
    if (id !== loadId) return;
    const scores = rows?.filter(row => row.setting && row.method && Object.hasOwn(kinds, row.kind));
    if (!scores?.length && !scaling?.length && !noEfx?.length) throw new Error("Empty run");
    const models = Array.isArray(manifest?.config?.models) ? manifest.config.models : [];
    activeData = {
      rows: scores?.length ? scores : null,
      modelNames: new Map(models.map(model => [model.label, model.model || model.label])),
    };
    get("selection").textContent = "";
    get("filters").hidden = !activeData.rows;
    get("interactive").hidden = !activeData.rows;
    renderInteractive();
    renderReports({ scaling, noEfx }, files, Boolean(activeData.rows));
  } catch {
    if (id !== loadId) return;
    activeRun = undefined;
    get("selection").textContent = "Не удалось загрузить результаты прогона.";
  }
}

function showRuns(list) {
  runs = list;
  fillSelect(runSelect, runs.length ? runs.map(run => [run.id, run.label]) : [["", "Нет прогонов"]]);
  runSelect.disabled = !runs.length;
  selectRun();
}

async function syncRuns() {
  if (syncing) return;
  syncing = true;
  refresh.disabled = true;
  syncStatus.hidden = true;
  try {
    const options = { cache: "no-cache", signal: AbortSignal.timeout(15000) };
    const branch = await (await request(api + "/branches/main", options)).json();
    const { sha, commit } = branch.commit;
    if (revision !== sha) {
      const tree = await (await request(api + "/git/trees/" + commit.tree.sha + "?recursive=1", options)).json();
      if (tree.truncated) throw new Error("Incomplete run list");
      revision = sha;
      showRuns(catalog(tree.tree.filter(entry => entry.type === "blob"),
        "https://raw.githubusercontent.com/" + repository + "/" + sha + "/"));
    } else if (!activeRun) {
      selectRun();
    }
  } catch {
    syncStatus.textContent = "Не удалось обновить список с GitHub.";
    syncStatus.hidden = false;
  } finally {
    syncing = false;
    refresh.disabled = false;
  }
}

for (const control of [group, size, get("extended"), get("reference"), ...kindControls]) {
  control.addEventListener("change", renderInteractive);
}
rate.addEventListener("change", () => get("bars").replaceChildren(...bars(visible)));
runSelect.addEventListener("change", selectRun);
refresh.addEventListener("click", syncRuns);

request(catalogueFile)
  .then(response => response.json())
  .then(({ files }) => {
    if (!revision) showRuns(catalog(files.map(path => ({ path }))));
  })
  .catch(() => {
    if (!revision) showRuns([]);
  });
syncRuns();
setInterval(() => {
  if (!document.hidden) syncRuns();
}, 120000);
