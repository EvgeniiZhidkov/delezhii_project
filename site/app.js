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
  card.append(element("h3", entry.label), canvas, tip);
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
  link.download = "";
  return link;
}

function start(data) {
  const isExtended = get("extended");
  const showReference = get("reference");
  const countAnswers = entries => entries.reduce((total, entry) => total + (entry.is_model ? entry.instances : 0), 0);
  const all = data.groups.all;
  get("experiment-size").textContent = data.setting.replace("×", " × ");
  get("answer-count").textContent = countAnswers([...all.goods, ...all.chores]);
  get("prompt-count").textContent = data.prompt_count;
  group.replaceChildren(...Object.entries(data.groups).map(([key, info]) => new Option(info.label, key)));
  let visible = [];
  function render() {
    const kind = document.querySelector('input[name="kind"]:checked').value;
    const entries = data.groups[group.value][kind];
    const axes = isExtended.checked ? extendedAxes : paperAxes;
    visible = entries.filter(entry => showReference.checked || entry.method !== "EF1-opt (exact)");
    get("selection").textContent = data.setting + " · Матриц на метод: " + entries[0].instances
      + " · Ответов модели: " + countAnswers(entries);
    observer.disconnect();
    radars.replaceChildren(...visible.map(entry => radar(entry, axes)));
    radars.querySelectorAll("canvas").forEach(canvas => observer.observe(canvas));
    get("bars").replaceChildren(...bars(visible));
    get("table").replaceChildren(...table(visible));
    get("total-title").textContent = kind === "goods" ? "Польза" : "Стоимость";
    get("total-note").textContent = kind === "goods"
      ? "Средняя общая полезность в баллах. Больше — лучше."
      : "Средние общие затраты в баллах. Меньше — лучше.";
    const scheme = isExtended.checked ? "extended" : "paper";
    get("downloads").replaceChildren(
      downloadLink("data/" + scheme + "-" + kind + ".pdf", "График всех входов · PDF"),
      downloadLink("data/results.csv", "Результаты по входам · CSV"),
      downloadLink("data/summary.csv", "Метрики по группам · CSV"),
      downloadLink("data/radar_summary.csv", "Оси радара · CSV"),
      downloadLink("data/bootstrap.csv", "Интервалы · CSV"),
    );
  }
  for (const control of [group, isExtended, showReference, ...document.querySelectorAll('input[name="kind"]')]) {
    control.addEventListener("change", render);
  }
  rate.addEventListener("change", () => get("bars").replaceChildren(...bars(visible)));
  render();
}

fetch("data.json")
  .then(response => {
    if (!response.ok) throw new Error("HTTP " + response.status);
    return response.json();
  })
  .then(start)
  .catch(() => {
    get("selection").textContent = "Данные не загрузились. Обновите страницу или откройте готовые результаты в GitHub.";
  });
