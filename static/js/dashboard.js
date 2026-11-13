"use strict";

const state = {
  meta: null,
  backtest: null,
  chart: null,
  live: null,
  activeTab: "growth",
  tradePage: 0,
  mode: "backtest",
  debounceTimer: null,
  liveDebounceTimer: null,
  pollTimer: null,
  replayTimer: null,
  replayPlaying: false,
  replayBusy: false,
  backtestRequest: 0,
  chartRequest: 0,
  liveRequest: 0,
  refreshSeconds: 60,
};

const elements = {
  error: document.getElementById("pageError"),
  errorDetail: document.getElementById("errorDetail"),
  confidence: document.getElementById("confidenceControl"),
  confidenceValue: document.getElementById("confidenceValue"),
  slippage: document.getElementById("slippageControl"),
  slippageValue: document.getElementById("slippageValue"),
  iv: document.getElementById("ivControl"),
  ivValue: document.getElementById("ivValue"),
  day: document.getElementById("dayControl"),
  loading: document.getElementById("loadingState"),
  scoreNet: document.getElementById("metricNetPnL"),
  scoreNetNoMl: document.getElementById("metricNetPnLNoML"),
  scoreWin: document.getElementById("metricWinRate"),
  scoreWinNoMl: document.getElementById("metricWinRateNoML"),
  scoreSharpe: document.getElementById("metricSharpe"),
  scoreSharpeNoMl: document.getElementById("metricSharpeNoML"),
  scoreDrawdown: document.getElementById("metricDrawdown"),
  scoreDrawdownNoMl: document.getElementById("metricDrawdownNoML"),
  scoreTrades: document.getElementById("metricTrades"),
  scoreTradesNoMl: document.getElementById("metricTradesNoML"),
  backtestChart: document.getElementById("backtestChart"),
  growthChart: document.getElementById("growthChart"),
  chartHint: document.getElementById("chartHint"),
  tabs: [...document.querySelectorAll(".tab-button")],
  tradeBody: document.getElementById("tradesBody"),
  tradePageLabel: document.getElementById("tradePageLabel"),
  previousTrades: document.getElementById("prevTrades"),
  nextTrades: document.getElementById("nextTrades"),
  downloadTrades: document.getElementById("downloadTrades"),
  modelCards: document.getElementById("modelMetricCards"),
  modelSummary: document.getElementById("modelSummary"),
  confusionBody: document.getElementById("confusionBody"),
  timingSummary: document.getElementById("timingSummary"),
  riskSummary: document.getElementById("riskSummary"),
  safetyBadges: document.getElementById("safetyBadges"),
  stressGrid: document.getElementById("stressGrid"),
  strategySummary: document.getElementById("strategySummary"),
  timeline: document.getElementById("validationTimeline"),
  pricingNote: document.getElementById("pricingNote"),
  glossary: document.getElementById("glossary"),
  backtestSection: document.getElementById("backtestSection"),
  liveSection: document.getElementById("liveSection"),
  modeButtons: [...document.querySelectorAll(".mode")],
  liveConfidence: document.getElementById("liveConfidenceControl"),
  liveConfidenceValue: document.getElementById("liveConfidenceValue"),
  liveIv: document.getElementById("liveIvControl"),
  liveIvValue: document.getElementById("liveIvValue"),
  expiry: document.getElementById("expiryControl"),
  providerName: document.getElementById("providerName"),
  replayButton: document.getElementById("replayButton"),
  liveStatus: document.getElementById("liveStatusLabel"),
  liveDot: document.querySelector(".live-dot"),
  lastBarTime: document.getElementById("lastBarTime"),
  dataAge: document.getElementById("dataAge"),
  fetchTiming: document.getElementById("fetchTiming"),
  scoringTiming: document.getElementById("scoringTiming"),
  liveBadges: document.getElementById("liveBadges"),
  liveSpot: document.getElementById("liveSpot"),
  liveChange: document.getElementById("liveChange"),
  liveVwap: document.getElementById("liveVwap"),
  liveZ: document.getElementById("liveZScore"),
  liveSignal: document.getElementById("liveSignal"),
  liveProbability: document.getElementById("liveProbability"),
  paperPnl: document.getElementById("paperPnl"),
  liveChart: document.getElementById("liveChart"),
  liveChartTime: document.getElementById("liveChartTime"),
  optionChain: document.getElementById("optionChain"),
  ivUsedLabel: document.getElementById("ivUsedLabel"),
  paperPositions: document.getElementById("paperPositions"),
  paperHistory: document.getElementById("paperHistory"),
  marketClosedNote: document.getElementById("marketClosedNote"),
  takeSignal: document.getElementById("takeSignalButton"),
  replayControls: document.getElementById("replayControls"),
  replayPlay: document.getElementById("replayPlay"),
  replayStep: document.getElementById("replayStep"),
  replayNextSignal: document.getElementById("replayNextSignal"),
  replaySpeed: document.getElementById("replaySpeed"),
  replayProgress: document.getElementById("replayProgress"),
  sizingSide: document.getElementById("sizingSide"),
  lotsInput: document.getElementById("lotsInput"),
  lotsDown: document.getElementById("lotsDown"),
  lotsUp: document.getElementById("lotsUp"),
  riskLots: document.getElementById("riskLots"),
  autoExit: document.getElementById("autoExit"),
  enforceRules: document.getElementById("enforceRules"),
  copyBot: document.getElementById("copyBot"),
  buyCe: document.getElementById("buyCe"),
  buyPe: document.getElementById("buyPe"),
  paperMessage: document.getElementById("paperMessage"),
  userAccountStats: document.getElementById("userAccountStats"),
  botAccountStats: document.getElementById("botAccountStats"),
  userPositions: document.getElementById("userPositions"),
  resetPaper: document.getElementById("resetPaper"),
  downloadPaperTrades: document.getElementById("downloadPaperTrades"),
};

function cssToken(name) {
  /** Read one named theme token from the stylesheet. */
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const PLOTLY_THEME = {
  font: { family: "Pixelify Sans, monospace", color: cssToken("--text"), size: 14 },
  numberFont: { family: "JetBrains Mono, monospace", color: cssToken("--text") },
  panel: cssToken("--panel"),
  line: cssToken("--line"),
  text: cssToken("--text"),
  sky: cssToken("--sky"),
  coral: cssToken("--coral"),
  profit: cssToken("--profit"),
  loss: cssToken("--loss"),
  black: cssToken("--black"),
  colors: [
    cssToken("--profit"),
    cssToken("--loss"),
    cssToken("--black"),
    cssToken("--sky"),
    cssToken("--coral"),
  ],
};

function formatNumber(value, digits = 2) {
  /** Format a numeric value with tabular locale formatting. */
  const parsed = Number(value);
  return Number.isFinite(parsed)
    ? parsed.toLocaleString("en-IN", {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits,
    })
    : "—";
}

function formatMoney(value) {
  /** Format rupee amounts with a readable Rs prefix. */
  const parsed = Number(value);
  return Number.isFinite(parsed)
    ? `Rs ${parsed.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`
    : "—";
}

function formatPercent(value, digits = 1) {
  /** Format percent metrics while preserving negative drawdowns. */
  const parsed = Number(value);
  return Number.isFinite(parsed) ? `${parsed.toFixed(digits)}%` : "—";
}

function escapeHtml(value) {
  /** Escape dynamic text before placing it into generated table markup. */
  const escapeMap = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return String(value ?? "").replace(/[&<>"']/g, (character) => escapeMap[character]);
}

function showError(message) {
  /** Display the shared taped-paper error card without clearing existing results. */
  elements.error.classList.remove("is-hidden");
  elements.errorDetail.textContent = message || "";
}

function clearError() {
  /** Hide the error card after a successful request. */
  elements.error.classList.add("is-hidden");
  elements.errorDetail.textContent = "";
}

function requestParameters(includeDate = true) {
  /** Build a query string from the active backtest controls. */
  const params = new URLSearchParams({
    conf: elements.confidence.value,
    slip: elements.slippage.value,
    iv: elements.iv.value,
  });
  if (includeDate) params.set("date", elements.day.value || "all");
  return params;
}

function updateControlLabels() {
  /** Synchronize visible slider readouts in both modes. */
  elements.confidenceValue.textContent = Number(elements.confidence.value).toFixed(2);
  elements.slippageValue.textContent = elements.slippage.value;
  elements.ivValue.textContent = Number(elements.iv.value).toFixed(1);
  elements.liveConfidenceValue.textContent = Number(elements.liveConfidence.value).toFixed(2);
  elements.liveIvValue.textContent = Number(elements.liveIv.value).toFixed(1);
}

function toggleLoading(loading) {
  /** Show a compact loading hint while backtest data is refreshed. */
  elements.loading.textContent = loading ? "loading…" : "";
}

async function fetchJson(url, options) {
  /** Fetch and decode a JSON response, surfacing its API error message. */
  const response = await fetch(url, options);
  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error("The server returned an unreadable response.");
  }
  if (!response.ok || payload.error) {
    throw new Error(payload.error || "The request could not be completed.");
  }
  return payload;
}

async function postJson(url, payload) {
  /** Send a JSON-only paper action and surface the API's explicit rejection. */
  return fetchJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function loadMeta() {
  /** Load metadata-driven defaults, day options, risk rules, and validation details. */
  try {
    const payload = await fetchJson("/api/meta");
    state.meta = payload;
    state.refreshSeconds = Number(payload.live_refresh_seconds);
    elements.confidence.value = payload.defaults.conf;
    elements.slippage.value = payload.defaults.slip;
    elements.iv.value = payload.defaults.iv;
    elements.liveConfidence.value = payload.defaults.conf;
    elements.liveIv.value = payload.defaults.iv;
    elements.expiry.min = payload.today;
    elements.expiry.value = payload.default_expiry;
    const riskPercent = Number(payload.risk_rules.risk_per_trade) * 100;
    elements.riskLots.textContent =
      `${formatPercent(riskPercent, Number.isInteger(riskPercent) ? 0 : 1)} risk`;
    elements.day.replaceChildren(new Option("All days", "all"));
    payload.test_days.forEach((day) => elements.day.add(new Option(day, day)));
    elements.providerName.textContent = payload.provider;
    updateControlLabels();
    renderExplainers();
    await refreshBacktest();
    clearError();
  } catch (error) {
    showError(error.message);
  }
}

function updateScorecards(ai, baseline) {
  /** Render the five required AI versus no-ML scorecards. */
  const entries = [
    [elements.scoreNet, elements.scoreNetNoMl, "Net P&L (Rs)", formatMoney, true],
    [elements.scoreWin, elements.scoreWinNoMl, "Win rate %", (value) => formatPercent(value), false],
    [elements.scoreSharpe, elements.scoreSharpeNoMl, "Sharpe (ann.)", (value) => formatNumber(value), false],
    [elements.scoreDrawdown, elements.scoreDrawdownNoMl, "Max drawdown %", (value) => formatPercent(value, 2), false],
    [elements.scoreTrades, elements.scoreTradesNoMl, "Trades taken", (value) => formatNumber(value, 0), false],
  ];
  entries.forEach(([main, compare, key, formatter, pnl]) => {
    const value = Number(ai[key]);
    main.textContent = formatter(value);
    if (pnl) {
      main.classList.toggle("profit", value > 0);
      main.classList.toggle("loss", value < 0);
    }
    compare.textContent = `No-ML: ${formatter(baseline[key])}`;
  });
}

function layoutFor(extra = {}) {
  /** Build a responsive Plotly layout using the shared theme. */
  return {
    paper_bgcolor: PLOTLY_THEME.panel,
    plot_bgcolor: PLOTLY_THEME.panel,
    font: PLOTLY_THEME.font,
    margin: { t: 12, r: 18, b: 48, l: 68 },
    autosize: true,
    hoverlabel: { font: PLOTLY_THEME.font },
    xaxis: {
      gridcolor: PLOTLY_THEME.line,
      zerolinecolor: PLOTLY_THEME.line,
      color: PLOTLY_THEME.text,
      tickfont: PLOTLY_THEME.numberFont,
      automargin: true,
    },
    yaxis: {
      gridcolor: PLOTLY_THEME.line,
      zerolinecolor: PLOTLY_THEME.line,
      color: PLOTLY_THEME.text,
      tickfont: PLOTLY_THEME.numberFont,
      automargin: true,
    },
    ...extra,
  };
}

function renderCandlestickChart(chart) {
  /** Draw price candles, VWAP, Bollinger bands, and confidence-aware markers. */
  if (!chart || !chart.candles.length || !window.Plotly) return;
  const bars = chart.candles;
  const traces = [
    {
      type: "candlestick",
      name: "Bank NIFTY",
      x: bars.map((bar) => bar.datetime),
      open: bars.map((bar) => bar.open),
      high: bars.map((bar) => bar.high),
      low: bars.map((bar) => bar.low),
      close: bars.map((bar) => bar.close),
      increasing: { line: { color: PLOTLY_THEME.profit }, fillcolor: PLOTLY_THEME.profit },
      decreasing: { line: { color: PLOTLY_THEME.loss }, fillcolor: PLOTLY_THEME.loss },
      whiskerwidth: 0.4,
    },
    {
      type: "scatter",
      mode: "lines",
      name: "VWAP",
      x: bars.map((bar) => bar.datetime),
      y: bars.map((bar) => bar.vwap),
      line: { color: PLOTLY_THEME.sky, width: 2 },
      connectgaps: false,
    },
    {
      type: "scatter",
      mode: "lines",
      name: "Bollinger upper",
      x: bars.map((bar) => bar.datetime),
      y: bars.map((bar) => bar.bb_upper),
      line: { color: PLOTLY_THEME.coral, width: 1 },
      connectgaps: false,
    },
    {
      type: "scatter",
      mode: "lines",
      name: "Bollinger lower",
      x: bars.map((bar) => bar.datetime),
      y: bars.map((bar) => bar.bb_lower),
      line: { color: PLOTLY_THEME.coral, width: 1 },
      connectgaps: false,
    },
  ];
  const markers = chart.markers || [];
  ["BUY_CE", "BUY_PE"].forEach((side) => {
    const points = markers.filter((marker) => marker.decision === side);
    traces.push({
      type: "scatter",
      mode: "markers",
      name: side === "BUY_CE" ? "BUY CE" : "BUY PE",
      x: points.map((point) => point.datetime),
      y: points.map((point) => point.price),
      customdata: points.map((point) => [point.z_score, point.confidence]),
      marker: {
        symbol: side === "BUY_CE" ? "triangle-up" : "triangle-down",
        size: 12,
        color: side === "BUY_CE" ? PLOTLY_THEME.profit : PLOTLY_THEME.loss,
        opacity: points.map((point) => point.approved ? 1 : 0.28),
      },
      hovertemplate: "Time %{x}<br>Price %{y:.2f}<br>z-score %{customdata[0]:.3f}<br>AI %{customdata[1]:.2f}<extra></extra>",
    });
  });
  Plotly.react(
    elements.backtestChart,
    traces,
    layoutFor({
      title: { text: `Bank NIFTY · ${chart.date}`, font: PLOTLY_THEME.font, x: 0.02 },
      xaxis: { type: "date" },
      yaxis: { title: "Index level" },
      legend: { orientation: "h", y: 1.12, x: 0, font: PLOTLY_THEME.font },
    }),
    { responsive: true, displayModeBar: false },
  );
  elements.chartHint.textContent = chart.hint || "";
}

function renderGrowthChart(payload) {
  /** Compare realized AI and no-ML growth with the buy-and-hold reference. */
  if (!payload || !window.Plotly) return;
  const curves = payload.equity_curves || {};
  const config = [
    ["ai_model", "AI model", PLOTLY_THEME.profit, 4, "solid"],
    ["no_ml", "No-ML", PLOTLY_THEME.loss, 2, "solid"],
    ["buy_and_hold", "Buy-and-Hold", PLOTLY_THEME.black, 2, "dash"],
  ];
  const traces = config.map(([key, name, color, width, dash]) => {
    const points = curves[key] || [];
    return {
      type: "scatter",
      mode: "lines",
      name,
      x: points.map((point, index) => point.date || index),
      y: points.map((point) => point.equity),
      line: { color, width, dash },
      connectgaps: true,
    };
  });
  Plotly.react(
    elements.growthChart,
    traces,
    layoutFor({
      xaxis: { title: "Trade or session", type: "date" },
      yaxis: { title: "Equity (Rs)" },
      legend: { orientation: "h", y: 1.12, x: 0, font: PLOTLY_THEME.font },
    }),
    { responsive: true, displayModeBar: false },
  );
}

async function refreshBacktest() {
  /** Refresh results and charts, ignoring responses from older slider settings. */
  const requestId = ++state.backtestRequest;
  const chartId = ++state.chartRequest;
  toggleLoading(true);
  const params = requestParameters();
  const chartParams = new URLSearchParams(params);
  try {
    const [payload, chart] = await Promise.all([
      fetchJson(`/api/backtest?${params}`),
      fetchJson(`/api/chart?${chartParams}`),
    ]);
    if (requestId !== state.backtestRequest || chartId !== state.chartRequest) return;
    state.backtest = payload;
    state.chart = chart;
    updateScorecards(payload.metrics, payload.no_ml_metrics);
    renderCandlestickChart(chart);
    renderGrowthChart(payload);
    renderActiveTab();
    updateDownloadLink();
    clearError();
  } catch (error) {
    if (requestId === state.backtestRequest) showError(error.message);
  } finally {
    if (requestId === state.backtestRequest) toggleLoading(false);
  }
}

function updateDownloadLink() {
  /** Keep CSV download parameters synchronized with current controls. */
  elements.downloadTrades.href = `/api/trades.csv?${requestParameters()}`;
}

function renderTrades() {
  /** Render ten AI trades per page with costs and readable exit reasons. */
  const trades = state.backtest?.trades || [];
  const pageSize = 10;
  const pageCount = Math.max(1, Math.ceil(trades.length / pageSize));
  state.tradePage = Math.min(state.tradePage, pageCount - 1);
  const start = state.tradePage * pageSize;
  const visible = trades.slice(start, start + pageSize);
  elements.tradeBody.innerHTML = visible.length
    ? visible.map((trade) => `
      <tr>
        <td>${escapeHtml(formatMarketTime(trade.entry_time))}</td>
        <td>${escapeHtml(formatMarketTime(trade.exit_time))}</td>
        <td class="${trade.side === "CE" ? "positive" : "negative"}">${escapeHtml(trade.side)}</td>
        <td class="number">${formatNumber(trade.strike, 0)}</td>
        <td class="number">${formatNumber(trade.lots, 0)}</td>
        <td class="number">${formatNumber(trade.buy)}</td>
        <td class="number">${formatNumber(trade.sell)}</td>
        <td>${escapeHtml(trade.exit_reason_label)}</td>
        <td class="number">${formatMoney(trade.costs)}</td>
        <td class="number ${Number(trade.pnl) >= 0 ? "positive" : "negative"}">${formatMoney(trade.pnl)}</td>
      </tr>`).join("")
    : '<tr><td colspan="10">No trades in this selection.</td></tr>';
  elements.tradePageLabel.textContent = trades.length
    ? `Trades ${start + 1}–${Math.min(start + pageSize, trades.length)} of ${trades.length}`
    : "No trades";
  elements.previousTrades.disabled = state.tradePage === 0;
  elements.nextTrades.disabled = state.tradePage >= pageCount - 1;
}

function renderModel() {
  /** Render model metrics, confusion counts, signal approval, and measured timing. */
  const prediction = state.backtest?.prediction_metrics || {};
  const cards = [
    ["Accuracy", formatPercent(prediction["Accuracy rate %"])],
    ["Precision", formatPercent(prediction["Precision %"])],
    ["Recall", formatPercent(prediction["Recall %"])],
    ["F1 score", formatNumber(prediction["F1-score"], 3)],
  ];
  elements.modelCards.innerHTML = cards.map(([label, value]) => `
    <article class="mini-metric"><span>${label}</span><strong>${value}</strong></article>`).join("");
  const signals = state.backtest?.signals || {};
  elements.modelSummary.innerHTML =
    `AI approved <span class="number">${formatNumber(signals.approved || 0, 0)}</span> of <span class="number">${formatNumber(signals.candidates || 0, 0)}</span> candidate signals.`;
  elements.confusionBody.innerHTML = `
    <tr><th>AI approved</th><td>${formatNumber(prediction.TP || 0, 0)} true positives</td><td>${formatNumber(prediction.FP || 0, 0)} false positives</td></tr>
    <tr><th>AI rejected</th><td>${formatNumber(prediction.FN || 0, 0)} false negatives</td><td>${formatNumber(prediction.TN || 0, 0)} true negatives</td></tr>`;
  const timings = state.backtest?.timings || {};
  elements.timingSummary.innerHTML =
    `Backtest: <span class="number">${formatNumber(timings.backtest_ms || 0)}</span> ms · Scoring: <span class="number">${formatNumber(timings.scoring_ms_per_signal || 0, 4)}</span> ms/signal · API: <span class="number">${formatNumber(timings.api_ms || 0)}</span> ms`;
}

function renderRisk() {
  /** Render risk metrics, meta-driven safety rules, and the stress grid. */
  const metrics = state.backtest?.metrics || {};
  const grossPnl = Math.abs(Number(metrics["Gross P&L (Rs)"]));
  const costsPct = grossPnl
    ? Number(metrics["Total costs (Rs)"]) / grossPnl * 100
    : 0;
  elements.riskSummary.innerHTML = `
    <div><span class="label">Max drawdown</span><strong class="metric-number">${formatPercent(metrics["Max drawdown %"], 2)}</strong></div>
    <div><span class="label">Total costs</span><strong class="metric-number">${formatMoney(metrics["Total costs (Rs)"])}</strong></div>
    <div><span class="label">Costs / gross P&amp;L</span><strong class="metric-number">${formatPercent(costsPct)}</strong></div>`;
  const meta = state.meta?.meta;
  if (!meta) return;
  const badges = [
    "One trade at a time",
    `Max <span class="number">${escapeHtml(meta.max_trades_per_day)}</span> trades/day`,
    `Stop at -<span class="number">${formatMoney(meta.daily_loss_cap_rs)}</span>/day`,
    `<span class="number">${formatPercent(Number(meta.risk_per_trade) * 100)}</span> of capital risked per trade`,
    `Exit by <span class="number">${escapeHtml(meta.square_off)}</span>`,
  ];
  elements.safetyBadges.innerHTML = badges.map((badge) =>
    `<span class="safety-badge">${badge}</span>`).join("");
}

async function renderStressGrid() {
  /** Load and show all requested slippage and IV stress scenarios. */
  elements.stressGrid.innerHTML = '<p class="muted">Loading stress results…</p>';
  try {
    const result = await fetchJson(`/api/stress?conf=${encodeURIComponent(elements.confidence.value)}`);
    const slips = [...new Set(result.grid.map((cell) => cell.slippage_ticks))].sort((a, b) => a - b);
    const ivs = [...new Set(result.grid.map((cell) => cell.iv))].sort((a, b) => a - b);
    const findCell = (iv, slip) => result.grid.find((cell) =>
      cell.iv === iv && cell.slippage_ticks === slip);
    elements.stressGrid.innerHTML = `
      <table class="stress-table">
        <thead><tr><th>IV / slippage</th>${slips.map((slip) => `<th>${slip} ticks</th>`).join("")}</tr></thead>
        <tbody>${ivs.map((iv) => `<tr><th>${formatNumber(iv, 1)}%</th>${slips.map((slip) => {
          const cell = findCell(iv, slip);
          return `<td>${formatMoney(cell?.net_pnl)} · Sharpe ${formatNumber(cell?.sharpe)}</td>`;
        }).join("")}</tr>`).join("")}</tbody>
      </table>`;
  } catch (error) {
    elements.stressGrid.textContent = "Stress results are temporarily unavailable.";
  }
}

function previousDate(value) {
  /** Return the calendar date before an ISO date for the train timeline label. */
  const previous = new Date(`${value}T00:00:00`);
  previous.setDate(previous.getDate() - 1);
  return previous.toISOString().slice(0, 10);
}

function renderExplainers() {
  /** Build validation, option-pricing, and glossary details from research metadata. */
  if (!state.meta) return;
  const meta = state.meta.meta;
  const validation = meta.validation;
  const start = new Date(`${validation.train_start}T00:00:00`);
  const testStart = new Date(`${validation.test_start}T00:00:00`);
  const testEnd = new Date(`${validation.test_end}T00:00:00`);
  const warmupDays = Math.max(1, (testStart - start) / 86400000);
  const testDays = Math.max(1, (testEnd - testStart) / 86400000);
  const trainWidth = warmupDays / (warmupDays + testDays) * 100;
  elements.strategySummary.innerHTML = `
    <p>VWAP sets the direction; Bollinger Band z-score pullbacks set the timing.</p>
    <p>The Random Forest approves or rejects each candidate before a paper trade is considered.</p>
    <p><strong>Validation:</strong> ${escapeHtml(validation.method)}</p>`;
  elements.timeline.innerHTML = `
    <div class="timeline-period timeline-train" style="flex-basis:${trainWidth}%">
      <strong>Train-only warm-up</strong><span>${escapeHtml(validation.train_start)} → ${escapeHtml(previousDate(validation.test_start))}</span>
    </div>
    <div class="timeline-period timeline-test" style="flex-basis:${100 - trainWidth}%">
      <strong>Test</strong><span>${escapeHtml(validation.test_start)} → ${escapeHtml(validation.test_end)}</span>
    </div>`;
  elements.pricingNote.textContent = meta.option_pricing.note;
  const terms = [
    ["VWAP", "The session’s average price, weighted by traded volume."],
    ["Bollinger Bands", "A rolling average with upper and lower boundaries that describe recent price variation."],
    ["Sharpe", "Return relative to the variability of daily returns."],
    ["Slippage", "The assumed price difference between a signal and its fill."],
    ["Drawdown", "The largest peak-to-trough decline in realized equity."],
  ];
  elements.glossary.innerHTML = terms.map(([term, explanation]) =>
    `<dt>${term}</dt><dd>${explanation}</dd>`).join("");
  renderRisk();
}

function renderActiveTab() {
  /** Refresh the currently selected explainer panel from current API results. */
  if (state.activeTab === "growth") renderGrowthChart(state.backtest);
  if (state.activeTab === "trades") renderTrades();
  if (state.activeTab === "model") renderModel();
  if (state.activeTab === "risk") renderRisk();
  if (state.activeTab === "how") renderExplainers();
}

function setActiveTab(tabName) {
  /** Activate one explainer tab and hide the others. */
  state.activeTab = tabName;
  elements.tabs.forEach((button) => {
    const active = button.dataset.tab === tabName;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".tab-content").forEach((panel) => {
    panel.classList.toggle("is-hidden", panel.id !== `tab-${tabName}`);
  });
  renderActiveTab();
  if (tabName === "risk") renderStressGrid();
}

function scheduleBacktest() {
  /** Debounce slider updates before requesting recalculations. */
  window.clearTimeout(state.debounceTimer);
  state.debounceTimer = window.setTimeout(() => {
    state.tradePage = 0;
    refreshBacktest();
  }, 150);
}

function formatTime(value) {
  /** Format an ISO market timestamp in the exchange's local timezone. */
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? String(value) : formatMarketTime(parsed);
}

function formatMarketTime(value) {
  /** Format a market timestamp in IST regardless of the browser timezone. */
  const parsed = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return `${parsed.toLocaleString("en-IN", {
    timeZone: "Asia/Kolkata",
    dateStyle: "medium",
    timeStyle: "short",
  })} IST`;
}

function renderLiveChart(snapshot) {
  /** Draw live/replay bars, VWAP, bands, and candidate markers. */
  const bars = snapshot.bars || [];
  if (!bars.length || !window.Plotly) return;
  const barIntervalMs = Number(snapshot.bar_interval_minutes) * 60 * 1000;
  const xRange = bars.length === 1 && Number.isFinite(barIntervalMs)
    ? [
      new Date(Date.parse(bars[0].datetime) - barIntervalMs / 2).toISOString(),
      new Date(Date.parse(bars[0].datetime) + barIntervalMs / 2).toISOString(),
    ]
    : undefined;
  const traces = [
    {
      type: "candlestick",
      name: "Bank NIFTY",
      x: bars.map((bar) => bar.datetime),
      open: bars.map((bar) => bar.open),
      high: bars.map((bar) => bar.high),
      low: bars.map((bar) => bar.low),
      close: bars.map((bar) => bar.close),
      increasing: { line: { color: PLOTLY_THEME.profit }, fillcolor: PLOTLY_THEME.profit },
      decreasing: { line: { color: PLOTLY_THEME.loss }, fillcolor: PLOTLY_THEME.loss },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "VWAP",
      x: bars.map((bar) => bar.datetime),
      y: bars.map((bar) => bar.vwap),
      line: { color: PLOTLY_THEME.sky, width: 2 },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "Bollinger upper",
      x: bars.map((bar) => bar.datetime),
      y: bars.map((bar) => bar.bb_upper),
      line: { color: PLOTLY_THEME.coral, width: 1 },
    },
    {
      type: "scatter",
      mode: "lines",
      name: "Bollinger lower",
      x: bars.map((bar) => bar.datetime),
      y: bars.map((bar) => bar.bb_lower),
      line: { color: PLOTLY_THEME.coral, width: 1 },
    },
  ];
  ["BUY_CE", "BUY_PE"].forEach((decision) => {
    const candidates = bars.filter((bar) => bar.decision === decision);
    traces.push({
      type: "scatter",
      mode: "markers",
      name: decision.replace("_", " "),
      x: candidates.map((bar) => bar.datetime),
      y: candidates.map((bar) => bar.close),
      marker: {
        symbol: decision === "BUY_CE" ? "triangle-up" : "triangle-down",
        size: 11,
        color: decision === "BUY_CE" ? PLOTLY_THEME.profit : PLOTLY_THEME.loss,
      },
      customdata: candidates.map((bar) => [bar.z_score, bar.confidence]),
      hovertemplate: "Time %{x}<br>Price %{y:.2f}<br>z-score %{customdata[0]:.3f}<br>AI %{customdata[1]:.2f}<extra></extra>",
    });
  });
  Plotly.react(
    elements.liveChart,
    traces,
    layoutFor({
      xaxis: { type: "date", ...(xRange ? { range: xRange } : {}) },
      yaxis: { title: "Index level" },
      legend: { orientation: "h", y: 1.12, x: 0, font: PLOTLY_THEME.font },
    }),
    { responsive: true, displayModeBar: false },
  );
}

function renderOptionChain(snapshot) {
  /** Render re-priced CE and PE premium/delta by strike. */
  const rows = snapshot.option_chain || [];
  elements.optionChain.innerHTML = `
    <table class="chain-table">
      <thead><tr><th>Strike</th><th>CE premium</th><th>CE Δ</th><th>PE premium</th><th>PE Δ</th></tr></thead>
      <tbody>${rows.map((row) => `
        <tr class="${row.atm ? "is-atm" : ""}">
          <td class="number">${formatNumber(row.strike, 0)}</td>
          <td class="number">${formatMoney(row.ce.premium)}</td>
          <td class="number">${formatNumber(row.ce.delta, 3)}</td>
          <td class="number">${formatMoney(row.pe.premium)}</td>
          <td class="number">${formatNumber(row.pe.delta, 3)}</td>
        </tr>`).join("")}</tbody>
    </table>`;
}

function renderPaper(snapshot) {
  /** Render the independent user/bot accounts, open positions, and trade history. */
  const settings = snapshot.paper_settings || {};
  elements.autoExit.checked = Boolean(settings.auto_exit);
  elements.enforceRules.checked = Boolean(settings.enforce_rules);
  elements.copyBot.checked = Boolean(settings.copy_bot);
  elements.paperPnl.textContent = formatMoney(snapshot.user_account?.pnl_today);
  elements.paperPnl.classList.toggle("profit", Number(snapshot.user_account?.pnl_today) > 0);
  elements.paperPnl.classList.toggle("loss", Number(snapshot.user_account?.pnl_today) < 0);
  if (snapshot.paper_message) elements.paperMessage.textContent = snapshot.paper_message;
  renderAccountStats(elements.userAccountStats, snapshot.user_account);
  renderAccountStats(elements.botAccountStats, snapshot.bot_account);
  renderUserPositions(snapshot.user_positions || []);
  renderClosedPaperTrades(snapshot.user_closed_trades || []);
  renderReplayControls(snapshot.replay, snapshot.provider);
  elements.downloadPaperTrades.href = "/api/live/trades.csv";
}

function squareOffLabel() {
  /** Convert the metadata close time to a concise local-market label. */
  const value = state.meta?.meta?.square_off;
  if (!value) return "Square-off exit";
  const [hour, minute] = value.split(":").map(Number);
  const period = hour >= 12 ? "PM" : "AM";
  return `${hour % 12 || 12}:${String(minute).padStart(2, "0")} ${period} exit`;
}

function renderAccountStats(target, account = {}) {
  /** Render the same capital and performance fields for each paper account. */
  const pnlTone = Number(account.pnl) >= 0 ? "positive" : "negative";
  const rows = [
    ["Equity", formatMoney(account.equity)],
    ["P&amp;L", `<span class="${pnlTone}">${formatMoney(account.pnl)}</span> · ${formatPercent(account.pnl_pct, 2)}`],
    ["Trades", formatNumber(account.trades, 0)],
    ["Win rate", formatPercent(account.win_rate, 1)],
    ["Total costs", formatMoney(account.total_costs)],
    ["Available cash", formatMoney(account.available_cash)],
  ];
  target.innerHTML = rows.map(([label, value]) => `
    <div><dt>${label}</dt><dd>${value}</dd></div>`).join("");
}

function exitReasonLabel(reason) {
  /** Convert simulator exit codes into judge-friendly language. */
  const labels = {
    target: "Target hit",
    stop: "Stop loss hit",
    time: "Max hold time",
    square_off: squareOffLabel(),
    session_end: "Session end",
    manual: "Manual sell",
    bot_exit: "Bot exit",
  };
  return labels[reason] || String(reason || "—");
}

function renderUserPositions(positions) {
  /** Display each open user position with its mark and paper close action. */
  elements.userPositions.innerHTML = positions.length
    ? `<table class="paper-table">
        <thead><tr><th>Side</th><th>Strike</th><th>Lots</th><th>Entry → current</th><th>Unrealised P&amp;L</th><th>Bars held</th><th></th></tr></thead>
        <tbody>${positions.map((position) => `
          <tr>
            <td>${escapeHtml(position.side)}</td>
            <td class="number">${formatNumber(position.strike, 0)}</td>
            <td class="number">${formatNumber(position.lots, 0)}</td>
            <td class="number">${formatMoney(position.buy)} → ${formatMoney(position.mark_premium)}
              <span class="${Number(position.change_pct) >= 0 ? "positive" : "negative"}">(${formatPercent(position.change_pct, 2)})</span></td>
            <td class="${Number(position.unrealized_pnl) >= 0 ? "positive" : "negative"}">${formatMoney(position.unrealized_pnl)}</td>
            <td class="number">${formatNumber(position.bars_held, 0)}</td>
            <td><button class="button button-secondary close-paper-position" data-position-id="${escapeHtml(position.position_id)}" type="button">Sell</button></td>
          </tr>`).join("")}</tbody>
      </table>`
    : '<p class="muted">No open user paper positions.</p>';
  elements.userPositions.querySelectorAll(".close-paper-position").forEach((button) => {
    button.addEventListener("click", () => closePaperPosition(button.dataset.positionId));
  });
}

function renderClosedPaperTrades(trades) {
  /** Render user trades with plain-English exits and download support. */
  elements.paperHistory.innerHTML = trades.length
    ? `<table class="paper-table">
        <thead><tr><th>Side</th><th>Strike</th><th>Lots</th><th>Entry time</th><th>Exit time</th><th>Buy</th><th>Sell</th><th>Exit reason</th><th>Costs</th><th>P&amp;L</th></tr></thead>
        <tbody>${trades.map((trade) => `
          <tr>
            <td>${escapeHtml(trade.side)}</td>
            <td class="number">${formatNumber(trade.strike, 0)}</td>
            <td class="number">${formatNumber(trade.lots, 0)}</td>
            <td>${escapeHtml(formatMarketTime(trade.opened_at))}</td>
            <td>${escapeHtml(formatMarketTime(trade.closed_at))}</td>
            <td class="number">${formatNumber(trade.buy)}</td>
            <td class="number">${formatNumber(trade.sell)}</td>
            <td>${escapeHtml(exitReasonLabel(trade.exit_reason))}</td>
            <td class="number">${formatMoney(trade.costs)}</td>
            <td class="${Number(trade.pnl) >= 0 ? "positive" : "negative"}">${formatMoney(trade.pnl)}</td>
          </tr>`).join("")}</tbody>
      </table>`
    : '<p class="muted">No closed user paper trades yet.</p>';
}

function renderReplayControls(replay, provider) {
  /** Synchronize replay controls and progress with server-side playback state. */
  const isReplay = provider === "replay";
  elements.replayControls.classList.toggle("is-hidden", !isReplay);
  if (!isReplay || !replay) return;
  state.replayPlaying = Boolean(replay.playing);
  elements.replayPlay.textContent = state.replayPlaying ? "Pause" : "Play";
  elements.replaySpeed.value = String(replay.speed);
  elements.replayProgress.textContent =
    `${formatNumber(Number(replay.cursor) + 1, 0)} / ${formatNumber(replay.total_bars, 0)} · ${formatMarketTime(replay.current_time)}`;
}

function renderBadges(snapshot) {
  /** Show delay, replay, missing-volume, market, and feed status. */
  elements.liveBadges.innerHTML = (snapshot.badges || []).map((badge) =>
    `<span class="badge ${badge.kind === "error" ? "error-badge" : ""}">${escapeHtml(badge.label)}</span>`).join("");
}

function paperOrderControls() {
  /** Include the live pricing inputs so paper fills use the displayed quote. */
  return {
    confidence: Number(elements.liveConfidence.value),
    iv: Number(elements.liveIv.value) / 100,
    expiry: elements.expiry.value,
  };
}

async function placePaperOrder(side) {
  /** Submit a manual paper buy and surface cash or risk-rule rejections. */
  const lots = Number(elements.lotsInput.value);
  if (!Number.isInteger(lots) || lots < 1) {
    elements.paperMessage.textContent = "Enter a whole number of lots greater than zero.";
    return;
  }
  try {
    const result = await postJson("/api/live/order", {
      ...paperOrderControls(),
      side,
      lots,
    });
    elements.paperMessage.textContent =
      `Paper ${side} bought · ${formatNumber(result.position.lots, 0)} lots at ${formatMoney(result.position.buy)}.`;
    await refreshLive();
  } catch (error) {
    elements.paperMessage.textContent = error.message;
  }
}

async function closePaperPosition(positionId) {
  /** Close one user position using its latest repriced sell premium. */
  try {
    const result = await postJson("/api/live/close", {
      ...paperOrderControls(),
      position_id: positionId,
    });
    elements.paperMessage.textContent =
      `Paper position sold · ${formatMoney(result.trade.pnl)} net P&L.`;
    await refreshLive();
  } catch (error) {
    elements.paperMessage.textContent = error.message;
  }
}

async function savePaperSettings() {
  /** Persist safety and copy-bot options in the process-local paper desk. */
  try {
    await postJson("/api/live/settings", {
      auto_exit: elements.autoExit.checked,
      enforce_rules: elements.enforceRules.checked,
      copy_bot: elements.copyBot.checked,
    });
    await refreshLive();
  } catch (error) {
    elements.paperMessage.textContent = error.message;
  }
}

function clearReplayTimer() {
  /** Clear the one outstanding client-side replay tick. */
  window.clearTimeout(state.replayTimer);
  state.replayTimer = null;
}

function scheduleReplayTick() {
  /** Schedule one replay step at the selected bars-per-second rate. */
  clearReplayTimer();
  if (!state.replayPlaying || state.mode !== "live") return;
  const speed = Number(elements.replaySpeed.value);
  state.replayTimer = window.setTimeout(runReplayTick, 1000 / speed);
}

async function runReplayTick() {
  /** Advance one bar and refresh all account marks without overlapping ticks. */
  if (!state.replayPlaying || state.replayBusy || state.mode !== "live") return;
  state.replayBusy = true;
  try {
    const result = await postJson("/api/live/replay", {
      action: "step",
      speed: Number(elements.replaySpeed.value),
    });
    state.replayPlaying = Boolean(result.replay.playing);
    await refreshLive();
  } catch (error) {
    state.replayPlaying = false;
    clearReplayTimer();
    showError(error.message);
  } finally {
    state.replayBusy = false;
  }
  scheduleReplayTick();
}

async function sendReplayAction(action) {
  /** Send a single replay action and synchronize the dashboard afterwards. */
  try {
    const result = await postJson("/api/live/replay", {
      action,
      speed: Number(elements.replaySpeed.value),
    });
    state.replayPlaying = Boolean(result.replay.playing);
    if (action === "next_signal" && !result.replay.found_signal) {
      elements.paperMessage.textContent = "There are no more candidate signals in this replay.";
    }
    if (state.replayPlaying) scheduleReplayTick();
    else clearReplayTimer();
    await refreshLive();
  } catch (error) {
    state.replayPlaying = false;
    clearReplayTimer();
    showError(error.message);
  }
}

async function resetPaperAccounts() {
  /** Reset user and bot balances and restart the replay demonstration. */
  try {
    await postJson("/api/live/reset", {});
    state.replayPlaying = false;
    clearReplayTimer();
    elements.paperMessage.textContent = "Both paper accounts were reset.";
    await refreshLive();
  } catch (error) {
    elements.paperMessage.textContent = error.message;
  }
}

function renderLive(snapshot) {
  /** Update live status cards, bars, option chain, and paper state in place. */
  state.live = snapshot;
  elements.providerName.textContent = snapshot.provider;
  elements.liveStatus.textContent = snapshot.error ? "Feed unavailable, retrying" : "Live snapshot updated";
  elements.lastBarTime.textContent = formatTime(snapshot.last_bar_time);
  elements.dataAge.innerHTML = snapshot.data_age_seconds == null
    ? "Data age —"
    : `Age <span class="number">${formatNumber(snapshot.data_age_seconds, 0)}</span> s`;
  elements.fetchTiming.innerHTML = `Fetch <span class="number">${formatNumber(snapshot.fetch_ms || 0)}</span> ms`;
  elements.scoringTiming.innerHTML = `Scoring <span class="number">${formatNumber(snapshot.scoring_ms || 0, 4)}</span> ms`;
  elements.liveDot.classList.remove("pulse");
  void elements.liveDot.offsetWidth;
  elements.liveDot.classList.add("pulse");
  renderBadges(snapshot);
  elements.liveSpot.textContent = formatNumber(snapshot.spot);
  const changeTone = Number(snapshot.change_today_pct) >= 0 ? "positive" : "negative";
  elements.liveChange.innerHTML =
    `<span class="number ${changeTone}">${formatPercent(snapshot.change_today_pct, 2)}</span> today`;
  elements.liveVwap.textContent = formatNumber(snapshot.vwap);
  elements.liveZ.textContent = formatNumber(snapshot.z_score);
  const signal = snapshot.signal || {};
  elements.liveSignal.textContent = signal.state || "Waiting for pullback";
  elements.liveSignal.classList.toggle("positive", Boolean(signal.approved));
  elements.liveSignal.classList.toggle("negative", Boolean(signal.candidate && !signal.approved));
  const canTakeSignal = ["BUY_CE", "BUY_PE"].includes(signal.candidate);
  elements.takeSignal.classList.toggle("is-hidden", !canTakeSignal);
  elements.takeSignal.dataset.side = canTakeSignal ? signal.candidate.slice(-2) : "";
  elements.liveProbability.textContent = signal.probability == null
    ? "No candidate"
    : `AI ${formatNumber(signal.probability, 2)}${signal.approved ? " ✓" : " · rejected"}`;
  elements.ivUsedLabel.textContent =
    `IV ${formatPercent(Number(snapshot.iv_used) * 100, 1)} · ${snapshot.iv_source}`;
  elements.liveChartTime.textContent = snapshot.market_open
    ? "Market open"
    : "Market closed — showing last session";
  elements.marketClosedNote.classList.toggle("is-hidden", Boolean(snapshot.market_open));
  elements.replayButton.textContent = snapshot.provider === "replay" ? "Switch to Yahoo" : "Switch to replay";
  renderLiveChart(snapshot);
  renderOptionChain(snapshot);
  renderPaper(snapshot);
}

async function refreshLive() {
  /** Fetch the live snapshot and ignore stale responses from earlier controls. */
  if (state.mode !== "live" || document.hidden) return;
  const requestId = ++state.liveRequest;
  const params = new URLSearchParams({
    conf: elements.liveConfidence.value,
    iv: elements.liveIv.value,
    expiry: elements.expiry.value,
  });
  try {
    const payload = await fetchJson(`/api/live/snapshot?${params}`);
    if (requestId !== state.liveRequest) return;
    renderLive(payload.snapshot);
    clearError();
  } catch (error) {
    if (requestId === state.liveRequest) showError(error.message);
  }
}

function scheduleLiveRefresh() {
  /** Poll at half the configured server refresh interval while visible in Live mode. */
  window.clearInterval(state.pollTimer);
  state.pollTimer = null;
  if (state.mode !== "live" || document.hidden) return;
  state.pollTimer = window.setInterval(
    refreshLive,
    Math.max(0.5, state.refreshSeconds / 2) * 1000,
  );
}

function setMode(mode) {
  /** Toggle between backtest and live modules without reloading the document. */
  if (mode !== "live" && state.replayPlaying) {
    postJson("/api/live/replay", {
      action: "pause",
      speed: Number(elements.replaySpeed.value),
    }).catch((error) => showError(error.message));
    state.replayPlaying = false;
    clearReplayTimer();
  }
  state.mode = mode;
  elements.backtestSection.classList.toggle("is-hidden", mode !== "backtest");
  elements.liveSection.classList.toggle("is-hidden", mode !== "live");
  elements.modeButtons.forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  if (mode === "live") {
    refreshLive();
    scheduleLiveRefresh();
  } else {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function switchProvider() {
  /** Switch the shared live feed using the sole write endpoint. */
  const provider = state.live?.provider === "replay" ? "yahoo" : "replay";
  state.replayPlaying = false;
  clearReplayTimer();
  try {
    await fetchJson("/api/live/provider", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider }),
    });
    await refreshLive();
  } catch (error) {
    showError(error.message);
  }
}

function onLiveControlChange() {
  /** Debounce live confidence and IV selection updates. */
  window.clearTimeout(state.liveDebounceTimer);
  state.liveDebounceTimer = window.setTimeout(refreshLive, 150);
}

function bindEvents() {
  /** Register slider, mode, tab, pagination, and visibility handlers. */
  [elements.confidence, elements.slippage, elements.iv].forEach((control) => {
    control.addEventListener("input", updateControlLabels);
    control.addEventListener("input", scheduleBacktest);
  });
  elements.day.addEventListener("change", scheduleBacktest);
  [elements.liveConfidence, elements.liveIv].forEach((control) => {
    control.addEventListener("input", updateControlLabels);
    control.addEventListener("input", onLiveControlChange);
  });
  elements.expiry.addEventListener("change", refreshLive);
  elements.tabs.forEach((button) => {
    button.addEventListener("click", () => setActiveTab(button.dataset.tab));
  });
  elements.modeButtons.forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });
  elements.previousTrades.addEventListener("click", () => {
    state.tradePage = Math.max(0, state.tradePage - 1);
    renderTrades();
  });
  elements.nextTrades.addEventListener("click", () => {
    state.tradePage += 1;
    renderTrades();
  });
  elements.replayButton.addEventListener("click", switchProvider);
  elements.takeSignal.addEventListener("click", () => {
    const side = state.live?.signal?.candidate?.slice(-2);
    if (side === "CE" || side === "PE") placePaperOrder(side);
  });
  elements.buyCe.addEventListener("click", () => placePaperOrder("CE"));
  elements.buyPe.addEventListener("click", () => placePaperOrder("PE"));
  elements.lotsDown.addEventListener("click", () => {
    elements.lotsInput.value = String(Math.max(1, Number(elements.lotsInput.value) - 1));
  });
  elements.lotsUp.addEventListener("click", () => {
    elements.lotsInput.value = String(Math.max(1, Number(elements.lotsInput.value) || 1) + 1);
  });
  elements.riskLots.addEventListener("click", () => {
    const side = elements.sizingSide.value;
    elements.lotsInput.value = String(state.live?.risk_lots?.[side] || 1);
  });
  [elements.autoExit, elements.enforceRules, elements.copyBot].forEach((control) => {
    control.addEventListener("change", savePaperSettings);
  });
  elements.resetPaper.addEventListener("click", resetPaperAccounts);
  elements.replayPlay.addEventListener("click", () => {
    sendReplayAction(state.replayPlaying ? "pause" : "play");
  });
  elements.replayStep.addEventListener("click", async () => {
    if (state.replayPlaying) await sendReplayAction("pause");
    await sendReplayAction("step");
  });
  elements.replayNextSignal.addEventListener("click", () => sendReplayAction("next_signal"));
  elements.replaySpeed.addEventListener("change", () => {
    if (state.replayPlaying) sendReplayAction("play");
  });
  document.addEventListener("keydown", (event) => {
    if (
      state.mode !== "live"
      || event.altKey
      || event.ctrlKey
      || event.metaKey
      || event.target instanceof HTMLElement
        && event.target.matches("input, select, textarea, [contenteditable='true']")
    ) return;
    const key = event.key.toLowerCase();
    if (key === "c" || key === "p") {
      event.preventDefault();
      placePaperOrder(key === "c" ? "CE" : "PE");
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      window.clearInterval(state.pollTimer);
      state.pollTimer = null;
      if (state.replayPlaying) sendReplayAction("pause");
    } else {
      scheduleLiveRefresh();
      refreshLive();
    }
  });
  window.addEventListener("resize", () => {
    if (!window.Plotly) return;
    [elements.backtestChart, elements.growthChart, elements.liveChart]
      .filter((chart) => chart && chart.data)
      .forEach((chart) => Plotly.Plots.resize(chart));
  });
}

bindEvents();
loadMeta();
