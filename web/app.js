'use strict';
// Static dashboard: renders the recorded snapshot at once, then swaps to live data from the Express API when it answers (cold-start plan, docs/design.md Phase 11).
const CFG = window.DEMO_CONFIG || { API_BASE: '', TIMEOUT_MS: 8000, WAKING_AFTER_MS: 3000 };
const state = { data: null, live: false, scenario: 'primary' };

const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) { if (k === 'class') el.className = v; else if (typeof v === 'function') el[k] = v; else if (v !== undefined && v !== null && v !== false) el.setAttribute(k, v === true ? '' : v); }
  for (const kid of kids.flat(Infinity)) if (kid !== null && kid !== undefined && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return el;
};
const $ = (id) => document.getElementById(id);
const say = (text, live = false) => { const b = $('banner'); b.textContent = text; b.className = live ? 'banner live' : 'banner'; };

async function getJSON(url, ms) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  try {
    const r = await fetch(url, { signal: ctl.signal });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } finally { clearTimeout(timer); }
}

// ---------- tables ----------
function table(rows, { cols, tailCol } = {}) {
  if (!rows || !rows.length) return h('p', { class: 'muted' }, 'No data.');
  const keys = cols || Object.keys(rows[0]);
  const isTail = (k) => tailCol && k === tailCol;
  return h('div', { class: 'scroll' }, h('table', {},
    h('thead', {}, h('tr', {}, keys.map((k) => h('th', {}, k, isTail(k) ? h('span', { class: 'tag' }, 'costly tail') : null)))),
    h('tbody', {}, rows.map((r) => h('tr', {}, keys.map((k) => h('td', { class: isTail(k) ? 'tail' : '' }, r[k])))))));
}

// ---------- chart: average on-hand inventory against fill rate ----------
function chart(curves) {
  const W = 640, H = 300, m = { l: 52, r: 16, t: 14, b: 40 };
  const groups = [['B3a', 'B3a (post-hoc)', '#8a5cc2'], ['Quantile', 'Quantile policy', '#1f5f8b'], ['B2', 'B2 / B2-sqrt', '#c26a1f'], ['Naive', 'Naive', '#7a7f8a']];
  const series = groups.map(([key, label, color]) => ({ label, color, pts: curves.filter((c) => c.policy.startsWith(key)).map((c) => ({ x: c.fill, y: c.inv, s: c.setting })).sort((a, b) => a.x - b.x) })).filter((s) => s.pts.length);
  const xs = series.flatMap((s) => s.pts.map((p) => p.x)), lo = Math.max(0.8, Math.min(...xs)), hi = Math.max(...xs);
  const yMax = Math.max(...series.filter((s) => s.label !== 'Naive').flatMap((s) => s.pts.map((p) => p.y))) * 1.05;
  const X = (v) => m.l + ((v - lo) / (hi - lo)) * (W - m.l - m.r), Y = (v) => H - m.b - (Math.min(v, yMax) / yMax) * (H - m.t - m.b);
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Average on-hand inventory against achieved fill rate for each policy; alpha 0.99 points are the costly tail');
  const add = (tag, attrs, text) => { const e = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text) e.textContent = text; svg.append(e); return e; };
  add('line', { x1: m.l, y1: H - m.b, x2: W - m.r, y2: H - m.b, stroke: 'currentColor', opacity: 0.4 });
  add('line', { x1: m.l, y1: m.t, x2: m.l, y2: H - m.b, stroke: 'currentColor', opacity: 0.4 });
  add('text', { x: W / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 12, fill: 'currentColor' }, 'achieved fill rate');
  add('text', { x: 12, y: H / 2, 'text-anchor': 'middle', 'font-size': 12, fill: 'currentColor', transform: `rotate(-90 12 ${H / 2})` }, 'average on-hand (units per series-day)');
  for (let i = 0; i <= 4; i++) { const v = lo + ((hi - lo) * i) / 4; add('text', { x: X(v), y: H - m.b + 14, 'text-anchor': 'middle', 'font-size': 11, fill: 'currentColor' }, v.toFixed(3)); }
  for (let i = 0; i <= 4; i++) { const v = (yMax * i) / 4; add('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'currentColor' }, v.toFixed(0)); }
  for (const s of series) {
    add('polyline', { points: s.pts.filter((p) => p.y <= yMax).map((p) => `${X(p.x)},${Y(p.y)}`).join(' '), fill: 'none', stroke: s.color, 'stroke-width': 2 });
    for (const p of s.pts.filter((q) => q.y <= yMax)) {
      const tail = /0\.99/.test(p.s);
      const dot = add('circle', { cx: X(p.x), cy: Y(p.y), r: tail ? 5 : 3.5, fill: tail ? 'none' : s.color, stroke: s.color, 'stroke-width': 2 });
      const t = document.createElementNS('http://www.w3.org/2000/svg', 'title'); t.textContent = `${s.label}, ${p.s}: fill ${p.x}, inventory ${p.y}${tail ? ' (costly tail)' : ''}`; dot.append(t);
    }
  }
  const legend = h('p', { class: 'label' }, series.map((s) => h('span', { style: `color:${s.color};margin-right:12px` }, '● ' + s.label)), h('span', {}, 'Open circles: alpha 0.99, the costly tail.'));
  return h('div', {}, svg, legend);
}

// ---------- sections ----------
function renderScenarioNav() {
  const nav = $('scenarios');
  nav.replaceChildren(...state.data.scenarios.map((s) => h('button', { type: 'button', 'aria-pressed': String(s.id === state.scenario), onclick: () => { state.scenario = s.id; renderScenario(); renderScenarioNav(); } }, s.title)));
}

const stat = (big, label) => h('div', { class: 'stat' }, h('div', { class: 'stat-num' }, big), h('div', { class: 'stat-label' }, label));
function statCards(s) {
  const b3a = s.comparison_comparator_anchored.find((r) => r.comparator.startsWith('B3a'));
  const b2 = s.comparison_comparator_anchored.find((r) => r.comparator.startsWith('B2'));
  const cards = [];
  if (b3a) cards.push(stat(b3a['mean over matched'], 'less inventory than the fair benchmark (B3a), at matched fill rate'));
  if (b2) cards.push(stat(b2['mean over matched'], 'less inventory than the pre-specified point policy (B2)'));
  if (s.expected_outcomes) cards.push(stat(`${s.expected_outcomes.filter((e) => e.held).length}/${s.expected_outcomes.length}`, 'predictions held, committed before this run'));
  return cards.length ? h('div', { class: 'stats' }, cards) : null;
}

function renderScenario() {
  const s = state.data.scenarios.find((x) => x.id === state.scenario) || state.data.scenarios[0];
  const L = state.data.labels;
  const box = $('scenario');
  const b3aRows = (rows) => rows.filter((r) => r.comparator.startsWith('B3a')).concat(rows.filter((r) => !r.comparator.startsWith('B3a')));
  const cols = ['comparator', 'alpha 0.8', 'alpha 0.9', 'alpha 0.95', 'alpha 0.99', 'settings matched', 'mean over matched', '95% interval (items)'];
  const parts = [
    h('h2', {}, s.title), h('p', { class: 'muted' }, s.blurb),
    statCards(s),
    h('div', { class: 'card lead' },
      h('h3', {}, 'Against B3a, the fair benchmark: inventory the quantile policy saves at the same fill rate'),
      h('p', { class: 'label' }, state.data.framing),
      h('details', {}, h('summary', {}, 'Full comparison tables, by service level'),
        h('p', {}, h('strong', {}, 'Statistic: comparator-anchored. '), L['comparator-anchored'] + '.'),
        table(b3aRows(s.comparison_comparator_anchored), { cols, tailCol: 'alpha 0.99' }),
        h('p', {}, h('strong', {}, 'Statistic: quantile-anchored (pre-registered). '), L['quantile-anchored'] + '. "n/a" means the comparator never reaches that fill rate (no extrapolation).'),
        table(b3aRows(s.comparison_quantile_anchored), { cols, tailCol: 'alpha 0.99' })),
      h('p', { class: 'label' }, state.data.caveats[0], ' ', state.data.caveats[1])),
    h('div', { class: 'card' }, h('h3', {}, 'Inventory against fill rate'), chart(s.curves)),
    h('details', { class: 'card' }, h('summary', {}, 'Achieved cycle service minus the target alpha'), table(s.service_gap, { cols: Object.keys(s.service_gap[0]), tailCol: 'alpha 0.99' }),
      s.by_period ? h('details', {}, h('summary', {}, 'Reduction by period (comparator-anchored)'), h('h3', {}, 'Holiday peak'), table(s.by_period.peak), h('h3', {}, 'Rest period'), table(s.by_period.rest)) : null),
    h('details', { class: 'card' }, h('summary', {}, 'By velocity segment (comparator-anchored)'), ['high', 'mid', 'low'].map((k) => [h('h3', {}, k), table(s.by_segment[k])])),
    h('details', { class: 'card' }, h('summary', {}, 'Cost'), h('p', {}, state.data.caveats[2]), table(s.cost)),
    s.expected_outcomes ? h('div', { class: 'card' }, h('h3', {}, 'Expected outcomes written before this run'), h('ul', {}, s.expected_outcomes.map((e) => h('li', { class: e.held ? 'held' : 'nothold' }, (e.held ? 'Held: ' : 'Did not hold: ') + e.text)))) : null,
    s.predictions ? h('div', { class: 'card' }, h('h3', {}, 'Predictions committed before the primary run'), h('ul', {}, s.predictions.map((t) => h('li', {}, t.replace(/\*\*/g, ''))))) : null];
  box.replaceChildren(...parts.filter(Boolean));
}

function renderForecast() {
  const f = state.data.forecast_accuracy;
  const keep = ['', 'scaled pinball, mean', 'scaled pinball, median over series', 'WAPE of the median', 'cov_hi 0.8', 'cov_hi 0.9', 'cov_hi 0.95', 'cov_hi 0.99'];
  const num = (v) => (/^-?\d+\.\d+$/.test(v) ? Number(v).toFixed(4) : v);
  const rows = f.methods.map((r) => Object.fromEntries(Object.entries(r).filter(([k]) => keep.includes(k)).map(([k, v]) => [k === '' ? 'method' : k, num(v)])));
  $('forecast').replaceChildren(
    h('h2', { id: 'h-forecast' }, 'Forecast accuracy on the test window'),
    h('div', { class: 'card' }, h('p', {}, 'Horizon 10, 25 review dates x 300 series, mean over four model versions. B3a first; lower scaled pinball is better; coverage is of the ordered quantity (0.80 / 0.90 / 0.95 / 0.99 nominal).'),
      table(rows), h('p', {}, f.against_b3a),
      h('p', { class: 'label' }, 'The point forecast alone is roughly on par with a 28-day average (WAPE below); the value is in the calibrated quantile forecast and the replay.'), table(f.point.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k === '' ? 'forecast' : k, v])))),
      h('p', { class: 'label' }, 'Source: ', f.source)));
}

function renderCaveats() {
  const nt = state.data.null_tests.test.map((r) => ({ demand: r.demand, 'cycle service at 0.90': r['cs90 mean'], 'reduction vs B3a (comparator-anchored)': r['red_b3a mean'] }));
  $('caveats').replaceChildren(
    h('h2', { id: 'h-caveats' }, 'How to read these numbers'),
    h('div', { class: 'card' }, h('ul', {}, state.data.caveats.map((c) => h('li', {}, c))),
      h('h3', {}, 'Shuffled-demand null test (test window)'), table(nt),
      h('p', { class: 'label' }, 'With realised demand shuffled, the reduction is still positive: the inventory comparison mixes forecast skill with the different shapes of the two rules. Forecast skill is shown by the pinball and coverage results above.'),
      h('p', { class: 'label' }, 'Risk labels are bands (HIGH below the median of protection-interval demand, MEDIUM between the median and P90, LOW at or above P90, an overstock flag above P99), not probabilities of stockout. A HIGH band with a median below one unit, or in a long zero run, is much less likely to run short than it suggests.')));
}

function renderWhatIf() {
  const live = state.live;
  const seriesList = h('datalist', { id: 'series-list' }, (state.seriesList || []).map((s) => h('option', { value: s })));
  const f = h('form', { id: 'wf' },
    h('label', {}, 'Series id', h('input', { name: 'series', list: 'series-list', placeholder: 'FOODS_3_090_CA_3_evaluation', disabled: !live, required: true }), seriesList),
    h('label', {}, 'Review date (Sunday)', h('input', { name: 'date', type: 'date', min: '2015-11-22', max: '2016-05-08', disabled: !live, required: true })),
    h('label', {}, 'Horizon (days)', h('select', { name: 'horizon', disabled: !live }, h('option', { value: '10' }, '10'), h('option', { value: '14' }, '14'))),
    h('label', {}, 'Service level', h('select', { name: 'alpha', disabled: !live }, ['0.8', '0.9', '0.95', '0.99'].map((a) => h('option', { value: a }, a === '0.99' ? '0.99 (costly tail)' : a)))),
    h('label', {}, 'Inventory position (units)', h('input', { name: 'position', type: 'number', min: '0', step: '1', value: '0', disabled: !live })),
    h('button', { class: 'go', type: 'submit', disabled: !live }, 'Compute order'));
  const out = h('div', { id: 'wf-out', 'aria-live': 'polite' });
  f.addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = new URLSearchParams(new FormData(f));
    out.replaceChildren(h('p', { class: 'muted' }, 'Looking up the stored quantile…'));
    try {
      const r = await getJSON(`${CFG.API_BASE}/api/whatif?${q}`, CFG.TIMEOUT_MS);
      const liveBtn = h('button', { class: 'go', type: 'button', onclick: async () => {
        liveBtn.disabled = true; liveOut.replaceChildren(h('p', { class: 'muted' }, 'Calling the model service live (a cold start can take about a minute)…'));
        try {
          const lq = new URLSearchParams({ series: q.get('series'), date: q.get('date'), horizon: q.get('horizon'), alpha: q.get('alpha'), position: q.get('position') });
          const lr = await getJSON(`${CFG.API_BASE}/api/whatif/live?${lq}`, 65000);
          liveOut.replaceChildren(h('p', { class: 'label' }, `Live model service: order-up-to ${lr.order_up_to} (q = ${Number(lr.quantile).toFixed(2)}), order ${lr.order_quantity} units — should match the stored value above, since it's the same model called fresh.`));
        } catch (err) { liveOut.replaceChildren(h('p', { class: 'muted' }, 'The model service did not answer (no stored feature row for this series/date, or it is waking up). Not every date has a live-servable row — only the ones the deployed model version served.')); }
        finally { liveBtn.disabled = false; }
      } }, 'Verify live');
      const liveOut = h('div', { 'aria-live': 'polite' });
      out.replaceChildren(...[h('p', {}, h('strong', {}, `Order ${r.order_quantity} units.`)),
        h('p', { class: 'label' }, `Target stock level ${r.order_up_to} units at alpha ${r.alpha} (q = ${Number(r.quantile).toFixed(2)}); you hold ${r.position}.`),
        h('p', {}, 'Risk band: ', h('span', { class: `band ${r.risk.band}` }, r.risk.band), r.risk.overstock ? ' (overstock flag)' : '', ' ', h('span', { class: 'label' }, r.risk.note)),
        r.risk.qualifiers.length ? h('p', { class: 'label' }, 'Qualifiers: ' + r.risk.qualifiers.join('; ')) : null, r.tail_note ? h('p', { class: 'tail' }, r.tail_note) : null,
        live ? liveBtn : null, liveOut].filter(Boolean));
    } catch (err) {
      const notFound = /HTTP 404/.test(err.message);
      out.replaceChildren(h('p', { class: 'muted' }, notFound
        ? 'No stored quantile for that series, date and horizon. Pick a Sunday between 2015-11-22 and 2016-05-08, and a series id from the list (start typing to see matches).'
        : 'The service did not answer — it may be waking up (a cold start can take about a minute). Try again.'));
    }
  });
  $('whatif').replaceChildren(h('h2', { id: 'h-whatif' }, 'Try it: get an order recommendation'),
    h('div', { class: 'card' }, live ? h('p', {}, 'Order quantity = max(0, ceil(q) - inventory position), computed from the stored quantile of a precomputed review date. Pick a Sunday between 2015-11-22 and 2016-05-08, and a series id from the list.')
      : h('p', { class: 'muted' }, 'Unavailable in snapshot mode. The what-if reads per-series quantile tables from the database, which are not published until the data-use terms of the M5 data are confirmed; it also needs the API to be awake.'), f, out));
}

function renderAll() { renderScenarioNav(); renderScenario(); renderForecast(); renderWhatIf(); renderCaveats(); }

// ---------- side nav: highlight the section in view ----------
function initPageNav() {
  const links = new Map([...document.querySelectorAll('.page-nav a')].map((a) => [a.getAttribute('href').slice(1), a]));
  const setActive = (id) => links.forEach((a, k) => a.classList.toggle('active', k === id));
  setActive(links.keys().next().value);
  const observer = new IntersectionObserver((entries) => {
    const hit = entries.find((e) => e.isIntersecting);
    if (hit) setActive(hit.target.id);
  }, { rootMargin: '-10% 0px -70% 0px' });
  for (const id of links.keys()) { const el = $(id); if (el) observer.observe(el); }
}

// ---------- boot: snapshot first, live when the API answers ----------
async function boot() {
  try { state.data = await getJSON('snapshot.json', CFG.TIMEOUT_MS); } catch (e) { say('Could not load the recorded results.'); return; }
  renderAll();
  initPageNav();
  if (!CFG.API_BASE) { say('Showing the recorded snapshot of the results (no live service configured).'); return; }
  say('Showing the recorded snapshot while the service wakes up (a cold start can take about a minute)…');
  const waking = setTimeout(() => say('Waking the service… showing the recorded snapshot in the meantime.'), CFG.WAKING_AFTER_MS);
  try {
    const list = await getJSON(`${CFG.API_BASE}/api/scenarios`, 70000);
    const payloads = await Promise.all(list.map((s) => getJSON(`${CFG.API_BASE}/api/scenarios/${s.id}`, CFG.TIMEOUT_MS)));
    if (payloads.length) { state.data = { ...state.data, scenarios: payloads }; state.live = true; renderAll(); say('Live: served from the API (precomputed results).', true); }
  } catch (e) { say('The service did not answer in time; showing the recorded snapshot of the same results.'); } finally { clearTimeout(waking); }
  if (state.live) {
    try { state.seriesList = await getJSON(`${CFG.API_BASE}/api/series`, CFG.TIMEOUT_MS); renderWhatIf(); } catch (e) { /* the series input still works as free text */ }
  }
  try {
    const st = await getJSON(`${CFG.API_BASE}/api/status`, CFG.TIMEOUT_MS);
    $('last-checked').textContent = st.last_checked ? `Last checked: ${new Date(st.last_checked).toUTCString()} (${st.ok ? 'up' : 'a check failed'}).` : 'Last checked: no check recorded yet.';
  } catch (e) { /* the footer keeps its default text */ }
}
boot();
