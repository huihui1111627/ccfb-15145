package web

import "net/http"

// indexHTML is the single-page operator console. It is deliberately
// dependency-free: vanilla JS polls the JSON API and renders the tables.
const indexHTML = `<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>CRV · 多版本自定义资源控制台</title>
<style>
  :root { --bg:#0f1420; --panel:#171e2e; --line:#28324a; --fg:#dce3f0;
          --muted:#8593ad; --accent:#5aa2ff; --good:#43c08a; --bad:#f06c75;
          --warn:#e8b04b; --stash:#a78bfa; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 -apple-system,"PingFang SC","Microsoft YaHei",monospace; }
  header { padding:16px 24px; border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:16px; background:var(--panel); }
  header h1 { font-size:17px; margin:0; }
  header .sub { color:var(--muted); font-size:12px; }
  main { padding:20px 24px; display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
          padding:16px; }
  .card.full { grid-column:1 / -1; }
  .card h2 { font-size:13px; margin:0 0 12px; color:var(--muted);
             text-transform:uppercase; letter-spacing:.08em; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; }
  tr.clickable { cursor:pointer; }
  tr.clickable:hover { background:#1d2740; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px;
          background:#22304d; font-size:12px; margin-right:4px; }
  .pill.storage { background:#143d2c; color:var(--good); border:1px solid #236347; }
  .pill.deprecated { color:var(--warn); border:1px solid #5c4a22; }
  .bar { height:8px; background:#22304d; border-radius:6px; overflow:hidden; min-width:120px; }
  .bar > i { display:block; height:100%; background:var(--accent); }
  .bar.done > i { background:var(--good); }
  button { background:#22304d; color:var(--fg); border:1px solid var(--line);
           border-radius:7px; padding:6px 12px; cursor:pointer; font-size:12px; }
  button:hover { border-color:var(--accent); }
  button.danger { border-color:#5c2b30; }
  button.danger:hover { border-color:var(--bad); }
  pre { background:#0c111d; border:1px solid var(--line); border-radius:8px;
        padding:12px; overflow:auto; max-height:340px; font-size:12px; }
  .tabs { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px; }
  .tabs button.active { background:var(--accent); color:#08111f; border-color:var(--accent); }
  .muted { color:var(--muted); }
  .good { color:var(--good); } .bad { color:var(--bad); } .warn { color:var(--warn); }
  .stash { color:var(--stash); }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  select,input { background:#0c111d; color:var(--fg); border:1px solid var(--line);
                 border-radius:7px; padding:6px 9px; font-size:12px; }
  .toast { position:fixed; right:20px; bottom:20px; background:var(--panel);
           border:1px solid var(--line); border-radius:8px; padding:10px 14px;
           opacity:0; transition:opacity .2s; max-width:380px; }
  .toast.show { opacity:1; }
  .kv { display:grid; grid-template-columns:160px 1fr; gap:4px 12px; font-size:12px; }
  .kv .muted { color:var(--muted); }
  code { color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>CRV 多版本自定义资源控制台</h1>
  <span class="sub">存储版本落盘 · 链式转换 · 注解无损藏字段 · 后台迁移</span>
  <span style="flex:1"></span>
  <span id="metrics" class="sub"></span>
</header>

<main>
  <section class="card full">
    <h2>资源定义与存储版本（点击行查看对象的各版本形态）</h2>
    <table id="kinds"><thead><tr>
      <th>资源</th><th>Group</th><th>版本</th><th>存储版本</th>
      <th>对象分布</th><th>存量</th><th>相邻版本演化</th>
    </tr></thead><tbody></tbody></table>
  </section>

  <section class="card">
    <h2>存储版本迁移</h2>
    <div class="row" style="margin-bottom:12px">
      <select id="migKind"></select>
      <select id="migVersion"></select>
      <button id="migBtn">切换存储版本并后台迁移</button>
    </div>
    <table id="migrations"><thead><tr>
      <th>资源</th><th>路径</th><th>进度</th><th>状态</th><th>重试</th>
    </tr></thead><tbody></tbody></table>
  </section>

  <section class="card">
    <h2>转换服务故障注入</h2>
    <div class="row" style="margin-bottom:10px">
      <select id="faultKind"></select>
      <button id="faultDown" class="danger">置为故障 (Down)</button>
      <button id="faultSlow">注入延迟 200ms</button>
      <button id="faultClear">清除全部故障</button>
    </div>
    <pre id="faultState" class="muted">无故障</pre>
    <p class="muted" style="font-size:12px">
      故障期间该资源的读写会<b class="bad">明确失败</b>，不会写入半转换数据；
      新字段藏在 <code>crv.io/stash.&lt;version&gt;</code> 注解中，老客户端读写不丢字段。
    </p>
  </section>

  <section class="card full">
    <h2 id="objTitle">对象多版本视图</h2>
    <div class="row" style="margin-bottom:10px">
      <select id="objKind"></select>
      <select id="objName"></select>
      <div class="tabs" id="versionTabs"></div>
    </div>
    <div class="kv" id="objMeta" style="margin-bottom:10px"></div>
    <pre id="objView" class="muted">选择一个资源和对象…</pre>
  </section>
</main>
<div class="toast" id="toast"></div>

<script>
const $ = s => document.querySelector(s);
const toast = msg => { const t = $('#toast'); t.textContent = msg; t.classList.add('show');
                        setTimeout(() => t.classList.remove('show'), 2600); };
const getJSON = async u => (await fetch(u)).json();
const postJSON = async (u, body) => (await fetch(u, {method:'POST',
  headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)})).json();

let kinds = [], selectedKind = null, selectedVersion = null;

async function loadKinds() {
  kinds = await getJSON('/api/kinds');
  const tb = $('#kinds tbody'); tb.innerHTML = '';
  $('#migKind').innerHTML = $('#faultKind').innerHTML = $('#objKind').innerHTML = '';
  kinds.forEach(k => {
    const dist = Object.entries(k.stored_by_version)
      .map(([v,n]) => "<span class=\"pill " + (v===k.storage_version?'storage':'') + "\">" + (v) + ": " + (n) + "</span>").join('');
    const versions = k.versions.map(v =>
      "<span class=\"pill " + (v===k.storage_version?'storage':'') + "\">" + (v) + "</span>").join('');
    const edges = k.edges.map(e =>
      "<div class=\"muted\">" + (e.from) + "→" + (e.to) + ": " + (e.rules.join('；')) + "</div>").join('');
    tb.insertAdjacentHTML('beforeend',
      "<tr class=\"clickable\" data-kind=\"" + (k.kind) + "\">
        <td><b>" + (k.kind) + "</b></td><td class=\"muted\">" + (k.group) + "</td>
        <td>" + (versions) + "</td><td><span class=\"pill storage\">" + (k.storage_version) + "</span></td>
        <td>" + (dist) + "</td><td>" + (k.object_count) + "</td><td style=\"max-width:380px\">" + (edges) + "</td>
      </tr>");
    $('#migKind').insertAdjacentHTML('beforeend', "<option>" + (k.kind) + "</option>");
    $('#faultKind').insertAdjacentHTML('beforeend', "<option>" + (k.kind) + "</option>");
    $('#objKind').insertAdjacentHTML('beforeend', "<option>" + (k.kind) + "</option>");
  });
  tb.querySelectorAll('tr').forEach(tr => tr.onclick = () => selectKind(tr.dataset.kind));
  if (!selectedKind && kinds.length) selectKind(kinds[0].kind);
  refreshMigVersions();
}

function currentKind() { return kinds.find(k => k.kind === selectedKind); }

function refreshMigVersions() {
  const k = currentKind(); if (!k) return;
  $('#migVersion').innerHTML = k.versions
    .map(v => "<option " + (v===k.storage_version?'selected':'') + ">" + (v) + "</option>").join('');
}

async function selectKind(name) {
  selectedKind = name;
  $('#migKind').value = name; $('#faultKind').value = name; $('#objKind').value = name;
  refreshMigVersions();
  const keys = await getJSON('/api/kinds/' + name + '/objects');
  $('#objName').innerHTML = keys.map(k => "<option>" + (k) + "</option>").join('');
  if (keys.length) await loadObject(keys[0]);
}

async function loadObject(key) {
  const k = currentKind();
  const [ns, name] = key.split('/');
  const view = await getJSON('/api/object?kind=' + encodeURIComponent(k.kind)
    + '&namespace=' + encodeURIComponent(ns) + '&name=' + encodeURIComponent(name));
  $('#objTitle').textContent = (k.kind) + " · " + (key);
  $('#objMeta').innerHTML =
    "<span class=\"muted\">实际落盘版本</span><span><span class=\"pill storage\">" + (view.stored_version) + "</span></span>
     <span class=\"muted\">对象 Key</span><span>" + (view.key) + "</span>";
  const tabs = $('#versionTabs'); tabs.innerHTML = '';
  const versions = Object.keys(view.version_views).sort();
  selectedVersion = selectedVersion && versions.includes(selectedVersion)
    ? selectedVersion : versions[versions.length - 1];
  versions.forEach(v => {
    const b = document.createElement('button');
    b.textContent = v; if (v === selectedVersion) b.className = 'active';
    b.onclick = () => { selectedVersion = v; renderObject(view); };
    tabs.appendChild(b);
  });
  window.__view = view;
  renderObject(view);
}

function renderObject(view) {
  document.querySelectorAll('#versionTabs button').forEach(b =>
    b.classList.toggle('active', b.textContent === selectedVersion));
  const raw = view.version_views[selectedVersion];
  // highlight stash annotations in purple
  const txt = JSON.stringify(JSON.parse(raw), null, 2)
    .replace(/("crv\.io\/stash\.[^"]+")/g, '<span class="stash">$1</span>');
  $('#objView').innerHTML = txt;
}

async function loadMigrations() {
  const rows = await getJSON('/api/migrations');
  $('#migrations tbody').innerHTML = rows.map(m => {
    const pct = (m.migrated + m.skipped) / Math.max(1, m.total) * 100;
    const state = m.state === 'completed' ? '<span class="good">completed</span>'
      : m.state === 'failed' ? '<span class="bad">failed</span>'
      : '<span class="warn">running</span>';
    return "<tr><td>" + (m.kind) + "</td><td>" + (m.from_version) + "→" + (m.to_version) + "</td>
      <td><div class=\"bar " + (pct>=100?'done':'') + "\"><i style=\"width:" + (pct) + "%\"></i></div>
        <span class=\"muted\">" + (m.migrated+m.skipped) + "/" + (m.total) + " (" + (pct.toFixed(0)) + "%)</span></td>
      <td>" + (state) + "</td><td>" + (m.failed_attempts) + "</td></tr>";
  }).join('');
}

async function loadFaults() {
  const f = await getJSON('/api/faults');
  $('#faultState').innerHTML = Object.keys(f).length
    ? Object.entries(f).map(([k,v]) =>
        "<span class=\"bad\">" + (k) + "</span>: down=" + (v.Down) + " slow=" + (v.Slow) + " "
        + "latency=" + (v.AddedLatency) + " failRate=" + (v.FailRate)).join('\n')
    : '<span class="good">无故障，所有资源正常转换</span>';
}

async function loadMetrics() {
  const m = await getJSON('/api/metrics');
  $('#metrics').textContent =
    "转换调用 " + (m.total_calls) + " · 失败 " + (m.failed_calls) + " · 资源 " + (kinds.length);
}

$('#migBtn').onclick = async () => {
  const kind = $('#migKind').value, version = $('#migVersion').value;
  const r = await postJSON('/api/migrate', {kind, version});
  if (r.error) toast('迁移失败: ' + r.error); else toast("已切换 " + (kind) + " 存储版本到 " + (version));
  loadKinds();
};
$('#faultDown').onclick = async () => {
  await postJSON('/api/faults', {kind: $('#faultKind').value, fault: {Down: true}});
  toast('已注入 Down 故障'); loadFaults();
};
$('#faultSlow').onclick = async () => {
  await postJSON('/api/faults', {kind: $('#faultKind').value,
    fault: {Slow: true, AddedLatency: 200000000}});
  toast('已注入 200ms 延迟'); loadFaults();
};
$('#faultClear').onclick = async () => {
  await fetch('/api/faults', {method:'DELETE'}); toast('故障已清除'); loadFaults();
};
$('#objKind').onchange = e => selectKind(e.target.value);
$('#objName').onchange = e => loadObject(e.target.value);

(async function init() {
  await loadKinds();
  loadMigrations(); loadFaults(); loadMetrics();
  setInterval(() => { loadKinds(); loadMigrations(); loadFaults(); loadMetrics(); }, 2500);
  setInterval(() => { if (window.__view && $('#objName').value) loadObject($('#objName').value); }, 4000);
})();
</script>
</body>
</html>`

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(indexHTML))
}
