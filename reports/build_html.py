"""把 report_data.json / sweep.json 内嵌进自包含 HTML 报告。

运行后生成 report.html，无需服务器、无外部依赖，双击即可在浏览器打开。
页面四块：逐层切分方式、每卡权重/激活大小、每步集合通信量、训练损失曲线。
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    with open(os.path.join(HERE, "report_data.json")) as f:
        data = json.load(f)
    with open(os.path.join(HERE, "sweep.json")) as f:
        sweep = json.load(f)

    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    html = html.replace("__SWEEP__", json.dumps(sweep, ensure_ascii=False))
    out = os.path.join(HERE, "report.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"报告页面已生成 {out}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>张量并行训练框架 · 切分与通信可视化</title>
<style>
  :root {
    --bg:#0f1419; --panel:#1a212b; --panel2:#222c38; --line:#2d3947;
    --text:#e6edf3; --muted:#8b98a5; --accent:#4ea1ff; --accent2:#57d9a3;
    --warn:#ffb454; --ar:#4ea1ff; --ag:#57d9a3; --rs:#c792ea;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:var(--bg); color:var(--text);
    font-family:-apple-system,"PingFang SC","Microsoft YaHei",Segoe UI,sans-serif;
    line-height:1.55;
  }
  header { padding:28px 32px 18px; border-bottom:1px solid var(--line); }
  header h1 { margin:0 0 6px; font-size:22px; }
  header .sub { color:var(--muted); font-size:13px; }
  .cfg-chips { margin-top:12px; display:flex; flex-wrap:wrap; gap:8px; }
  .chip { background:var(--panel2); border:1px solid var(--line);
    border-radius:14px; padding:3px 12px; font-size:12px; color:var(--muted); }
  .chip b { color:var(--text); }
  main { padding:24px 32px 60px; max-width:1280px; margin:0 auto; }
  section { background:var(--panel); border:1px solid var(--line);
    border-radius:12px; padding:20px 22px; margin-bottom:24px; }
  section h2 { margin:0 0 4px; font-size:17px; display:flex; align-items:center; gap:8px; }
  section h2 .num { background:var(--accent); color:#08111c; width:22px; height:22px;
    border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
    font-size:13px; font-weight:700; }
  section .desc { color:var(--muted); font-size:12.5px; margin:0 0 16px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { text-align:left; padding:9px 12px; border-bottom:1px solid var(--line);
    vertical-align:top; }
  th { color:var(--muted); font-weight:600; font-size:12px; }
  td.mono, .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  .tag { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px;
    font-family:ui-monospace,monospace; }
  .tag.ar { background:rgba(78,161,255,.15); color:var(--ar); }
  .tag.ag { background:rgba(87,217,163,.15); color:var(--ag); }
  .tag.rs { background:rgba(199,146,234,.15); color:var(--rs); }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
  .card { background:var(--panel2); border:1px solid var(--line);
    border-radius:10px; padding:14px 16px; }
  .card h3 { margin:0 0 10px; font-size:13.5px; color:var(--muted); font-weight:600; }
  .bar-row { display:flex; align-items:center; gap:10px; margin:8px 0; font-size:12.5px; }
  .bar-row .lbl { width:130px; color:var(--muted); flex-shrink:0; }
  .bar-track { flex:1; background:#0d1218; border-radius:6px; height:18px; overflow:hidden; }
  .bar-fill { height:100%; border-radius:6px; }
  .bar-val { width:110px; text-align:right; font-family:ui-monospace,monospace; font-size:11.5px; }
  .legend { display:flex; gap:16px; margin:6px 0 14px; font-size:12px; color:var(--muted); }
  .legend span { display:inline-flex; align-items:center; gap:6px; }
  .dot { width:10px; height:10px; border-radius:2px; display:inline-block; }
  svg { width:100%; height:auto; display:block; }
  .axis { stroke:var(--muted); stroke-width:1; }
  .gridline { stroke:var(--line); stroke-width:1; stroke-dasharray:3 4; }
  .axlabel { fill:var(--muted); font-size:11px; }
  .loss-line { fill:none; stroke:var(--accent2); stroke-width:2.2; }
  .comm-bar { stroke-width:0; }
  .summary-num { font-family:ui-monospace,monospace; color:var(--accent); }
  .muted { color:var(--muted); }
  .ok { color:var(--accent2); font-weight:600; }
  .scroll { overflow-x:auto; }
</style>
</head>
<body>
<header>
  <h1>玻璃温室花卉育种语言模型 · 张量并行切分与通信可视化</h1>
  <div class="sub">numpy 模拟多卡 + 模拟集合通信（环形通信量记账）｜所有前向损失与参数梯度均与单卡不切分参考实现在 float64 机器精度内一致</div>
  <div class="cfg-chips" id="cfgChips"></div>
</header>
<main>

  <section>
    <h2><span class="num">1</span>每层的切分方式</h2>
    <p class="desc">列切（按输出维/头）与行切（按输入维）配对，配对之间无通信，只在行切出口做一次归约分散；词表按词维切，序列并行沿 token 维切。</p>
    <div class="scroll"><table id="planTable">
      <thead><tr><th>层 / 模块</th><th>切分方式</th><th>前向通信</th><th>反向通信</th></tr></thead>
      <tbody></tbody>
    </table></div>
  </section>

  <section>
    <h2><span class="num">2</span>每张卡上的权重与激活大小</h2>
    <p class="desc">字节 = 元素数 × 8（float64）。权重/梯度按各卡等分口径，激活为前向结束时保留到反向的张量。</p>
    <div class="grid2">
      <div class="card">
        <h3>每卡显存构成（字节）</h3>
        <div class="legend">
          <span><i class="dot" style="background:#4ea1ff"></i>权重</span>
          <span><i class="dot" style="background:#ffb454"></i>梯度</span>
          <span><i class="dot" style="background:#57d9a3"></i>激活</span>
        </div>
        <div id="memBars"></div>
      </div>
      <div class="card">
        <h3>权重参数元素数明细（每卡）</h3>
        <div id="weightBars"></div>
      </div>
    </div>
    <div class="card" style="margin-top:20px">
      <h3>保留到反向的激活张量明细（rank 0）</h3>
      <div class="scroll"><table id="actTable">
        <thead><tr><th>激活张量</th><th>元素数</th><th>字节</th></tr></thead>
        <tbody></tbody>
      </table></div>
    </div>
  </section>

  <section>
    <h2><span class="num">3</span>每步各类集合操作的通信量</h2>
    <p class="desc">环形公式：all-reduce = 2(P−1)/P·n，all-gather / reduce-scatter = (P−1)/P·n_full。下表的公式值与运行时实测记账逐项相等。</p>
    <div class="grid2">
      <div class="card">
        <h3>每卡每步通信量（按操作类型）</h3>
        <div id="commBars"></div>
        <div class="legend" style="margin-top:12px">
          <span><i class="dot" style="background:var(--ar)"></i>all-reduce</span>
          <span><i class="dot" style="background:var(--ag)"></i>all-gather</span>
          <span><i class="dot" style="background:var(--rs)"></i>reduce-scatter</span>
        </div>
      </div>
      <div class="card">
        <h3>每步集合操作次数与通信事件</h3>
        <table id="commCountTable">
          <thead><tr><th>操作</th><th>次数</th><th>每卡字节</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
    <div class="card" style="margin-top:20px">
      <h3>逐事件通信流水（前向 → 反向）</h3>
      <div class="scroll"><table id="eventTable">
        <thead><tr><th>#</th><th>操作</th><th>来源层</th><th>每卡元素</th><th>每卡字节</th></tr></thead>
        <tbody></tbody>
      </table></div>
    </div>
  </section>

  <section>
    <h2><span class="num">4</span>训练损失曲线</h2>
    <p class="desc">固定一批随机 token，SGD 训练 80 步。损失持续下降说明分片前向 + 手写反向的梯度方向正确。</p>
    <svg id="lossSvg" viewBox="0 0 900 320" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="legend" style="margin-top:8px">
      <span>起始损失 <b class="summary-num" id="lossStart"></b></span>
      <span>结束损失 <b class="summary-num" id="lossEnd"></b></span>
      <span>下降 <b class="ok" id="lossDrop"></b></span>
    </div>
  </section>

  <section>
    <h2>附：随机配置扫描（峰值显存 vs 通信量）</h2>
    <p class="desc">24 组随机 (层数/头数/词表/卡数/序列长) 组合，每组都通过单卡参考对账。</p>
    <svg id="sweepSvg" viewBox="0 0 900 340" preserveAspectRatio="xMidYMid meet"></svg>
  </section>

</main>
<script>
const DATA = __DATA__;
const SWEEP = __SWEEP__;
const fmt = b => b >= 1048576 ? (b/1048576).toFixed(2)+" MB"
            : b >= 1024 ? (b/1024).toFixed(2)+" KB" : b.toFixed(0)+" B";
const esc = s => String(s).replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

// 配置 chips
const c = DATA.config;
[["层数",c.layers],["头数",c.heads],["隐藏维",c.hidden],["FF 维",c.ff],
 ["词表",c.vocab],["卡数 P",c.world_size],["序列长",c.seq_len]]
 .forEach(([k,v])=>{
   const el=document.createElement("span"); el.className="chip";
   el.innerHTML=k+" <b>"+v+"</b>"; document.getElementById("cfgChips").appendChild(el);
 });

// 1) 切分方式表
const planTb=document.querySelector("#planTable tbody");
DATA.layer_plan.forEach(r=>{
  const tr=document.createElement("tr");
  tr.innerHTML=`<td>${esc(r.layer)}</td><td>${esc(r.sharding)}</td>
    <td class="muted">${esc(r.comm_fwd)}</td><td class="muted">${esc(r.comm_bwd)}</td>`;
  planTb.appendChild(tr);
});

// 2) 显存
const mem=DATA.memory;
const rank0=mem.per_rank[0];
const memMax=Math.max(...mem.per_rank.map(r=>r.param_bytes+r.grad_bytes+r.activation_bytes));
const memBars=document.getElementById("memBars");
mem.per_rank.forEach(r=>{
  const total=r.param_bytes+r.grad_bytes+r.activation_bytes;
  const w=(r.param_bytes/memMax*100), g=(r.grad_bytes/memMax*100), a=(r.activation_bytes/memMax*100);
  const row=document.createElement("div");
  row.innerHTML=`<div class="bar-row"><div class="lbl">rank ${r.rank}</div>
    <div class="bar-track">
      <div class="bar-fill" style="width:${w}%;background:#4ea1ff;float:left"></div>
      <div class="bar-fill" style="width:${g}%;background:#ffb454;float:left"></div>
      <div class="bar-fill" style="width:${a}%;background:#57d9a3;float:left"></div>
    </div>
    <div class="bar-val">${fmt(total)}</div></div>`;
  memBars.appendChild(row);
});
const wItems=Object.entries(mem.weight_breakdown_elems);
const wMax=Math.max(...wItems.map(([,n])=>n));
const wb=document.getElementById("weightBars");
wItems.forEach(([k,n])=>{
  const row=document.createElement("div");
  row.innerHTML=`<div class="bar-row"><div class="lbl">${esc(k)}</div>
    <div class="bar-track"><div class="bar-fill" style="width:${n/wMax*100}%;background:var(--accent)"></div></div>
    <div class="bar-val">${n} 元素</div></div>`;
  wb.appendChild(row);
});
const actTb=document.querySelector("#actTable tbody");
mem.activation_items.forEach(it=>{
  const tr=document.createElement("tr");
  tr.innerHTML=`<td class="mono">${esc(it.name)}</td><td class="mono">${it.elements}</td>
    <td class="mono">${fmt(it.bytes)}</td>`;
  actTb.appendChild(tr);
});

// 3) 通信
const comm=DATA.communication.formula;
const opColor={all_reduce:"var(--ar)",all_gather:"var(--ag)",reduce_scatter:"var(--rs)"};
const opTag={all_reduce:"ar",all_gather:"ag",reduce_scatter:"rs"};
const commMax=Math.max(...Object.values(comm.bytes_per_rank));
const cb=document.getElementById("commBars");
Object.entries(comm.bytes_per_rank).forEach(([op,b])=>{
  const row=document.createElement("div");
  row.innerHTML=`<div class="bar-row"><div class="lbl"><span class="tag ${opTag[op]}">${op.replace("_","-")}</span></div>
    <div class="bar-track"><div class="bar-fill" style="width:${b/commMax*100}%;background:${opColor[op]}"></div></div>
    <div class="bar-val">${fmt(b)}</div></div>`;
  cb.appendChild(row);
});
const cct=document.querySelector("#commCountTable tbody");
Object.entries(comm.events).forEach(([op,cnt])=>{
  const tr=document.createElement("tr");
  tr.innerHTML=`<td><span class="tag ${opTag[op]}">${op.replace("_","-")}</span></td>
    <td class="mono">${cnt}</td><td class="mono">${fmt(comm.bytes_per_rank[op])}</td>`;
  cct.appendChild(tr);
});
const evTb=document.querySelector("#eventTable tbody");
DATA.communication.events.forEach((ev,i)=>{
  const tr=document.createElement("tr");
  tr.innerHTML=`<td class="mono muted">${i+1}</td>
    <td><span class="tag ${opTag[ev.name]}">${ev.name.replace("_","-")}</span></td>
    <td class="mono">${esc(ev.layer)}</td>
    <td class="mono">${ev.elements_per_rank}</td>
    <td class="mono">${fmt(ev.bytes_per_rank)}</td>`;
  evTb.appendChild(tr);
});

// 4) 损失曲线
(function(){
  const losses=DATA.loss_curve, W=900,H=320,pad=48;
  const svg=document.getElementById("lossSvg");
  const min=Math.min(...losses), max=Math.max(...losses);
  const x=i=>pad+i*(W-2*pad)/(losses.length-1);
  const y=v=>H-pad-(v-min)/(max-min||1)*(H-2*pad);
  let s="";
  for(let i=0;i<=4;i++){
    const gy=pad+i*(H-2*pad)/4, val=max-i*(max-min)/4;
    s+=`<line class="gridline" x1="${pad}" y1="${gy}" x2="${W-pad}" y2="${gy}"/>`;
    s+=`<text class="axlabel" x="${pad-8}" y="${gy+4}" text-anchor="end">${val.toFixed(2)}</text>`;
  }
  s+=`<line class="axis" x1="${pad}" y1="${H-pad}" x2="${W-pad}" y2="${H-pad}"/>`;
  s+=`<line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${H-pad}"/>`;
  s+=`<text class="axlabel" x="${W-pad}" y="${H-pad+22}" text-anchor="end">训练步</text>`;
  s+=`<text class="axlabel" x="${pad-30}" y="${pad-8}">损失</text>`;
  const path=losses.map((v,i)=>(i?"L":"M")+x(i).toFixed(1)+","+y(v).toFixed(1)).join(" ");
  s+=`<path class="loss-line" d="${path}"/>`;
  svg.innerHTML=s;
  document.getElementById("lossStart").textContent=losses[0].toFixed(4);
  document.getElementById("lossEnd").textContent=losses[losses.length-1].toFixed(4);
  document.getElementById("lossDrop").textContent=
    (100*(losses[0]-losses[losses.length-1])/losses[0]).toFixed(1)+"%";
})();

// 附：扫描散点/柱状
(function(){
  const W=900,H=340,pad=52;
  const svg=document.getElementById("sweepSvg");
  const maxMem=Math.max(...SWEEP.map(d=>d.peak_bytes_rank0));
  const maxComm=Math.max(...SWEEP.map(d=>d.comm_bytes_per_rank));
  const bw=(W-2*pad)/SWEEP.length;
  let s="";
  SWEEP.forEach((d,i)=>{
    const x0=pad+i*bw+bw*0.18;
    const hMem=(d.peak_bytes_rank0/maxMem)*(H-2*pad)*0.85;
    const hComm=(d.comm_bytes_per_rank/maxComm)*(H-2*pad)*0.85;
    const base=H-pad;
    s+=`<rect class="comm-bar" x="${x0}" y="${base-hMem}" width="${bw*0.32}" height="${hMem}" fill="#4ea1ff" opacity="0.85"/>`;
    s+=`<rect class="comm-bar" x="${x0+bw*0.36}" y="${base-hComm}" width="${bw*0.32}" height="${hComm}" fill="#57d9a3" opacity="0.85"/>`;
    s+=`<text class="axlabel" x="${x0+bw*0.35}" y="${H-pad+16}" text-anchor="middle" transform="rotate(0)">${d.config.world_size}卡</text>`;
  });
  s+=`<line class="axis" x1="${pad}" y1="${H-pad}" x2="${W-pad}" y2="${H-pad}"/>`;
  s+=`<text class="axlabel" x="${pad}" y="${pad-14}"><tspan fill="#4ea1ff">■</tspan> 峰值显存/卡　<tspan fill="#57d9a3">■</tspan> 每卡每步通信量（24 组配置，各自归一化）</text>`;
  svg.innerHTML=s;
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
