"""Self-contained operator dashboard (vanilla HTML/JS, no build step).

Served at ``GET /``. Polls ``/v1/infer`` for live decisions, renders an
evidence card per person (scores, matched rules, explanation, counterfactual),
and lets a reviewer file an appeal verdict against a logged event.
"""

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Clothic AI - Compliance Dashboard</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; margin: 0; background:#0f1115; color:#e6e6e6; }
  header { padding:14px 20px; background:#171a21; border-bottom:1px solid #262a33;
           display:flex; align-items:center; gap:16px; }
  header h1 { font-size:18px; margin:0; }
  header .sub { color:#8b93a1; font-size:13px; }
  .controls { margin-left:auto; display:flex; gap:8px; align-items:center; }
  select, button { background:#222732; color:#e6e6e6; border:1px solid #333a47;
                   border-radius:6px; padding:6px 10px; font-size:13px; cursor:pointer; }
  button.primary { background:#2d6cdf; border-color:#2d6cdf; }
  main { padding:18px; display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:14px; }
  .card { background:#171a21; border:1px solid #262a33; border-radius:10px; overflow:hidden; }
  .card .bar { height:6px; }
  .card .body { padding:12px 14px; }
  .row { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
  .badge { font-size:11px; font-weight:700; padding:3px 8px; border-radius:20px; letter-spacing:.4px; }
  .scores { display:grid; grid-template-columns:1fr 1fr; gap:4px 12px; font-size:12px; color:#aab2c0; margin:8px 0; }
  .scores b { color:#e6e6e6; }
  .why { font-size:13px; line-height:1.5; color:#cfd6e2; }
  .cover { margin-top:8px; font-size:12px; color:#9fc7ff; }
  .cover .chip { display:inline-block; background:#1d2738; border:1px solid #2c3a52;
                 border-radius:5px; padding:1px 6px; margin:2px 3px 0 0; }
  .fix { margin-top:8px; font-size:12px; color:#7fd6a0; }
  .rules { margin-top:8px; font-size:11px; color:#8b93a1; }
  .appeal { margin-top:10px; display:flex; gap:6px; }
  .appeal select { flex:1; }
  .compliant { background:#1f8f4e; } .minor_violation { background:#d98a17; }
  .major_violation { background:#d63b3b; } .insufficient_evidence { background:#7a7f8a; }
  .stats { padding:0 20px 8px; color:#8b93a1; font-size:12px; }
  .toast { position:fixed; bottom:16px; right:16px; background:#2d6cdf; color:#fff;
           padding:10px 14px; border-radius:8px; opacity:0; transition:opacity .3s; }
</style>
</head>
<body>
<header>
  <h1>Clothic AI</h1>
  <span class="sub">Outfit-compliance · visual attributes + policy reasoning</span>
  <div class="controls">
    <select id="profile"></select>
    <button id="toggle" class="primary">▶ Start live</button>
  </div>
</header>
<div class="stats" id="stats"></div>
<main id="cards"></main>
<div class="toast" id="toast"></div>
<script>
const COLORS = {compliant:'#1f8f4e', minor_violation:'#d98a17',
                major_violation:'#d63b3b', insufficient_evidence:'#7a7f8a'};
let timer = null;

async function loadProfiles(){
  const r = await fetch('/v1/profiles'); const j = await r.json();
  const sel = document.getElementById('profile');
  sel.innerHTML = j.profiles.map(p=>`<option value="${p}">${p}</option>`).join('');
}

function fmt(x){ return x==null ? 'n/a' : Number(x).toFixed(2); }

function card(p){
  const s = p.scores;
  const rules = (p.matched_rules||[]).map(r=>r.id).join(', ') || '-';
  const fix = p.remediation && p.remediation.steps.length
      ? `<div class="fix">✔ Fix: ${p.remediation.steps.join('; ')} `
        + `(${p.remediation.verified?'verified→compliant':'partial'})</div>` : '';
  const cover = (p.coverage && p.coverage.length)
      ? `<div class="cover">Coverage: ` + p.coverage.map(c =>
          `<b>${c.garment_type}</b> ` + Object.entries(c.regions).map(([r,v]) =>
            `<span class="chip">${r} ${Number(v).toFixed(2)}</span>`).join('')
        ).join(' ') + `</div>` : '';
  const eid = p._event_id;
  return `<div class="card">
    <div class="bar" style="background:${COLORS[p.decision]}"></div>
    <div class="body">
      <div class="row">
        <strong>Track #${p.track_id}</strong>
        <span class="badge ${p.decision}">${p.decision.replace('_',' ').toUpperCase()}</span>
      </div>
      <div class="scores">
        <span>violation <b>${fmt(s.overall_violation)}</b></span>
        <span>compliance <b>${fmt(s.compliance_score)}</b></span>
        <span>exposure <b>${fmt(s.exposure_score)}</b></span>
        <span>uncertainty <b>${fmt(s.uncertainty_score)}</b></span>
      </div>
      <div class="why">${p.explanation}</div>
      ${cover}
      ${fix}
      <div class="rules">rules: ${rules} · action: ${p.action}</div>
      ${eid!=null ? `<div class="appeal">
        <select id="v${eid}">
          <option value="confirm">Confirm</option>
          <option value="override_compliant">Override → compliant</option>
          <option value="override_violation">Override → violation</option>
        </select>
        <button onclick="review(${eid})">File review</button></div>` : ''}
    </div></div>`;
}

async function tick(){
  const profile_id = document.getElementById('profile').value;
  const r = await fetch('/v1/infer', {method:'POST', headers:{'Content-Type':'application/json'},
                          body: JSON.stringify({profile_id, camera_id:'dashboard'})});
  const frame = await r.json();
  // Map freshly-logged events back to cards (most recent first, same order).
  const ev = await (await fetch('/v1/events?camera_id=dashboard&limit=' + frame.persons.length)).json();
  const recent = ev.events.slice().reverse();
  frame.persons.forEach((p,i)=>{ p._event_id = recent[i] ? recent[i].id : null; });
  document.getElementById('cards').innerHTML = frame.persons.map(card).join('');
  const st = ev.stats || {};
  document.getElementById('stats').textContent =
     'logged events - ' + Object.entries(st).map(([k,v])=>`${k}: ${v}`).join('  ·  ');
}

async function review(eid){
  const verdict = document.getElementById('v'+eid).value;
  await fetch(`/v1/events/${eid}/review`, {method:'POST', headers:{'Content-Type':'application/json'},
              body: JSON.stringify({reviewer:'operator', verdict})});
  toast(`Review filed for event ${eid}: ${verdict}`);
}

function toast(msg){
  const t=document.getElementById('toast'); t.textContent=msg; t.style.opacity=1;
  setTimeout(()=>t.style.opacity=0, 2000);
}

document.getElementById('toggle').onclick = (e)=>{
  if(timer){ clearInterval(timer); timer=null; e.target.textContent='▶ Start live'; }
  else { tick(); timer=setInterval(tick, 1500); e.target.textContent='⏸ Pause'; }
};
loadProfiles().then(tick);
</script>
</body>
</html>"""
