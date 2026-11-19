'use strict';
const $ = (id) => document.getElementById(id);
const state = { cameras: [], clients: [], selected: null, events: [], editMode: null, editPoints: [], socket: null, loggedIn: false, role: null, editingCamera: false, detectionTypes: {} };

const kindTitle = {
  intrusion: 'Zone intrusion', entry: 'Person entered', exit: 'Person exited',
  fire_detected: 'Fire detected', smoke: 'Smoke detected', heat_source: 'Heat source',
  violence: 'Violence', weapon: 'Weapon detected', fight: 'Fight detected',
  assault: 'Assault', proximity_alert: 'Suspicious proximity',
  fall_detected: 'Fall detected', loitering: 'Loitering', unusual_activity: 'Unusual activity'
};
const kindSeverity = {
  fire_detected: 'critical', smoke: 'high', heat_source: 'low',
  violence: 'critical', weapon: 'critical', fight: 'critical', assault: 'high', proximity_alert: 'medium',
  fall_detected: 'high', loitering: 'medium', unusual_activity: 'high', intrusion: 'high', entry: 'low', exit: 'low'
};
const kindIcon = {
  intrusion: '⚑', entry: '↗', exit: '↘',
  fire_detected: '🔥', smoke: '💨', heat_source: '🌡',
  violence: '⚔', weapon: '🔫', fight: '🤜', assault: '👊', proximity_alert: '⚠',
  fall_detected: '🧍‍♂️➡️🧎', loitering: '🕐', unusual_activity: '⚠'
};
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtTime = (s) => s ? new Date(s).toLocaleString('en-IN', { timeZone:'Asia/Kolkata', month:'short', day:'numeric', hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:true }) : '—';
const getCam = () => state.cameras.find(c => c.id === state.selected);
const dotClass = (s) => ['online','connecting','reconnecting'].includes(s) ? s : s === 'loading model' ? 'loading' : '';

async function req(path, opts={}) {
  const response = await fetch(path, { credentials:'same-origin', ...opts,
    headers: { 'Content-Type':'application/json', ...(opts.headers || {}) } });
  let data; try { data = await response.json(); } catch { data = null; }
  if (!response.ok) {
    const detail = data && data.detail;
    throw new Error(Array.isArray(detail) ? detail.map(x => x.msg).join('; ') : detail || `HTTP ${response.status}`);
  }
  return data;
}

function toast(message, isError=false) {
  const div = document.createElement('div');
  div.className = 'toast' + (isError ? ' error' : '');
  const h = document.createElement('strong'); h.textContent = isError ? 'Action failed' : 'VisionGuard';
  const p = document.createElement('div'); p.textContent = message;
  div.append(h,p); $('toast-area').appendChild(div);
  setTimeout(() => div.remove(), 5000);
}

async function checkSession() {
  const session = await req('/api/session');
  state.loggedIn = session.authenticated; state.role = session.role || 'admin';
  $('client-admin-panel').classList.toggle('hidden', state.role !== 'admin');
  $('add-camera').classList.toggle('hidden', state.role !== 'admin'); $('add-camera-side').classList.toggle('hidden', state.role !== 'admin'); $('empty-add').classList.toggle('hidden', state.role !== 'admin');
  $('auth').classList.toggle('hidden', session.authenticated);
  $('application').classList.toggle('hidden', !session.authenticated);
  $('logout').classList.toggle('hidden', !session.password_required || !session.authenticated);
  return session.authenticated;
}

async function refreshCameras() {
  const container = $('camera-list');
  try {
  const cams = await req('/api/cameras');
  state.cameras = cams;
  if (!cams.some(c => c.id === state.selected)) state.selected = cams[0]?.id || null;
  renderCameraList(); renderSelected(); renderDetectionConfig();
  if(state.role==='admin') loadClients().catch(()=>{});
  } catch (error) {
    container.innerHTML = `<div class="camera-load-error"><b>Could not load cameras</b><small>${esc(error.message)}</small><button class="outline-btn" id="retry-cameras">Retry</button></div>`;
    $('retry-cameras')?.addEventListener('click', () => refreshCameras().catch(() => {}));
    throw error;
  }
}

function renderCameraList() {
  const container = $('camera-list');
  if (!state.cameras.length) { container.innerHTML = '<p class="muted small" style="padding:0 12px">No cameras configured</p>'; return; }
  container.innerHTML = state.cameras.map(cam => `
    <button class="camera-row ${state.selected === cam.id ? 'active':''}" data-cam="${esc(cam.id)}">
      <span class="cam-icon">▣</span><span class="cam-text"><b>${esc(cam.name)}</b><small>${esc(cam.status)} · ${cam.current_people || 0} people</small></span>
      <i class="light ${dotClass(cam.status)}"></i>
    </button>`).join('');
  container.querySelectorAll('[data-cam]').forEach(btn => btn.onclick = () => {
    state.selected = btn.dataset.cam; resetEditor(); renderCameraList(); renderSelected(true); renderDetectionConfig();
  });
}

function renderSelected(force=false) {
  const cam = getCam();
  const video = $('video-feed');
  $('selected-title').textContent = cam?.name || 'Camera preview';
  $('selected-subtitle').textContent = cam ? `${cam.source} · ${cam.error || 'AI-powered live monitoring'}` : 'Select a camera to view analytics';
  $('selected-status').textContent = (cam?.status || 'NO SIGNAL').toUpperCase();
  $('selected-status').className = 'status-pill ' + (cam?.status || '').toLowerCase().replace(/[^a-z]/g,'');
  $('current-count').textContent = `${cam?.current_people || 0} people`;
  $('current-fps').textContent = `${cam?.fps || 0} FPS`;
  $('draw-zone').disabled = $('draw-line').disabled = !cam;
  $('remove-zone').disabled = !cam || !cam.zone?.length;
  $('remove-line').disabled = !cam || !cam.tripwire?.length;
  $('toggle-camera').disabled = $('delete-camera').disabled = $('edit-camera').disabled = $('test-camera').disabled = !cam;
  $('delete-camera').classList.toggle('hidden', state.role !== 'admin'); $('edit-camera').classList.toggle('hidden', state.role !== 'admin');
  $('toggle-camera').textContent = cam?.enabled ? 'Pause camera' : 'Resume camera';
  const next = cam?.enabled ? cam.id : null;
  const activeId = video.dataset.source || null;
  if (force || activeId !== next) {
    video.dataset.source = next || '';
    video.classList.add('hidden');
    $('video-empty').classList.remove('hidden');
    $('video-empty').querySelector('h3').textContent = next ? 'Connecting to camera…' : 'No active video feed';
    $('video-empty').querySelector('p').textContent = next ? (cam?.error || 'Waiting for the first processed frame…') : 'Add an RTSP stream, USB webcam or a video file to start.';
    if (next) {
      video.onerror = () => {
        if (video.dataset.source !== next) return;
        $('video-empty').classList.remove('hidden');
        $('video-empty').querySelector('h3').textContent = 'Connecting to camera…';
        $('video-empty').querySelector('p').textContent = cam?.error || 'Waiting for camera frames. Check the source URL and camera status.';
        setTimeout(() => { if (video.dataset.source === next) video.src = `/stream/${encodeURIComponent(next)}?v=${Date.now()}`; }, 3000);
      };
      video.onload = () => {
        if (video.dataset.source !== next) return;
        video.classList.remove('hidden'); $('video-empty').classList.add('hidden');
        if (video.naturalWidth && video.naturalHeight) $('video-display').style.aspectRatio = `${video.naturalWidth} / ${video.naturalHeight}`;
      };
      video.src = `/stream/${encodeURIComponent(next)}?v=${Date.now()}`;
    } else { video.removeAttribute('src'); }
  }
  $('video-stamp').classList.toggle('hidden', !cam || !cam.enabled);
  renderDetectionBadges(cam);
}

function renderDetectionBadges(cam) {
  const container = $('detection-badges');
  if (!cam || !cam.config) { container.innerHTML = ''; return; }
  const cfg = cam.config;
  const badges = [];
  if (cfg.fire_enabled) badges.push('<span class="detection-badge fire">🔥 Fire + smoke confirmation</span>');
  if (cfg.violence_enabled) badges.push('<span class="detection-badge violence">⚔ Violence / knife / weapon</span>');
  if (cfg.anomaly_enabled) badges.push('<span class="detection-badge anomaly">🧍 Anomaly</span>');
  if (cfg.object_enabled) badges.push('<span class="detection-badge object">📦 Objects</span>');
  if (cfg.tts_enabled) badges.push('<span class="detection-badge tts">🔊 Voice</span>');
  container.innerHTML = badges.join(' ');
}

async function refreshStats() {
  const s = await req('/api/stats');
  $('stat-online').textContent = `${s.online} / ${s.cameras}`;
  $('stat-cameras-detail').textContent = `${s.cameras - s.online} awaiting signal`;
  $('stat-people').textContent = s.current_people;
  $('stat-intrusions').textContent = s.intrusions_24h;
  $('stat-crossings').textContent = s.entered + s.exited;
  $('stat-crossings-detail').textContent = `${s.entered} entries / ${s.exited} exits`;
  // Fire and violence stats from events_by_kind_24h
  const fireCount = (s.events_by_kind_24h?.fire_detected || 0) + (s.events_by_kind_24h?.smoke || 0) + (s.events_by_kind_24h?.heat_source || 0);
  const violenceCount = (s.events_by_kind_24h?.violence || 0) + (s.events_by_kind_24h?.weapon || 0) + (s.events_by_kind_24h?.fight || 0) + (s.events_by_kind_24h?.assault || 0) + (s.events_by_kind_24h?.proximity_alert || 0);
  $('stat-fire').textContent = fireCount;
  $('stat-violence').textContent = violenceCount;
}

async function refreshEvents() {
  const kind = $('filter-kind').value;
  const qs = kind ? `?kind=${encodeURIComponent(kind)}&limit=80` : '?limit=80';
  state.events = await req('/api/events' + qs);
  renderEvents();
}

function renderEvents() {
  const body = $('events-body');
  if (!state.events.length) {
    body.innerHTML = '<tr><td colspan="6" class="table-empty">No events recorded yet. Add a camera and configure a detection rule.</td></tr>';
    return;
  }
  body.innerHTML = state.events.map(item => {
    const severity = kindSeverity[item.kind] || 'low';
    const severityClass = `severity-${severity}`;
    const icon = kindIcon[item.kind] || '●';
    return `<tr>
      <td><span class="event-pill ${esc(item.kind)}">${icon} ${esc(kindTitle[item.kind] || item.kind)}</span></td>
      <td><strong>${esc(item.camera_name)}</strong></td>
      <td>${esc(item.message)}</td>
      <td>${fmtTime(item.created_at)}</td>
      <td><span class="${severityClass}">${esc(severity.toUpperCase())}</span></td>
      <td>${item.snapshot ? `<button class="snapshot-btn" data-snapshot="${esc(item.snapshot)}">View ↗</button>` : '—'}</td>
    </tr>`;
  }).join('');
  body.querySelectorAll('[data-snapshot]').forEach(btn => btn.onclick = () => {
    $('snapshot-img').src = `/api/snapshots/${encodeURIComponent(btn.dataset.snapshot)}`;
    $('snapshot-modal').classList.remove('hidden');
  });
  const recent = state.events.filter(item => ['intrusion','fire_detected','smoke','violence','weapon','fight','assault','fall_detected','proximity_alert','unusual_activity'].includes(item.kind)).slice(0, 5);
  $('alert-counter').textContent = String(recent.length);
  $('alert-list').innerHTML = recent.length ? recent.map(item => `<div class="alert-item"><span class="alert-dot">${kindIcon[item.kind] || '⚑'}</span><div>
    <strong>${esc(item.camera_name)} — ${esc(kindTitle[item.kind] || item.kind)}</strong><p>${esc(item.message)}</p><small>${fmtTime(item.created_at)}</small></div></div>`).join('')
    : '<p class="empty-alerts">No critical alerts yet.<br />Your monitored zones and AI detections will appear here.</p>';
}

function openModal(){ state.editingCamera=false; $('camera-modal-title').textContent='Add a video source'; $('camera-form').reset(); $('camera-modal').classList.remove('hidden'); $('camera-error').textContent=''; $('client-assignment-wrap').classList.toggle('hidden',state.role!=='admin'); }
async function editCameraModal(){ const cam=getCam(); if(!cam)return; state.editingCamera=true; $('camera-modal-title').textContent='Edit camera source'; $('camera-form').reset(); $('camera-form').elements.name.value=cam.name; try{const raw=await req(`/api/cameras/${cam.id}/source`);$('camera-form').elements.source.value=raw.source;}catch(e){toast(e.message,true);return;} $('camera-form').elements.client_id.value=cam.client_id||''; $('client-assignment-wrap').classList.toggle('hidden',state.role!=='admin'); $('camera-modal').classList.remove('hidden'); }
function closeModal(){ $('camera-modal').classList.add('hidden'); state.editingCamera=false; }
async function addCamera(e){
  e.preventDefault(); $('camera-error').textContent='';
  const data=Object.fromEntries(new FormData(e.target).entries());
  if(!data.client_id) delete data.client_id;
  const wasEditing=state.editingCamera;
  try {
    let cam;
    if(state.editingCamera){ cam=await req(`/api/cameras/${getCam().id}`,{method:'PATCH',body:JSON.stringify({name:data.name,source:data.source,client_id:data.client_id||null})}); }
    else { cam=await req('/api/cameras',{ method:'POST',body:JSON.stringify(data) }); }
    closeModal(); e.target.reset(); state.selected=cam.id;
    await refreshCameras(); await refreshStats(); toast(wasEditing?'Camera settings saved. Reconnecting…':'Camera added. Connecting to video source…'); state.editingCamera=false;
  } catch(e) { $('camera-error').textContent=e.message; }
}

async function loadClients(){
  state.clients=await req('/api/clients');
  const select=$('camera-client'); if(select) select.innerHTML='<option value="">Administrator / Unassigned</option>'+state.clients.map(c=>`<option value="${esc(c.id)}">${esc(c.name)} (${esc(c.username)})</option>`).join('');
  $('client-list').innerHTML=state.clients.length?state.clients.map(c=>`<div class="client-row"><b>${esc(c.name)}</b><small>${esc(c.username)} · ${c.active?'Active':'Disabled'}</small><button class="outline-btn" data-client-toggle="${esc(c.id)}" data-active="${c.active?'0':'1'}">${c.active?'Disable':'Enable'}</button></div>`).join(''):'No clients created yet.';
  document.querySelectorAll('[data-client-toggle]').forEach(b=>b.onclick=async()=>{await req(`/api/clients/${b.dataset.clientToggle}`,{method:'PATCH',body:JSON.stringify({active:b.dataset.active==='1'})});loadClients();});
}
async function createClient(e){e.preventDefault();$('client-error').textContent='';const d=Object.fromEntries(new FormData(e.target).entries());try{await req('/api/clients',{method:'POST',body:JSON.stringify(d)});e.target.reset();await loadClients();toast('Client workspace created. Share its login securely.');}catch(err){$('client-error').textContent=err.message;}}

function resetEditor(){
  state.editMode=null; state.editPoints=[];
  $('draw-overlay').classList.add('hidden'); $('save-drawing').classList.add('hidden');
  $('cancel-drawing').classList.add('hidden'); $('editor-hint').classList.add('hidden');
  $('draw-zone').classList.remove('hidden'); $('draw-line').classList.remove('hidden');
}
function beginDrawing(mode){
  if (!getCam()) return toast('Select a camera first',true);
  state.editMode=mode; state.editPoints=[];
  $('draw-overlay').classList.remove('hidden'); $('save-drawing').classList.remove('hidden');
  $('cancel-drawing').classList.remove('hidden'); $('draw-zone').classList.add('hidden');
  $('draw-line').classList.add('hidden'); $('editor-hint').classList.remove('hidden');
  $('editor-hint').textContent = mode === 'zone' ? 'Tap 3 or more points around the restricted area, then Save rule.' : 'Tap the starting and ending points of the counting line. Crossing direction depends on the line direction.';
  renderDrawing();
}
function renderDrawing(){
  const points = state.editPoints.map(([x,y]) => [Math.round(x*1000),Math.round(y*1000)]);
  const color=state.editMode==='zone' ? '#ff728a' : '#ffd070';
  const path = points.length ? `<polyline points="${points.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${color}" stroke-width="5" stroke-dasharray="12 9" />` : '';
  const filled = state.editMode==='zone' && points.length >= 3 ? `<polygon points="${points.map(p=>p.join(',')).join(' ')}" fill="#ff728a" fill-opacity=".16" />` : '';
  const dots = points.map(p => `<circle cx="${p[0]}" cy="${p[1]}" r="10" fill="${color}" stroke="#08111d" stroke-width="4" />`).join('');
  $('draw-overlay').innerHTML=filled+path+dots;
  $('save-drawing').disabled = !(state.editMode === 'zone' ? points.length >= 3 : points.length === 2);
}
async function saveDrawing(){
  const cam=getCam(); if (!cam) return;
  const key=state.editMode==='zone' ? 'zone' : 'tripwire';
  try {
    await req(`/api/cameras/${cam.id}`,{method:'PATCH',body:JSON.stringify({[key]:state.editPoints})});
    resetEditor(); await refreshCameras(); toast(`${key === 'zone' ? 'Restricted zone' : 'Counting line'} saved`);
  } catch(e) { toast(e.message,true); }
}
async function clearRule(key){
  const cam=getCam(); if(!cam) return;
  try { await req(`/api/cameras/${cam.id}`,{method:'PATCH',body:JSON.stringify({[key]:[]})}); resetEditor(); await refreshCameras(); toast('Rule cleared'); }
  catch(e){ toast(e.message,true); }
}

function connectWS(){
  if(!state.loggedIn) return;
  const protocol=location.protocol==='https:' ? 'wss:' : 'ws:';
  const ws=new WebSocket(`${protocol}//${location.host}/ws/alerts`);
  state.socket=ws;
  ws.onmessage=(msg)=>{
    try {
      const payload=JSON.parse(msg.data);
      if(payload.type==='event'){
        const event=payload.event;
        toast(`${kindTitle[event.kind] || event.kind} at ${event.camera_name}`);
        refreshEvents().catch(console.warn); refreshStats().catch(console.warn);
      }
    } catch(e){ console.warn(e); }
  };
  ws.onclose=()=>{ if(state.loggedIn && state.socket === ws) setTimeout(connectWS,3000); };
}

// Detection config
async function renderDetectionConfig() {
  const container = $('detection-config-content');
  const cam = getCam();
  if (!cam) {
    container.innerHTML = '<p class="muted small">Select a camera to configure detection modules.</p>';
    return;
  }
  const cfg = cam.config || {};
  container.innerHTML = `
    <div class="config-section">
      <h3>${esc(cam.name)} — Detection Modules</h3>
      <div class="toggle-grid">
        <label class="toggle-item">
          <input type="checkbox" id="cfg-fire" ${cfg.fire_enabled ? 'checked' : ''} />
          <span class="toggle-label"><strong>🔥 Fire & Smoke</strong><br><small>Detect fire, smoke, heat sources</small></span>
        </label>
        <label class="toggle-item">
          <input type="checkbox" id="cfg-violence" ${cfg.violence_enabled ? 'checked' : ''} />
          <span class="toggle-label"><strong>⚔ Violence & Weapons</strong><br><small>Fight, assault, knife, gun detection</small></span>
        </label>
        <label class="toggle-item">
          <input type="checkbox" id="cfg-anomaly" ${cfg.anomaly_enabled ? 'checked' : ''} />
          <span class="toggle-label"><strong>🧍 Anomaly (Pose)</strong><br><small>Pose-based fall checks (not a trained action classifier)</small></span>
        </label>
        <label class="toggle-item">
          <input type="checkbox" id="cfg-object" ${cfg.object_enabled ? 'checked' : ''} />
          <span class="toggle-label"><strong>📦 General Objects</strong><br><small>All COCO classes (vehicles, animals, items)</small></span>
        </label>
      </div>
      <div class="toggle-grid" style="margin-top: 16px;">
        <label class="toggle-item">
          <input type="checkbox" id="cfg-tts" ${cfg.tts_enabled ? 'checked' : ''} />
          <span class="toggle-label"><strong>🔊 Voice Siren</strong><br><small>Speak alerts locally</small></span>
        </label>
        <label class="toggle-item">
          <input type="checkbox" id="cfg-telegram" ${cfg.telegram_enabled ? 'checked' : ''} />
          <span class="toggle-label"><strong>📨 Telegram Security Alerts</strong><br><small>Only suspicious detections above 75% confidence</small></span>
        </label>
      </div>
      <div class="form-group" style="margin-top: 16px;">
        <label>Sensitivity <input type="range" id="cfg-sensitivity" min="0.1" max="5" step="0.1" value="${cfg.sensitivity || 1.0}" /><span id="cfg-sensitivity-value">${cfg.sensitivity || 1.0}x</span></label>
      </div>
    </div>
  `;
  
  // Add event listeners
  ['fire','violence','anomaly','object','tts','telegram'].forEach(key => {
    const el = $(`cfg-${key}`);
    if (el) el.onchange = () => updateCameraConfig(key + '_enabled', el.checked);
  });
  const sens = $('cfg-sensitivity');
  if (sens) sens.oninput = (e) => {
    $('cfg-sensitivity-value').textContent = parseFloat(e.target.value).toFixed(1) + 'x';
  };
  sens.onchange = (e) => updateCameraConfig('sensitivity', parseFloat(e.target.value));
}

async function updateCameraConfig(key, value) {
  const cam = getCam();
  if (!cam) return;
  try {
    await req(`/api/cameras/${cam.id}/config`, { method: 'PATCH', body: JSON.stringify({ [key]: value }) });
    await refreshCameras();
    toast(`Updated ${key}`);
  } catch(e) { toast(e.message, true); }
}

// Alert config
async function testTTS() {
  try {
    const res = await req('/api/alerts/test', { method: 'POST', body: JSON.stringify({ camera_id: getCam()?.id }) });
    if (res.results?.tts) {
      toast('Voice siren test triggered!');
    } else {
      toast('TTS test failed. Check pyttsx3 installation.', true);
    }
  } catch(e) { toast(e.message, true); }
}

async function testAllAlerts() {
  const resultsEl = $('alert-test-results');
  resultsEl.classList.remove('hidden');
  resultsEl.innerHTML = '<p>Testing local voice alert...</p>'; 
  try {
    const res = await req('/api/alerts/test', { method: 'POST', body: JSON.stringify({ camera_id: getCam()?.id }) });
    const r = res.results || {};
    resultsEl.innerHTML = `
      <div class="test-result ${r.tts ? 'pass' : 'fail'}">
        <span>${r.tts ? '✓' : '✗'}</span> Voice Siren (TTS): ${r.tts ? 'Triggered' : 'Failed or not enabled'}
      </div>
    `;
    toast('Alert test completed');
  } catch(e) { 
    resultsEl.innerHTML = `<div class="test-result fail"><span>✗</span> Error: ${e.message}</div>`;
    toast(e.message, true);
  }
}

$('tts-rate').oninput = (e) => $('tts-rate-value').textContent = e.target.value;
$('tts-volume').oninput = (e) => $('tts-volume-value').textContent = e.target.value + '%';

async function runUploadedVideoTest(event) {
  event.preventDefault();
  const input = $('test-video-file');
  const file = input.files?.[0];
  if (!file) { toast('Choose a video file first.', true); return; }
  if (file.size > 250 * 1024 * 1024) { toast('Video must be 250 MB or smaller.', true); return; }
  const button = $('run-video-test');
  const progress = $('test-video-progress');
  const results = $('test-video-results');
  const output = $('test-video-output');
  button.disabled = true; button.textContent = 'Analysing…';
  progress.classList.remove('hidden'); results.classList.add('hidden'); output.classList.add('hidden');
  const form = new FormData(); form.append('file', file);
  try {
    const response = await fetch('/api/test-videos', {method:'POST', body:form, credentials:'same-origin'});
    let data = {}; try { data = await response.json(); } catch {}
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    const rows = Object.entries(data.events || {});
    const unavailable = Object.entries(data.models_loaded || {}).filter(([, loaded]) => !loaded).map(([name]) => name);
    const diagnostics = data.fire_detector || {};
    const notices = (data.warnings || []).map(note => `<p class="field-hint">⚠ ${esc(note)}</p>`).join('');
    const photos = (data.confirmed_events || []).filter(e => e.snapshot_url);
    results.innerHTML = `<h3>Analysis complete</h3>
      <div class="test-summary-grid"><div><b>${data.frames_analyzed}</b><small>Frames analysed</small></div><div><b>${data.duration_seconds}s</b><small>Video duration</small></div><div><b>${Math.round(data.minimum_confidence * 100)}%</b><small>Alert confidence floor</small></div></div>
      <p>Fire detector: <strong>${esc(diagnostics.status || 'unknown')}</strong> · ${diagnostics.successful_frames || 0} frames evaluated · classes: ${esc((diagnostics.enabled_labels || []).join(', ') || 'none')}.</p>
      ${notices}
      <h4>Qualified detection frames</h4>${rows.length ? rows.map(([kind,count])=>`<div class="test-event-row"><span>${esc(kind.replaceAll('_',' '))}</span><b>${count}</b></div>`).join('') : '<p class="muted small">No event with genuine model confidence ≥50% was detected.</p>'}
      <h4>Confirmed unusual incidents</h4><p>${data.confirmed_alert_types?.length ? data.confirmed_alert_types.map(esc).join(', ') : 'No event was confirmed on 2+ analysed frames.'}</p>
      ${photos.length ? `<h4>Saved incident photos (${photos.length})</h4><div class="test-event-photos">${photos.map(e=>`<figure><a href="${esc(e.snapshot_url)}" target="_blank" rel="noopener"><img loading="lazy" src="${esc(e.snapshot_url)}" alt="${esc(e.kind)} incident photo"></a><figcaption>${esc(e.kind.replaceAll('_',' '))} · ${Math.round(e.confidence*100)}% · ${e.time_seconds}s</figcaption></figure>`).join('')}</div>` : ''}
      <p class="field-hint">Only confirmed unusual incident photos are saved. The full annotated video remains available for reviewing all frames.</p>
      ${unavailable.length ? `<p>Models unavailable: ${unavailable.map(esc).join(', ')}.</p>` : ''}
      <a class="outline-btn" href="${esc(data.annotated_video_url)}" target="_blank" rel="noopener">Open annotated video</a>`;
    results.classList.remove('hidden');
    output.src = data.annotated_video_url; output.classList.remove('hidden');
    toast('Video analysis completed.');
  } catch (err) { results.innerHTML = `<p class="error">${esc(err.message)}</p>`; results.classList.remove('hidden'); toast(err.message, true); }
  finally { progress.classList.add('hidden'); button.disabled = false; button.textContent = 'Upload & analyse'; }
}

async function init(){
  $('clock').textContent=new Date().toLocaleString('en-IN',{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit',hour12:true});
  setInterval(()=> $('clock').textContent=new Date().toLocaleString('en-IN',{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit',hour12:true}),10000);
  $('test-video-form')?.addEventListener('submit', runUploadedVideoTest);
  $('test-video-file')?.addEventListener('change', e => $('test-video-filename').textContent = e.target.files?.[0]?.name || 'No file selected');
  $('login-form').addEventListener('submit', async(e)=>{
    e.preventDefault(); $('login-error').textContent='';
    try { await req('/api/login',{method:'POST',body:JSON.stringify({username:$('username').value,password:$('password').value})});
      $('password').value=''; if(await checkSession()) await loadDashboard(); }
    catch(err){ $('login-error').textContent=err.message; }
  });
  $('logout').onclick=async()=>{try{await req('/api/logout',{method:'POST'});}catch{} state.loggedIn=false;
    if(state.socket)state.socket.close(); await checkSession();};
  ['add-camera','add-camera-side','empty-add'].forEach(id=>$(id).onclick=openModal);
  ['close-modal','cancel-modal'].forEach(id=>$(id).onclick=closeModal);
  $('camera-form').addEventListener('submit',addCamera);
  $('test-camera-source').onclick = async () => {
    const source = $('camera-form').elements.source.value.trim();
    if (!source) { $('camera-error').textContent = 'Enter a camera source first.'; return; }
    $('camera-error').textContent = 'Testing source…';
    try { const result = await req('/api/cameras/test-source', {method:'POST', body:JSON.stringify({source})});
      $('camera-error').textContent = result.message; $('camera-error').style.color = result.ok ? '#35d49a' : '#ff7b8a';
    } catch (err) { $('camera-error').textContent = err.message; $('camera-error').style.color = '#ff7b8a'; }
  };
  $('discover-usb').onclick = async () => {
    $('camera-error').textContent = 'Searching USB cameras…';
    try { const found = await req('/api/cameras/discover');
      if (!found.length) { $('camera-error').textContent = 'No USB camera found. You can still add an RTSP or offline source manually.'; return; }
      const first = found[0]; $('camera-form').elements.source.value = first.source;
      if (!$('camera-form').elements.name.value.trim()) $('camera-form').elements.name.value = first.name;
      $('camera-error').textContent = `Found ${found.length} camera(s). Selected ${first.name} (${first.width}×${first.height}).`;
    } catch (err) { $('camera-error').textContent = err.message; }
  };
  
  $('start-webcam-test').onclick = async () => {
    const btn = $('start-webcam-test');
    const errorEl = $('camera-error');
    btn.disabled = true;
    btn.textContent = 'Starting webcam…';
    errorEl.textContent = 'Opening webcam…';
    
    try {
      // First discover to get available cameras
      const found = await req('/api/cameras/discover');
      if (!found.length) {
        errorEl.textContent = 'No USB camera found.';
        errorEl.style.color = '#ff7b8a';
        return;
      }
      
      // Auto-fill with first camera
      const first = found[0];
      $('camera-form').elements.source.value = first.source;
      if (!$('camera-form').elements.name.value.trim()) {
        $('camera-form').elements.name.value = first.name;
      }
      
      errorEl.textContent = `Starting webcam test with ${first.name} (${first.width}×${first.height})…`;
      errorEl.style.color = '#35d49a';
      
      // Add a temporary camera and test it
      const testName = 'Webcam Test - ' + new Date().toLocaleTimeString();
      const cam = await req('/api/cameras', {
        method: 'POST',
        body: JSON.stringify({
          name: testName,
          source: first.source,
          enabled: true
        })
      });
      
      errorEl.textContent = 'Webcam connected! Redirecting to live view…';
      errorEl.style.color = '#35d49a';
      
      // Select the new camera
      state.selected = cam.id;
      await refreshCameras();
      closeModal();
      
      // The live view will auto-connect
      setTimeout(() => {
        refreshCameras().catch(console.warn);
      }, 2000);
      
    } catch (err) {
      errorEl.textContent = err.message;
      errorEl.style.color = '#ff7b8a';
    } finally {
      btn.disabled = false;
      btn.textContent = '🎥 Start Webcam Test';
    }
  };
  $('camera-modal').onclick=(e)=>{if(e.target===$('camera-modal'))closeModal();};
  $('filter-kind').onchange=()=>refreshEvents().catch(e=>toast(e.message,true));
  $('draw-zone').onclick=()=>beginDrawing('zone'); $('draw-line').onclick=()=>beginDrawing('line');
  $('save-drawing').onclick=saveDrawing; $('cancel-drawing').onclick=resetEditor;
  $('remove-zone').onclick=()=>clearRule('zone'); $('remove-line').onclick=()=>clearRule('tripwire');
  $('toggle-camera').onclick=async()=>{
    const cam=getCam(); if(!cam)return;
    try{await req(`/api/cameras/${cam.id}`,{method:'PATCH',body:JSON.stringify({enabled:!cam.enabled})});
      await Promise.all([refreshCameras(),refreshStats()]); toast(cam.enabled?'Camera paused':'Camera resumed');}
    catch(e){toast(e.message,true);}
  };
  $('delete-camera').onclick=async()=>{
    const cam=getCam(); if(!cam || !confirm(`Delete camera "${cam.name}"? Previous event logs will remain.`))return;
    try{await req(`/api/cameras/${cam.id}`,{method:'DELETE'});state.selected=null;resetEditor();
      await Promise.all([refreshCameras(),refreshStats(),refreshEvents()]);toast('Camera deleted');}
    catch(e){toast(e.message,true);}
  };
  $('draw-overlay').onclick=(event)=>{
    if(!state.editMode) return;
    if(state.editMode==='line' && state.editPoints.length===2) return;
    if(state.editMode==='zone' && state.editPoints.length >=30) return;
    const box=$('draw-overlay').getBoundingClientRect();
    const x=Math.max(0,Math.min(1,(event.clientX-box.left)/box.width));
    const y=Math.max(0,Math.min(1,(event.clientY-box.top)/box.height));
    state.editPoints.push([+x.toFixed(4),+y.toFixed(4)]); renderDrawing();
  };
  $('close-snapshot').onclick=()=>$('snapshot-modal').classList.add('hidden');
  $('snapshot-modal').onclick=(e)=>{if(e.target===$('snapshot-modal'))$('snapshot-modal').classList.add('hidden');};
  document.addEventListener('keydown',(e)=>{if(e.key==='Escape'){closeModal();resetEditor();$('snapshot-modal').classList.add('hidden');}});
  
  // Alert config buttons
  $('edit-camera').onclick = editCameraModal;
  $('test-camera').onclick = async()=>{const cam=getCam();if(!cam)return;try{const r=await req(`/api/cameras/${cam.id}/test`,{method:'POST'});toast(r.message);setTimeout(refreshCameras,1500);}catch(e){toast(e.message,true);}};
  $('client-form').addEventListener('submit',createClient);
  $('test-tts').onclick = testTTS;
  $('test-all-alerts').onclick = testAllAlerts;
  
  // Clear all data
  $('clear-all-data').onclick = async () => {
    if (!confirm('This will permanently delete ALL events, alert logs, snapshots, and test videos. Are you sure?')) return;
    const btn = $('clear-all-data');
    const statusEl = $('clear-data-status');
    btn.disabled = true;
    btn.textContent = 'Clearing…';
    statusEl.textContent = 'Clearing all data…';
    statusEl.style.color = '#fb923c';
    try {
      const res = await req('/api/admin/clear-all-data', {method: 'POST'});
      statusEl.textContent = '✓ ' + res.message;
      statusEl.style.color = '#22c55e';
      toast('All data cleared successfully');
      // Refresh dashboard
      await Promise.all([refreshCameras(), refreshStats(), refreshEvents()]);
    } catch (err) {
      statusEl.textContent = '✗ ' + err.message;
      statusEl.style.color = '#ff7b8a';
      toast(err.message, true);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Clear All Data & Reset';
    }
  };
  
  try{ if(await checkSession()) await loadDashboard(); }
  catch(err){ toast(err.message,true); }
}
async function loadDashboard(){
  const results = await Promise.allSettled([refreshCameras(), refreshStats(), refreshEvents()]);
  for (const result of results) if (result.status === 'rejected') console.error('Dashboard section failed:', result.reason);
  connectWS();
  if(!window.__vgPolling){
    window.__vgPolling=true;
    setInterval(()=> {if(state.loggedIn) Promise.all([refreshCameras(),refreshStats()]).catch(console.warn);},4000);
    setInterval(()=> {if(state.loggedIn) refreshEvents().catch(console.warn);},15000);
  }
}
init();