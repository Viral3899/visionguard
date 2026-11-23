'use strict';
const $ = (id) => document.getElementById(id);
const state = { cameras: [], clients: [], selected: null, events: [], editMode: null, editPoints: [], socket: null, loggedIn: false, role: null, editingCamera: false, detectionTypes: {}, viewMode: 'grid', gridLayout: 'auto' };

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

// Buzzer audio and urgent speech alert system
let buzzerAudio = null;
let buzzerTimer = null;

function initAudio() {
  if (!buzzerAudio) {
    buzzerAudio = new Audio('/api/alerts/buzzer.mp3');
    buzzerAudio.loop = true;
  }
}

function playBuzzerSiren(durationSeconds=10) {
  initAudio();
  if (!buzzerAudio) return;
  try {
    if (buzzerTimer) {
      clearTimeout(buzzerTimer);
      buzzerTimer = null;
    }
    buzzerAudio.currentTime = 0;
    const p = buzzerAudio.play();
    if (p && typeof p.catch === 'function') {
      p.catch(e => console.debug('Autoplay note:', e));
    }
    buzzerTimer = setTimeout(() => {
      try {
        buzzerAudio.pause();
        buzzerAudio.currentTime = 0;
      } catch (e) {}
      buzzerTimer = null;
    }, durationSeconds * 1000);
  } catch (err) {
    console.warn('Buzzer playback error:', err);
  }
}

function speakUrgentAlert(kind, cameraName) {
  if (!('speechSynthesis' in window)) return;
  const k = String(kind || '').toLowerCase();
  const cam = cameraName ? `at ${cameraName}` : '';
  let phrase = '';
  if (k.includes('fire')) {
    phrase = `Fire! Fire! Fire! Fire detected ${cam}.`;
  } else if (k.includes('smoke')) {
    phrase = `Fire! Smoke! Fire! Smoke detected ${cam}.`;
  } else if (k.includes('viol') || k.includes('fight') || k.includes('assault')) {
    phrase = `Violence! Violence! Violence! Violence detected ${cam}.`;
  } else if (k.includes('weap') || k.includes('knife') || k.includes('gun')) {
    phrase = `Weapon! Weapon! Weapon! Weapon detected ${cam}.`;
  } else if (k.includes('intr')) {
    phrase = `Intrusion! Intrusion! Intrusion! Zone intrusion ${cam}.`;
  } else if (k.includes('fall')) {
    phrase = `Emergency! Fall! Fall! Fall detected ${cam}.`;
  } else {
    phrase = `Alert! Alert! Alert! Security incident ${cam}.`;
  }

  try {
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(phrase);
    utter.rate = 1.05;
    utter.pitch = 1.1;
    utter.volume = 1.0;
    window.speechSynthesis.speak(utter);
  } catch (err) {
    console.warn('Speech synthesis error:', err);
  }
}

async function refreshCameras() {
  const container = $('camera-list');
  try {
    const cams = await req('/api/cameras');
    state.cameras = cams;
    if (!cams.some(c => c.id === state.selected)) state.selected = cams[0]?.id || null;
    renderCameraList();
    if (state.viewMode === 'grid') {
      renderCameraGrid();
    } else {
      renderSelected();
    }
    renderDetectionConfig();
    if (state.role === 'admin') loadClients().catch(() => {});
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
      <span class="cam-icon">▣</span><span class="cam-text"><b>${esc(cam.name)}</b><small>${esc(cam.status)} · ${cam.current_people || 0} in view · 5m: ${cam.unique_people_5m || 0} ${cam.is_recording ? '· <span style="color:#ff4466">● REC</span>' : ''}</small></span>
      <i class="light ${dotClass(cam.status)}"></i>
    </button>`).join('');
  container.querySelectorAll('[data-cam]').forEach(btn => btn.onclick = () => {
    state.selected = btn.dataset.cam; resetEditor(); renderCameraList(); renderSelected(true); renderDetectionConfig();
    switchToFocusView(btn.dataset.cam);
  });
}

function renderSelected(force=false) {
  const cam = getCam();
  const video = $('video-feed');
  if (!video) return;
  $('selected-title').textContent = cam?.name || 'Camera preview';
  $('selected-subtitle').textContent = cam ? `${cam.source} · ${cam.error || 'AI-powered live monitoring'}` : 'Select a camera to view analytics';
  $('selected-status').textContent = (cam?.status || 'NO SIGNAL').toUpperCase();
  $('selected-status').className = 'status-pill ' + (cam?.status || '').toLowerCase().replace(/[^a-z]/g,'');
  $('current-count').textContent = `${cam?.current_people || 0} in view · ${cam?.unique_people_5m || 0} (5m dedup)`;
  $('current-fps').textContent = `${cam?.fps || 0} FPS`;

  const recBtn = $('record-camera-btn');
  const recToolBtn = $('record-camera-tool-btn');
  const isRec = Boolean(cam?.is_recording);
  const durSec = cam?.recording_duration || 0;
  if (recBtn) {
    recBtn.textContent = isRec ? `⏹ Stop (${formatSecs(durSec)})` : '⏺ Record';
    recBtn.classList.toggle('recording-active', isRec);
    recBtn.disabled = !cam || !cam.enabled;
  }
  if (recToolBtn) {
    recToolBtn.textContent = isRec ? `⏹ Stop Recording (${formatSecs(durSec)})` : '⏺ Start Recording';
    recToolBtn.classList.toggle('recording-active', isRec);
    recToolBtn.disabled = !cam || !cam.enabled;
  }
  if (isRec) {
    if (!recordingTimerInterval) startRecordingTimer();
  } else {
    stopRecordingTimer();
  }

  updateObjectButtons();

  $('draw-zone').disabled = $('draw-line').disabled = !cam;
  $('remove-zone').disabled = !cam || !cam.zone?.length;
  $('remove-line').disabled = !cam || !cam.tripwire?.length;
  $('toggle-camera').disabled = $('delete-camera').disabled = $('edit-camera').disabled = $('test-camera').disabled = !cam;
  $('delete-camera').classList.toggle('hidden', state.role !== 'admin'); $('edit-camera').classList.toggle('hidden', state.role !== 'admin');
  $('toggle-camera').textContent = cam?.enabled ? 'Pause camera' : 'Resume camera';

  if (state.viewMode !== 'focus') {
    return;
  }

  const next = cam?.enabled ? cam.id : null;
  const activeId = video.dataset.source || null;
  if (force || activeId !== next) {
    video.dataset.source = next || '';
    if (next) {
      $('video-empty').classList.add('hidden');
      video.classList.remove('hidden');
      video.onerror = () => {
        if (video.dataset.source !== next || state.viewMode !== 'focus') return;
        $('video-empty').classList.remove('hidden');
        $('video-empty').querySelector('h3').textContent = 'Connecting to camera…';
        $('video-empty').querySelector('p').textContent = cam?.error || 'Waiting for camera frames. Check the source URL and camera status.';
        setTimeout(() => {
          if (video.dataset.source === next && state.viewMode === 'focus') {
            video.src = `/stream/${encodeURIComponent(next)}`;
          }
        }, 3000);
      };
      video.onload = () => {
        if (video.dataset.source !== next) return;
        video.classList.remove('hidden');
        $('video-empty').classList.add('hidden');
        if (video.naturalWidth && video.naturalHeight) {
          $('video-display').style.aspectRatio = `${video.naturalWidth} / ${video.naturalHeight}`;
        }
        ptzController.reset();
      };
      video.src = `/stream/${encodeURIComponent(next)}`;
    } else {
      video.removeAttribute('src');
      video.classList.add('hidden');
      $('video-empty').classList.remove('hidden');
      $('video-empty').querySelector('h3').textContent = 'No active video feed';
      $('video-empty').querySelector('p').textContent = 'Add an RTSP stream, USB webcam or a video file to start.';
      ptzController.reset();
    }
  }
  $('video-stamp').classList.toggle('hidden', !cam || !cam.enabled);
  renderDetectionBadges(cam);
}

function setViewMode(mode, cameraId=null) {
  state.viewMode = mode;
  if (cameraId) {
    state.selected = cameraId;
  }
  const isGrid = mode === 'grid';
  const gridSection = $('multi-camera-grid-section');
  const workspaceGrid = $('workspace-grid');
  const topBreadcrumb = $('breadcrumb-current');
  const navGrid = $('nav-grid');
  const navSingle = $('nav-single');

  if (gridSection) gridSection.classList.toggle('hidden', !isGrid);
  if (workspaceGrid) workspaceGrid.classList.toggle('hidden', isGrid);

  if (navGrid) navGrid.classList.toggle('selected', isGrid);
  if (navSingle) navSingle.classList.toggle('selected', !isGrid);

  const cam = getCam();
  if (topBreadcrumb) {
    topBreadcrumb.textContent = isGrid ? 'LIVE CAMERA MATRIX' : (cam ? cam.name.toUpperCase() : 'SINGLE CAMERA FOCUS');
  }

  if (isGrid) {
    // Detach focus stream to immediately free the connection
    const video = $('video-feed');
    if (video) {
      video.removeAttribute('src');
      video.dataset.source = '';
    }
    ptzController.reset();
    renderCameraGrid(true);
  } else {
    // Detach all grid streams to prevent socket starvation
    const grid = $('camera-matrix-grid');
    if (grid) {
      grid.querySelectorAll('img.tile-stream').forEach(img => {
        img.removeAttribute('src');
      });
    }
    renderSelected(true);
    renderDetectionConfig();
  }
}

function switchToGridView() {
  setViewMode('grid');
}

function switchToFocusView(camId) {
  setViewMode('focus', camId);
  resetEditor();
  renderCameraList();
  $('workspace-grid')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderCameraGrid(forceRebuild=false) {
  const grid = $('camera-matrix-grid');
  if (!grid) return;
  if (!state.cameras.length) {
    grid.innerHTML = `
      <div class="empty-grid-msg">
        <div class="empty-icon" style="font-size:48px;color:#23374d;">▣</div>
        <h3 style="margin:0;color:#94a8c0;">No cameras configured</h3>
        <p style="margin:0;color:#6b7d9a;font-size:13px;">Add an RTSP stream, USB camera or video source to start live multi-camera monitoring.</p>
        <button class="primary-btn" id="grid-add-camera">+ Add camera</button>
      </div>`;
    $('grid-add-camera')?.addEventListener('click', openModal);
    return;
  }

  grid.className = 'camera-matrix-grid layout-' + state.gridLayout;

  const currentTiles = grid.querySelectorAll('.camera-tile');
  const existingIds = Array.from(currentTiles).map(t => t.dataset.tileId).join(',');
  const newIds = state.cameras.map(c => c.id).join(',');

  if (forceRebuild || existingIds !== newIds) {
    grid.innerHTML = state.cameras.map(cam => {
      const isOnline = ['online', 'connecting', 'reconnecting'].includes(cam.status);
      const streamSrc = (state.viewMode === 'grid' && cam.enabled) ? `/stream/${encodeURIComponent(cam.id)}` : '';
      const isSelected = cam.id === state.selected;
      const cfg = cam.config || {};
      const badges = [];
      if (cfg.fire_enabled) badges.push('<span class="detection-badge fire" style="font-size:9px;padding:2px 5px">🔥 Fire</span>');
      if (cfg.violence_enabled) badges.push('<span class="detection-badge violence" style="font-size:9px;padding:2px 5px">⚔ Weapon</span>');
      if (cfg.anomaly_enabled) badges.push('<span class="detection-badge anomaly" style="font-size:9px;padding:2px 5px">🧍 Anomaly</span>');

      return `
        <div class="camera-tile ${isSelected ? 'active-selected' : ''}" data-tile-id="${esc(cam.id)}" title="Double-click to open full view with PTZ controls (${esc(cam.name)})">
          <div class="tile-top-bar">
            <div class="tile-title-group">
              <i class="light ${dotClass(cam.status)}"></i>
              <span class="tile-name">${esc(cam.name)}</span>
            </div>
            <div class="tile-actions">
              <button class="tile-obj-btn ${cam.show_objects !== false && cfg.show_objects !== 0 ? 'active' : ''}" data-cam-id="${esc(cam.id)}" title="${cam.show_objects !== false && cfg.show_objects !== 0 ? 'Daily Objects: ON (Click to Hide)' : 'Daily Objects: OFF (Click to Show)'}">
                ${cam.show_objects !== false && cfg.show_objects !== 0 ? '👁 Obj' : '👁‍🗨 Off'}
              </button>
              <button class="tile-rec-btn ${cam.is_recording ? 'active-rec' : ''}" data-cam-id="${esc(cam.id)}" title="${cam.is_recording ? 'Stop Recording' : 'Start Recording'}">
                ${cam.is_recording ? `<span class="rec-dot-pulse"></span> <span class="tile-rec-timer" data-cam-id="${esc(cam.id)}">${formatSecs(cam.recording_duration || 0)}</span> Stop` : '⏺ Rec'}
              </button>
            </div>
            <div class="tile-stats">
              <span>👥 ${cam.current_people || 0}</span>
              <span>5m: ${cam.unique_people_5m || 0}</span>
              <span>${cam.fps || 0} FPS</span>
            </div>
          </div>
          ${cam.enabled ? `
            <img class="tile-stream" ${streamSrc ? `src="${streamSrc}"` : ''} alt="${esc(cam.name)} stream"
                 onerror="this.style.display='none'; if(this.nextElementSibling) this.nextElementSibling.style.display='flex';"
                 onload="this.style.display='block'; if(this.nextElementSibling) this.nextElementSibling.style.display='none';" />
            <div class="tile-placeholder" style="display:none">
              <div class="tile-placeholder-icon">▣</div>
              <b>${esc(cam.name)}</b>
              <small>${esc(cam.error || 'Awaiting video feed…')}</small>
            </div>
          ` : `
            <div class="tile-placeholder">
              <div class="tile-placeholder-icon">⏸</div>
              <b>${esc(cam.name)}</b>
              <small>Camera paused</small>
            </div>
          `}
          <div class="tile-bottom-bar">
            <div class="tile-badges">${badges.join(' ')}</div>
            <div class="tile-dblclick-hint">Double-click to open ↗</div>
          </div>
        </div>
      `;
    }).join('');

    grid.querySelectorAll('.camera-tile').forEach(tile => {
      const camId = tile.dataset.tileId;
      tile.onclick = (e) => {
        state.selected = camId;
        grid.querySelectorAll('.camera-tile').forEach(t => t.classList.toggle('active-selected', t.dataset.tileId === camId));
        renderCameraList();
      };
      tile.ondblclick = (e) => {
        e.preventDefault();
        switchToFocusView(camId);
      };
    });

    grid.querySelectorAll('.tile-rec-btn').forEach(btn => {
      btn.onclick = (e) => {
        e.stopPropagation();
        toggleRecording(btn.dataset.camId);
      };
    });
    grid.querySelectorAll('.tile-obj-btn').forEach(btn => {
      btn.onclick = (e) => {
        e.stopPropagation();
        toggleObjectsDisplay(btn.dataset.camId);
      };
    });
  } else {
    // In-place update: never re-assign active src to avoid tearing down connections
    state.cameras.forEach(cam => {
      const tile = grid.querySelector(`[data-tile-id="${cam.id}"]`);
      if (!tile) return;
      tile.classList.toggle('active-selected', cam.id === state.selected);
      const nameEl = tile.querySelector('.tile-name');
      if (nameEl) nameEl.textContent = cam.name;
      const lightEl = tile.querySelector('.light');
      if (lightEl) lightEl.className = 'light ' + dotClass(cam.status);
      const statsEl = tile.querySelector('.tile-stats');
      if (statsEl) statsEl.innerHTML = `<span>👥 ${cam.current_people || 0}</span><span>5m: ${cam.unique_people_5m || 0}</span><span>${cam.fps || 0} FPS</span>`;

      const actionsEl = tile.querySelector('.tile-actions');
      if (actionsEl) {
        const objBtn = actionsEl.querySelector('.tile-obj-btn');
        if (objBtn) {
          const isObjOn = cam.show_objects !== false && (cam.config?.show_objects !== 0);
          objBtn.className = `tile-obj-btn ${isObjOn ? 'active' : ''}`;
          objBtn.textContent = isObjOn ? '👁 Obj' : '👁‍🗨 Off';
        }
        const recBtn = actionsEl.querySelector('.tile-rec-btn');
        if (recBtn) {
          const isRec = Boolean(cam.is_recording);
          recBtn.className = `tile-rec-btn ${isRec ? 'active-rec' : ''}`;
          recBtn.innerHTML = isRec ? `<span class="rec-dot-pulse"></span> <span class="tile-rec-timer" data-cam-id="${esc(cam.id)}">${formatSecs(cam.recording_duration || 0)}</span> Stop` : '⏺ Rec';
        }
      }

      const img = tile.querySelector('img.tile-stream');
      if (state.viewMode === 'grid' && cam.enabled && img && !img.getAttribute('src')) {
        img.src = `/stream/${encodeURIComponent(cam.id)}`;
      }
    });
  }
}

const ptzController = {
  scale: 1.0,
  panX: 0,
  panY: 0,
  isPanning: false,
  startX: 0,
  startY: 0,
  minScale: 1.0,
  maxScale: 5.0,
  step: 0.25,

  init() {
    const display = $('video-display');
    const viewport = $('video-viewport');
    if (!display || !viewport) return;

    display.addEventListener('wheel', (e) => {
      if (state.editMode) return;
      e.preventDefault();
      const delta = e.deltaY < 0 ? this.step : -this.step;
      const rect = display.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      this.zoomAt(delta, mouseX, mouseY, rect.width, rect.height);
    }, { passive: false });

    viewport.addEventListener('mousedown', (e) => {
      if (state.editMode || this.scale <= 1.0) return;
      if (e.button !== 0) return;
      this.isPanning = true;
      this.startX = e.clientX - this.panX;
      this.startY = e.clientY - this.panY;
      viewport.classList.add('is-panning');
      e.preventDefault();
    });

    window.addEventListener('mousemove', (e) => {
      if (!this.isPanning) return;
      const rect = display.getBoundingClientRect();
      this.panX = e.clientX - this.startX;
      this.panY = e.clientY - this.startY;
      this.clamp(rect.width, rect.height);
      this.apply();
    });

    window.addEventListener('mouseup', () => {
      if (this.isPanning) {
        this.isPanning = false;
        viewport.classList.remove('is-panning');
      }
    });

    viewport.addEventListener('dblclick', (e) => {
      if (state.editMode) return;
      e.preventDefault();
      if (this.scale > 1.05) {
        this.reset();
      } else {
        const rect = display.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        this.zoomAt(1.0, mouseX, mouseY, rect.width, rect.height);
      }
    });

    $('ptz-zoom-in')?.addEventListener('click', () => this.zoomIn());
    $('ptz-zoom-out')?.addEventListener('click', () => this.zoomOut());
    $('ptz-zoom-reset')?.addEventListener('click', () => this.reset());
    $('ptz-fullscreen')?.addEventListener('click', () => this.toggleFullscreen());
  },

  zoomAt(deltaScale, focalX, focalY, containerW, containerH) {
    const oldScale = this.scale;
    let newScale = Math.max(this.minScale, Math.min(this.maxScale, oldScale + deltaScale));
    if (Math.abs(newScale - oldScale) < 0.001) return;

    const centerX = containerW / 2;
    const centerY = containerH / 2;
    const dx = focalX - centerX - this.panX;
    const dy = focalY - centerY - this.panY;

    const ratio = newScale / oldScale;
    this.panX -= dx * (ratio - 1);
    this.panY -= dy * (ratio - 1);
    this.scale = newScale;

    this.clamp(containerW, containerH);
    this.apply();
  },

  zoomIn() {
    const display = $('video-display');
    const rect = display ? display.getBoundingClientRect() : { width: 800, height: 450 };
    this.zoomAt(this.step * 1.5, rect.width / 2, rect.height / 2, rect.width, rect.height);
  },

  zoomOut() {
    const display = $('video-display');
    const rect = display ? display.getBoundingClientRect() : { width: 800, height: 450 };
    this.zoomAt(-this.step * 1.5, rect.width / 2, rect.height / 2, rect.width, rect.height);
  },

  reset() {
    this.scale = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isPanning = false;
    this.apply();
  },

  clamp(containerW, containerH) {
    if (this.scale <= 1.0) {
      this.scale = 1.0;
      this.panX = 0;
      this.panY = 0;
      return;
    }
    const maxPanX = (containerW * (this.scale - 1)) / 2;
    const maxPanY = (containerH * (this.scale - 1)) / 2;
    this.panX = Math.max(-maxPanX, Math.min(maxPanX, this.panX));
    this.panY = Math.max(-maxPanY, Math.min(maxPanY, this.panY));
  },

  apply() {
    const viewport = $('video-viewport');
    const video = $('video-feed');
    const overlay = $('draw-overlay');
    const badge = $('ptz-zoom-badge');
    const minimap = $('ptz-minimap-container');
    const crop = $('ptz-minimap-crop');

    const isZoomed = this.scale > 1.01;
    if (viewport) viewport.classList.toggle('ptz-draggable', isZoomed);

    const transform = isZoomed ? `translate(${this.panX}px, ${this.panY}px) scale(${this.scale})` : '';
    if (video) video.style.transform = transform;
    if (overlay) overlay.style.transform = transform;

    if (badge) badge.textContent = `${Math.round(this.scale * 100)}%`;

    if (minimap && crop) {
      minimap.classList.toggle('hidden', !isZoomed);
      if (isZoomed) {
        const cropW = 100 / this.scale;
        const cropH = 100 / this.scale;
        const display = $('video-display');
        const rect = display ? display.getBoundingClientRect() : { width: 800, height: 450 };
        const maxPanX = (rect.width * (this.scale - 1)) / 2 || 1;
        const maxPanY = (rect.height * (this.scale - 1)) / 2 || 1;

        const normX = this.panX / maxPanX;
        const normY = this.panY / maxPanY;

        const left = (50 - (normX * (50 - cropW / 2))) - cropW / 2;
        const top = (50 - (normY * (50 - cropH / 2))) - cropH / 2;

        crop.style.width = `${cropW}%`;
        crop.style.height = `${cropH}%`;
        crop.style.left = `${Math.max(0, Math.min(100 - cropW, left))}%`;
        crop.style.top = `${Math.max(0, Math.min(100 - cropH, top))}%`;
      }
    }
  },

  toggleFullscreen() {
    const display = $('video-display');
    if (!display) return;
    if (!document.fullscreenElement) {
      display.requestFullscreen?.().catch(() => {});
    } else {
      document.exitFullscreen?.().catch(() => {});
    }
  }
};

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
  if ($('stat-passers')) $('stat-passers').textContent = s.unique_people_5m || 0;
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
  ptzController.reset();
}
function beginDrawing(mode){
  if (!getCam()) return toast('Select a camera first',true);
  if (state.viewMode !== 'focus') switchToFocusView(state.selected);
  ptzController.reset();
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
        const isSecurity = ['fire_detected','smoke','heat_source','violence','weapon','fight','knife','gun','assault','intrusion','fall_detected','unusual_activity'].includes(event.kind);
        if (isSecurity) {
          playBuzzerSiren(10);
          speakUrgentAlert(event.kind, event.camera_name);
        }
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
        <label class="toggle-item">
          <input type="checkbox" id="cfg-show_objects" ${cfg.show_objects !== 0 ? 'checked' : ''} />
          <span class="toggle-label"><strong>👁 Daily Objects Overlay</strong><br><small>Everyday items & person boxes (Alarming threats always show)</small></span>
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
  const showObjEl = $('cfg-show_objects');
  if (showObjEl) showObjEl.onchange = () => updateCameraConfig('show_objects', showObjEl.checked);
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
  playBuzzerSiren(10);
  speakUrgentAlert('fire', 'Test Camera');
  try {
    const res = await req('/api/alerts/test', { method: 'POST', body: JSON.stringify({ camera_id: getCam()?.id }) });
    if (res.results?.tts || res.results?.buzzer) {
      toast('Voice siren & 10s buzzer test triggered!');
    } else {
      toast('TTS test failed. Check pyttsx3 installation.', true);
    }
  } catch(e) { toast(e.message, true); }
}

async function testAllAlerts() {
  playBuzzerSiren(10);
  speakUrgentAlert('violence', 'Test Camera');
  const resultsEl = $('alert-test-results');
  resultsEl.classList.remove('hidden');
  resultsEl.innerHTML = '<p>Testing 10s buzzer sound and urgent voice alert...</p>'; 
  try {
    const res = await req('/api/alerts/test', { method: 'POST', body: JSON.stringify({ camera_id: getCam()?.id }) });
    const r = res.results || {};
    resultsEl.innerHTML = `
      <div class="test-result ${r.buzzer ? 'pass' : 'fail'}">
        <span>${r.buzzer ? '✓' : '✗'}</span> Buzzer Sound (10s): ${r.buzzer ? 'Playing buzzer MP3 (10s)' : 'Failed'}
      </div>
      <div class="test-result ${r.tts ? 'pass' : 'fail'}">
        <span>${r.tts ? '✓' : '✗'}</span> Voice Siren (TTS): ${r.tts ? 'Triggered urgent speech' : 'Failed or not enabled'}
      </div>
      <div class="test-result ${r.telegram ? 'pass' : 'fail'}">
        <span>${r.telegram ? '✓' : '✗'}</span> Telegram: ${r.telegram ? 'Sent test message' : 'Disabled / not configured'}
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

// Video recording controls
const cameraRecordingStartTimes = {};
let recordingTimerInterval = null;
let recordingStartTime = 0;

function formatSecs(sec) {
  const s = Math.max(0, Math.floor(sec || 0));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${String(m).padStart(2, '0')}:${String(rem).padStart(2, '0')}`;
}

async function toggleRecording(cameraId) {
  const cam = cameraId ? state.cameras.find(c => c.id === cameraId) : getCam();
  if (!cam) return toast('Select a camera to record', true);

  const isRec = Boolean(cam.is_recording);
  try {
    if (isRec) {
      const res = await req(`/api/cameras/${encodeURIComponent(cam.id)}/recording/stop`, { method: 'POST' });
      cam.is_recording = false;
      delete cameraRecordingStartTimes[cam.id];
      toast(`Recording saved: ${res.filename} (${res.duration_seconds}s, ${Math.round((res.file_size_bytes || 0) / 1024)} KB)`);
      if (getCam()?.id === cam.id) stopRecordingTimer();
      await refreshRecordings();
    } else {
      const res = await req(`/api/cameras/${encodeURIComponent(cam.id)}/recording/start`, { method: 'POST' });
      cam.is_recording = true;
      cameraRecordingStartTimes[cam.id] = Date.now();
      toast(`Recording started for ${cam.name}`);
      if (getCam()?.id === cam.id) startRecordingTimer();
    }
    renderSelected();
    renderCameraList();
    if (state.viewMode === 'grid') renderCameraGrid();
  } catch (err) {
    toast(`Recording error: ${err.message}`, true);
  }
}

async function toggleObjectsDisplay(cameraId) {
  const cam = cameraId ? state.cameras.find(c => c.id === cameraId) : getCam();
  if (!cam) return toast('Select a camera first', true);
  const current = cam.show_objects !== undefined ? cam.show_objects : (cam.config?.show_objects !== 0);
  const next = !current;
  try {
    await req(`/api/cameras/${encodeURIComponent(cam.id)}/config`, {
      method: 'PATCH',
      body: JSON.stringify({ show_objects: next })
    });
    cam.show_objects = next;
    if (!cam.config) cam.config = {};
    cam.config.show_objects = next ? 1 : 0;
    updateObjectButtons();
    if (state.viewMode === 'grid') renderCameraGrid();
    toast(`Daily objects ${next ? 'enabled' : 'hidden'} for ${cam.name}. Alarming threat alerts remain always visible.`);
  } catch (err) {
    toast(`Failed to update object display: ${err.message}`, true);
  }
}

async function toggleAllObjectsDisplay() {
  if (!state.cameras.length) return;
  const anyOn = state.cameras.some(c => (c.show_objects !== undefined ? c.show_objects : (c.config?.show_objects !== 0)));
  const next = !anyOn;
  try {
    await Promise.all(state.cameras.map(c =>
      req(`/api/cameras/${encodeURIComponent(c.id)}/config`, {
        method: 'PATCH',
        body: JSON.stringify({ show_objects: next })
      })
    ));
    state.cameras.forEach(c => {
      c.show_objects = next;
      if (!c.config) c.config = {};
      c.config.show_objects = next ? 1 : 0;
    });
    updateObjectButtons();
    if (state.viewMode === 'grid') renderCameraGrid();
    toast(`Daily objects ${next ? 'enabled' : 'hidden'} across all cameras. Alarming threat alerts remain always visible.`);
  } catch (err) {
    toast(`Error toggling objects: ${err.message}`, true);
  }
}

function updateObjectButtons() {
  const cam = getCam();
  const show = cam ? (cam.show_objects !== undefined ? cam.show_objects : (cam.config?.show_objects !== 0)) : true;
  const btn = $('toggle-objects-btn');
  if (btn) {
    btn.textContent = show ? '👁 Objects: ON' : '👁 Objects: OFF';
    btn.style.borderColor = show ? '#06b6d4' : '#2d445b';
    btn.style.color = show ? '#22d3ee' : '#94a8c0';
  }
  const gridBtn = $('grid-toggle-objects-btn');
  if (gridBtn) {
    const anyOn = state.cameras.some(c => (c.show_objects !== undefined ? c.show_objects : (c.config?.show_objects !== 0)));
    gridBtn.textContent = anyOn ? '👁 Objects: ON' : '👁 Objects: OFF';
    gridBtn.style.borderColor = anyOn ? '#06b6d4' : '#2d445b';
    gridBtn.style.color = anyOn ? '#22d3ee' : '#94a8c0';
  }
}

function startRecordingTimer() {
  if (recordingTimerInterval) clearInterval(recordingTimerInterval);
  const cam = getCam();
  if (cam && !cameraRecordingStartTimes[cam.id]) {
    cameraRecordingStartTimes[cam.id] = Date.now() - ((cam.recording_duration || 0) * 1000);
  }
  recordingStartTime = (cam && cameraRecordingStartTimes[cam.id]) ? cameraRecordingStartTimes[cam.id] : Date.now();
  $('rec-badge')?.classList.remove('hidden');
  recordingTimerInterval = setInterval(() => {
    const elapsedSec = Math.floor((Date.now() - recordingStartTime) / 1000);
    const m = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
    const s = String(elapsedSec % 60).padStart(2, '0');
    if ($('rec-timer')) $('rec-timer').textContent = `REC ${m}:${s}`;
  }, 1000);
}

function stopRecordingTimer() {
  if (recordingTimerInterval) {
    clearInterval(recordingTimerInterval);
    recordingTimerInterval = null;
  }
  $('rec-badge')?.classList.add('hidden');
  if ($('rec-timer')) $('rec-timer').textContent = 'REC 00:00';
}

async function refreshRecordings() {
  const tbody = $('recordings-body');
  if (!tbody) return;
  try {
    const recs = await req('/api/recordings');
    const counter = $('recordings-counter');
    if (counter) counter.textContent = `${recs.length} clip${recs.length === 1 ? '' : 's'}`;

    if (!recs.length) {
      tbody.innerHTML = '<tr><td colspan="6" class="muted" style="text-align:center;padding:24px;">No recordings yet. Click "Record" on any camera to save video clips.</td></tr>';
      return;
    }

    tbody.innerHTML = recs.map(r => {
      const sizeKB = Math.round((r.file_size_bytes || 0) / 1024);
      const sizeMB = (sizeKB / 1024).toFixed(1);
      const sizeStr = sizeKB > 1024 ? `${sizeMB} MB` : `${sizeKB} KB`;
      return `<tr>
        <td><strong>${esc(r.camera_name || r.camera_id)}</strong></td>
        <td><code>${esc(r.filename)}</code></td>
        <td>${fmtTime(r.created_at || r.start_time)}</td>
        <td>${r.duration_seconds || 0}s</td>
        <td>${sizeStr}</td>
        <td>
          <button class="outline-btn small-btn play-rec-btn" data-filename="${esc(r.filename)}">▶ Play</button>
          <a class="outline-btn small-btn" href="/api/recordings/${encodeURIComponent(r.filename)}" download="${esc(r.filename)}">↓ Download</a>
          ${state.role === 'admin' ? `<button class="danger-soft small-btn del-rec-btn" data-filename="${esc(r.filename)}">✕ Delete</button>` : ''}
        </td>
      </tr>`;
    }).join('');

    tbody.querySelectorAll('.play-rec-btn').forEach(btn => {
      btn.onclick = () => playRecordingModal(btn.dataset.filename);
    });
    tbody.querySelectorAll('.del-rec-btn').forEach(btn => {
      btn.onclick = async () => {
        if (!confirm(`Delete recording ${btn.dataset.filename}?`)) return;
        try {
          await req(`/api/recordings/${encodeURIComponent(btn.dataset.filename)}`, { method: 'DELETE' });
          toast('Recording deleted');
          await refreshRecordings();
        } catch (e) {
          toast(e.message, true);
        }
      };
    });
  } catch (err) {
    console.warn('Failed to load recordings:', err);
  }
}

function playRecordingModal(filename) {
  const container = $('recording-player-container');
  const video = $('recording-video-player');
  const title = $('preview-title');
  if (!container || !video) return;

  if (title) title.textContent = `Viewing: ${filename}`;
  video.src = `/api/recordings/${encodeURIComponent(filename)}`;
  container.classList.remove('hidden');
  container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
  ptzController.init();
  $('home-grid-btn')?.addEventListener('click', switchToGridView);
  $('focus-back-btn')?.addEventListener('click', switchToGridView);
  $('nav-grid')?.addEventListener('click', (e) => { e.preventDefault(); switchToGridView(); });
  $('nav-single')?.addEventListener('click', (e) => { e.preventDefault(); switchToFocusView(state.selected || state.cameras[0]?.id); });
  $('grid-refresh-btn')?.addEventListener('click', () => { refreshCameras(); toast('Camera feeds refreshed'); });
  $('grid-layout-selector')?.querySelectorAll('.layout-btn').forEach(btn => {
    btn.onclick = () => {
      $('grid-layout-selector').querySelectorAll('.layout-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.gridLayout = btn.dataset.layout;
      const grid = $('camera-matrix-grid');
      if (grid) grid.className = 'camera-matrix-grid layout-' + state.gridLayout;
    };
  });

  document.addEventListener('keydown', (e) => {
    if (['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) return;
    if (e.key === 'Escape' || e.key === 'h' || e.key === 'H') {
      closeModal();
      resetEditor();
      $('snapshot-modal')?.classList.add('hidden');
      switchToGridView();
    } else if (state.viewMode === 'focus') {
      if (e.key === '+' || e.key === '=') {
        e.preventDefault();
        ptzController.zoomIn();
      } else if (e.key === '-' || e.key === '_') {
        e.preventDefault();
        ptzController.zoomOut();
      } else if (e.key === '0') {
        e.preventDefault();
        ptzController.reset();
      } else if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        ptzController.toggleFullscreen();
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        toggleRecording();
      } else if (e.key === 'o' || e.key === 'O') {
        e.preventDefault();
        toggleObjectsDisplay();
      }
    }
  });
  
  // Recording controls
  $('record-camera-btn')?.addEventListener('click', () => toggleRecording());
  $('record-camera-tool-btn')?.addEventListener('click', () => toggleRecording());
  $('refresh-recordings-btn')?.addEventListener('click', () => refreshRecordings());
  $('toggle-objects-btn')?.addEventListener('click', () => toggleObjectsDisplay());
  $('grid-toggle-objects-btn')?.addEventListener('click', () => toggleAllObjectsDisplay());
  $('close-preview-btn')?.addEventListener('click', () => {
    $('recording-player-container')?.classList.add('hidden');
    $('recording-video-player')?.pause();
  });

  // Per-camera live timer ticker
  setInterval(() => {
    state.cameras.forEach(cam => {
      if (cam.is_recording) {
        if (!cameraRecordingStartTimes[cam.id]) {
          cameraRecordingStartTimes[cam.id] = Date.now() - ((cam.recording_duration || 0) * 1000);
        }
        const elapsedSec = Math.floor((Date.now() - cameraRecordingStartTimes[cam.id]) / 1000);
        const timerStr = formatSecs(elapsedSec);
        document.querySelectorAll(`.tile-rec-timer[data-cam-id="${cam.id}"]`).forEach(el => {
          el.textContent = timerStr;
        });
        if (state.selected === cam.id) {
          if ($('rec-timer')) $('rec-timer').textContent = `REC ${timerStr}`;
          const recBtn = $('record-camera-btn');
          if (recBtn) recBtn.textContent = `⏹ Stop (${timerStr})`;
        }
      } else {
        delete cameraRecordingStartTimes[cam.id];
      }
    });
  }, 1000);

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
  const results = await Promise.allSettled([refreshCameras(), refreshStats(), refreshEvents(), refreshRecordings()]);
  for (const result of results) if (result.status === 'rejected') console.error('Dashboard section failed:', result.reason);
  setViewMode('grid');
  connectWS();
  if(!window.__vgPolling){
    window.__vgPolling=true;
    setInterval(()=> {if(state.loggedIn) Promise.all([refreshCameras(),refreshStats()]).catch(console.warn);},4000);
    setInterval(()=> {if(state.loggedIn) refreshEvents().catch(console.warn);},15000);
    setInterval(()=> {if(state.loggedIn) refreshRecordings().catch(console.warn);},10000);
  }
}
init();