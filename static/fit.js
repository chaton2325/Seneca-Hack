/* =========================================================================
   fitness.js — Client (camera → Socket.IO → UI + TTS coach + aligned overlay)
   Goals:
   - Keep selectors/IDs unchanged (safe drop-in)
   - Clear structure & English comments
   - Robustness: feature/permission checks, error handling, backpressure
   - Accurate overlay alignment with object-fit: contain
   ========================================================================= */

   (() => {
    'use strict';
  
    /* =========================
       0) DOM references (cache)
       ========================= */
    const els = {
      video:       document.getElementById('video'),
      overlay:     document.getElementById('overlay'),
      feedback:    document.getElementById('feedbackText'),
      scoreText:   document.getElementById('scoreText'),
      gaugeArc:    document.getElementById('gaugeArc'),
      repCount:    document.getElementById('repCount'),
      tempo:       document.getElementById('tempoValue'),
      rom:         document.getElementById('romValue'),
      sym:         document.getElementById('symValue'),
      exerciseSel: document.getElementById('exerciseSelect'),
      exerciseName:document.getElementById('exerciseName'),
      startBtn:    document.getElementById('startBtn'),
      stopBtn:     document.getElementById('stopBtn'),
      resetBtn:    document.getElementById('resetBtn'),
      fps:         document.getElementById('fpsValue'),
      safetyPill:  document.getElementById('safetyPill'),
      liveBadge:   document.getElementById('liveBadge'),
      voiceToggle: document.getElementById('voiceToggle'),
      voiceRate:   document.getElementById('voiceRate'),
      voiceVol:    document.getElementById('voiceVol'),
      log:         document.getElementById('logList'),
    };
  
    /* =========================
       1) Canvas / DPR utilities
       ========================= */
    const getDPR = () => window.devicePixelRatio || 1;
  
    // Resize overlay canvas to match its CSS container (crisp on HiDPI)
    function resizeOverlayToContainer(overlayEl) {
      if (!overlayEl) return;
      const cssW = overlayEl.clientWidth  || overlayEl.parentElement?.clientWidth  || 0;
      const cssH = overlayEl.clientHeight || overlayEl.parentElement?.clientHeight || 0;
  
      const ratio = getDPR();
      overlayEl.width  = Math.floor(cssW * ratio);
      overlayEl.height = Math.floor(cssH * ratio);
  
      const ctx = overlayEl.getContext('2d');
      // Draw using CSS pixels (not raw device pixels)
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    }
  
    // Compute the actual drawing rect for the video when using object-fit: contain
    function getVideoDrawRect(videoEl, overlayEl) {
      const cw = overlayEl.clientWidth;
      const ch = overlayEl.clientHeight;
      const vw = videoEl.videoWidth  || cw;
      const vh = videoEl.videoHeight || ch;
  
      const containerRatio = cw / ch;
      const videoRatio     = vw / vh;
  
      if (videoRatio > containerRatio) {
        // Horizontal bars (letterboxing)
        const displayW = cw;
        const displayH = cw / videoRatio;
        const offsetY  = (ch - displayH) / 2;
        return { x: 0, y: offsetY, w: displayW, h: displayH };
      } else {
        // Vertical bars (pillarboxing)
        const displayH = ch;
        const displayW = ch * videoRatio;
        const offsetX  = (cw - displayW) / 2;
        return { x: offsetX, y: 0, w: displayW, h: displayH };
      }
    }
  
    /* ================
       2) Socket.IO
       ================ */
    // Guard if Socket.IO client is not available
    if (typeof io !== 'function') {
      console.error('[FitMaster] Socket.IO client not found.');
    }
  
    const socket = typeof io === 'function' ? io() : null;
  
    if (socket) {
      socket.on('connect',    () => setLive(true));
      socket.on('disconnect', () => setLive(false));
      socket.on('pose_result', (data) => {
        updateUI(data);                     // Update numbers, gauge, safety pill
        drawSkeleton(data?.landmarks);      // Draw aligned skeleton overlay
        window.VoiceCoach?.speakFeedback(data); // Optional TTS guidance
      });
    }
  
    function setLive(state) {
      if (!els.liveBadge) return;
      els.liveBadge.classList.toggle('online', state);
      els.liveBadge.classList.toggle('offline', !state);
      els.liveBadge.textContent = state ? 'EN LIGNE' : 'HORS LIGNE';
    }
  
    /* ==========================================
       3) Camera capture + frame sending (15 FPS)
       ========================================== */
    let stream = null;
    let sending = false;              // Backpressure guard
    let fpsTick = 0, lastFpsTs = performance.now();
  
    // Hidden canvas used for encoding the current video frame
    const captureCanvas = document.createElement('canvas');
    const cctx = captureCanvas.getContext('2d', { willReadFrequently: true });
  
    async function startCamera() {
      if (stream) return; // already started
  
      // Feature/permission guard
      if (!navigator.mediaDevices?.getUserMedia) {
        log('Camera not supported on this browser.');
        return;
      }
  
      try {
        // Ask for a 1280×720 stream (browser may negotiate differently)
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false
        });
  
        els.video.srcObject = stream;
        await els.video.play();
  
        // Ensure overlay matches the displayed area
        resizeOverlayToContainer(els.overlay);
  
        // Match capture canvas to the *native* video resolution for best quality
        captureCanvas.width  = els.video.videoWidth;
        captureCanvas.height = els.video.videoHeight;
  
        // Recompute overlay size on viewport changes
        window.addEventListener('resize', () => resizeOverlayToContainer(els.overlay), { passive: true });
  
        // Start sending frames with backpressure at ~15 FPS
        sendLoop();
      } catch (err) {
        console.error('[FitMaster] Camera error:', err);
        log(`Camera error: ${err?.message || err}`);
      }
    }
  
    function stopCamera() {
      if (stream) {
        for (const t of stream.getTracks()) t.stop();
        stream = null;
      }
      sending = false;
      clearOverlay();
      window.VoiceCoach?.stop?.();
    }
  
    function sendLoop() {
      if (!stream || !socket) return;
  
      const targetInterval = 1000 / 15; // cap at 15 FPS
      let lastSend = 0;
  
      const loop = (ts) => {
        if (!stream) return;
  
        const dt = ts - lastSend;
        if (dt >= targetInterval && !sending) {
          try {
            // Draw current frame into hidden canvas
            cctx.drawImage(els.video, 0, 0, captureCanvas.width, captureCanvas.height);
            const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.6); // light compression
  
            // Backpressure: set flag until server ACK
            sending = true;
            socket.emit('process_frame', { image: dataUrl }, () => { sending = false; });
            lastSend = ts;
  
            // FPS UI update once per second
            fpsTick++;
            if (ts - lastFpsTs >= 1000) {
              if (els.fps) els.fps.textContent = String(fpsTick);
              fpsTick = 0;
              lastFpsTs = ts;
            }
          } catch (err) {
            sending = false; // ensure we release backpressure on error
            console.error('[FitMaster] Frame send error:', err);
          }
        }
  
        requestAnimationFrame(loop);
      };
  
      requestAnimationFrame(loop);
    }
  
    /* ===========================
       4) Controls & interactions
       =========================== */
    els.startBtn?.addEventListener('click', startCamera);
    els.stopBtn?.addEventListener('click',  stopCamera);
  
    els.resetBtn?.addEventListener('click', async () => {
      try {
        const r  = await fetch('/reset_counter', { method: 'POST' });
        const js = await r.json();
        log(`Reset → ${JSON.stringify(js)}`);
        if (els.repCount) els.repCount.textContent = '0';
      } catch (err) {
        log(`Reset error: ${err?.message || err}`);
      }
    });
  
    els.exerciseSel?.addEventListener('change', async (e) => {
      const ex = e.target.value;
      if (els.exerciseName) els.exerciseName.textContent = ex.replace('_', ' ');
      try {
        const r  = await fetch('/change_exercise', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ exercise: ex })
        });
        const js = await r.json();
        log(`Exercise → ${ex} (${js.success ? 'ok' : 'error'})`);
      } catch (err) {
        log(`Exercise change error: ${err?.message || err}`);
      }
    });
  
    /* ===============
       5) UI helpers
       =============== */
    function updateUI(data) {
      if (!data) return;
  
      // Feedback text
      if (els.feedback) els.feedback.textContent = data.feedback || '…';
  
      // Score gauge
      const score = Math.max(0, Math.min(100, Number(data.score || 0)));
      if (els.scoreText) els.scoreText.textContent = String(score);
  
      const r = 50; // SVG circle radius
      const circumference = 2 * Math.PI * r;
      const arc = (score / 100) * circumference;
      els.gaugeArc?.setAttribute('stroke-dasharray', `${arc}, ${circumference - arc}`);
  
      // Stats
      if (typeof data.count === 'number' && els.repCount) {
        els.repCount.textContent = String(data.count);
      }
      if (els.tempo) els.tempo.textContent = (typeof data.tempo === 'number') ? `${data.tempo.toFixed(1)} s` : '—';
      if (els.rom)   els.rom.textContent   = (typeof data.rom   === 'number') ? `${Math.round(data.rom)}%` : '—';
      if (els.sym)   els.sym.textContent   = (typeof data.symmetry === 'number') ? `${data.symmetry.toFixed(1)}°` : '—';
  
      // Safety pill (class & label)
      if (els.safetyPill) {
        const sp = els.safetyPill;
        sp.classList.remove('ok', 'warn', 'danger');
        const s = data.safety || 'ok';
        sp.classList.add(s);
        sp.textContent = `Sécurité : ${s.toUpperCase()}`;
      }
    }
  
    function log(msg) {
      if (!els.log) return;
      const li = document.createElement('li');
      li.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
      els.log.prepend(li);
      // Keep list compact
      if (els.log.children.length > 60) els.log.removeChild(els.log.lastChild);
    }
  
    /* =========================================
       6) Skeleton drawing (MediaPipe landmarks)
       ========================================= */
    // Pairs of landmark indices to connect
    const CONN = [
      // left arm
      [11,13],[13,15],
      // right arm
      [12,14],[14,16],
      // shoulders
      [11,12],
      // torso
      [11,23],[12,24],[23,24],
      // left leg
      [23,25],[25,27],[27,31],[27,29],[29,31],
      // right leg
      [24,26],[26,28],[28,32],[28,30],[30,32]
    ];
    const VTH = 0.5; // visibility threshold
  
    function clearOverlay() {
      const ctx = els.overlay?.getContext('2d');
      if (!ctx || !els.overlay) return;
      ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
    }
  
    function drawSkeleton(landmarks) {
      clearOverlay();
      if (!landmarks || !landmarks.length || !els.video?.videoWidth) return;
  
      const rect = getVideoDrawRect(els.video, els.overlay);
      const ctx = els.overlay.getContext('2d');
  
      // Stroke & fill styles
      ctx.lineWidth = 3;
      ctx.lineJoin  = 'round';
      ctx.lineCap   = 'round';
      ctx.strokeStyle = 'rgba(55,255,136,0.9)'; // lines
      ctx.fillStyle   = 'rgba(34,211,238,0.95)'; // points
  
      // Connections (lines)
      ctx.beginPath();
      for (const [a, b] of CONN) {
        const pa = landmarks[a], pb = landmarks[b];
        if (!pa || !pb) continue;
        if ((pa.visibility ?? 1) < VTH || (pb.visibility ?? 1) < VTH) continue;
  
        const x1 = rect.x + pa.x * rect.w;
        const y1 = rect.y + pa.y * rect.h;
        const x2 = rect.x + pb.x * rect.w;
        const y2 = rect.y + pb.y * rect.h;
  
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }
      ctx.stroke();
  
      // Keypoints (circles)
      for (let i = 0; i < landmarks.length; i++) {
        const p = landmarks[i];
        if (!p || (p.visibility ?? 1) < VTH) continue;
  
        const x = rect.x + p.x * rect.w;
        const y = rect.y + p.y * rect.h;
  
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  
    /* ==========================
       7) Voice Coach (Web Speech)
       ========================== */
    (function setupVoiceCoach() {
      const synth = window.speechSynthesis;
      const supported = !!synth;
  
      if (!supported) {
        if (els.voiceToggle) {
          els.voiceToggle.disabled = true;
          els.voiceToggle.title = 'Speech synthesis not supported by this browser';
        }
        return;
      }
  
      let lastText = '';
      let lastTs = 0;
      const COOLDOWN_MS = 1500; // minimal interval between utterances
  
      function speak(text) {
        if (!els.voiceToggle?.checked) return;
        if (!text || typeof text !== 'string') return;
  
        const now = Date.now();
        // Avoid flooding with identical or too-frequent messages
        if (text === lastText && now - lastTs < 6000) return;
        if (now - lastTs < COOLDOWN_MS) return;
  
        try {
          synth.cancel(); // stop any ongoing utterance
          const u = new SpeechSynthesisUtterance(text);
          u.lang   = 'fr-FR';
          u.rate   = parseFloat(els.voiceRate?.value || 1.0);
          u.volume = parseFloat(els.voiceVol?.value || 0.9);
          u.pitch  = 1.0;
          synth.speak(u);
          lastText = text;
          lastTs = now;
        } catch (err) {
          console.error('[FitMaster] TTS error:', err);
        }
      }
  
      // Expose a minimal interface for the main message handler
      window.VoiceCoach = {
        speakFeedback(data) {
          if (!data) return;
          let t = String(data.feedback || '');
          if (data.safety === 'danger') t = 'Attention danger. ' + t;
          else if (data.safety === 'warn') t = 'Attention. ' + t;
          speak(t);
        },
        stop() { synth.cancel(); }
      };
  
      // Stop TTS when user clicks stop
      els.stopBtn?.addEventListener('click', () => synth.cancel());
    })();
  
    /* ===========================================
       8) Exercise preview (merged once, no dupes)
       =========================================== */
    document.addEventListener('DOMContentLoaded', () => {
      const select = document.getElementById('exerciseSelect');
      if (!select) return;
  
      const previews = {
        squats:        document.getElementById('preview-squats'),
        bicep_curls:   document.getElementById('preview-bicep_curls'),
        pushups:       document.getElementById('preview-pushups'),
      };
  
      function updatePreview() {
        // Hide all
        Object.values(previews).forEach(v => { if (v) v.style.display = 'none'; });
        // Show selected
        const choice = select.value;
        if (previews[choice]) previews[choice].style.display = 'block';
      }
  
      updatePreview(); // on load
      select.addEventListener('change', updatePreview);
    });
  
    /* ==================================================
       9) Video metadata → initial overlay calibration
       ================================================== */
    // Ensure overlay is correctly sized once video knows its intrinsic size
    els.video?.addEventListener('loadedmetadata', () => {
      resizeOverlayToContainer(els.overlay);
      // Also update captureCanvas in case startCamera wasn't called yet
      if (els.video.videoWidth && els.video.videoHeight) {
        captureCanvas.width  = els.video.videoWidth;
        captureCanvas.height = els.video.videoHeight;
      }
    });
  
  })();
  