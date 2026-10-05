/**
 * Quantum AgriAI — Main Application Controller
 * Handles Navigation, Presets, Form Synchronization, Inference Triggers, and Auth State
 */

document.addEventListener('DOMContentLoaded', () => {
  // Navigation Tabs Switching
  const navTabs = document.querySelectorAll('.nav-tab-btn');
  const sections = document.querySelectorAll('.section-panel');

  function switchSection(targetId) {
    sections.forEach(sec => {
      sec.classList.remove('active');
      if (sec.id === `section-${targetId}`) {
        sec.classList.add('active');
      }
    });

    navTabs.forEach(tab => {
      tab.classList.remove('active');
      if (tab.dataset.target === targetId) {
        tab.classList.add('active');
      }
    });

    // Special trigger for canvases or SVGs if switching to quantum or market
    if (targetId === 'quantum' && window.QuantumLab) {
      window.QuantumLab.renderCircuit();
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.target;
      switchSection(target);
    });
  });

  // Expose switchSection globally for in-page buttons
  window.switchSection = switchSection;

  // Form Input Synchronization (Slider <-> Number Input)
  const fields = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph_value', 'rainfall'];

  fields.forEach(field => {
    const range = document.getElementById(`rng-${field}`);
    const num = document.getElementById(`num-${field}`);
    const valLabel = document.getElementById(`val-${field}`);

    if (range && num) {
      range.addEventListener('input', () => {
        num.value = range.value;
        if (valLabel) valLabel.textContent = range.value;
      });

      num.addEventListener('input', () => {
        range.value = num.value;
        if (valLabel) valLabel.textContent = num.value;
      });
    }
  });

  // Preset Configurations
  const presets = {
    wheat: { nitrogen: 85, phosphorus: 52, potassium: 42, temperature: 18.5, humidity: 62, ph_value: 6.7, rainfall: 98 },
    rice: { nitrogen: 80, phosphorus: 48, potassium: 40, temperature: 26.5, humidity: 82, ph_value: 6.4, rainfall: 240 },
    cotton: { nitrogen: 118, phosphorus: 46, potassium: 20, temperature: 26.0, humidity: 78, ph_value: 6.9, rainfall: 85 },
    turmeric: { nitrogen: 90, phosphorus: 55, potassium: 65, temperature: 27.5, humidity: 85, ph_value: 6.2, rainfall: 220 },
    soybean: { nitrogen: 65, phosphorus: 42, potassium: 38, temperature: 24.0, humidity: 68, ph_value: 6.6, rainfall: 95 }
  };

  window.loadPreset = function(presetKey) {
    const p = presets[presetKey];
    if (!p) return;

    for (const [k, v] of Object.entries(p)) {
      const range = document.getElementById(`rng-${k}`);
      const num = document.getElementById(`num-${k}`);
      const valLabel = document.getElementById(`val-${k}`);
      if (range) range.value = v;
      if (num) num.value = v;
      if (valLabel) valLabel.textContent = v;
    }

    showToast(`Loaded "${presetKey.toUpperCase()}" agricultural preset conditions.`);
    triggerPrediction();
  };

  // Prediction Trigger & UI Update
  const predictBtn = document.getElementById('btn-run-prediction');
  if (predictBtn) {
    predictBtn.addEventListener('click', () => {
      triggerPrediction();
    });
  }

  function getFormValues() {
    return {
      nitrogen: parseFloat(document.getElementById('num-nitrogen').value) || 80,
      phosphorus: parseFloat(document.getElementById('num-phosphorus').value) || 40,
      potassium: parseFloat(document.getElementById('num-potassium').value) || 40,
      temperature: parseFloat(document.getElementById('num-temperature').value) || 25,
      humidity: parseFloat(document.getElementById('num-humidity').value) || 72,
      ph_value: parseFloat(document.getElementById('num-ph_value').value) || 6.5,
      rainfall: parseFloat(document.getElementById('num-rainfall').value) || 700
    };
  }

  function triggerPrediction() {
    if (!window.QuantumAgriEngine) return;

    const btn = document.getElementById('btn-run-prediction');
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '⚛️ Evaluating Quantum Amplitudes…';
    }

    setTimeout(() => {
      const inputs = getFormValues();
      const results = window.QuantumAgriEngine.predict(inputs);
      renderPredictionResults(results, inputs);

      if (btn) {
        btn.disabled = false;
        btn.innerHTML = '⚛️ Run Quantum-Classical Inference';
      }
    }, 280);
  }

  window.triggerPrediction = triggerPrediction;

  function renderPredictionResults(res, inputs) {
    const winner = res.winner;
    const top3 = res.top3;
    const models = res.modelPredictions;
    const diags = res.diagnostics;
    const planting = res.planting;

    // 1. Winner Hero Card
    const winEmoji = document.getElementById('res-winner-emoji');
    const winName = document.getElementById('res-winner-name');
    const winConf = document.getElementById('res-winner-conf');
    const winSeason = document.getElementById('res-winner-season');
    const winPrice = document.getElementById('res-winner-price');
    const winDuration = document.getElementById('res-winner-duration');

    if (winEmoji) winEmoji.textContent = winner.emoji || '🌾';
    if (winName) winName.textContent = winner.crop;
    if (winConf) winConf.textContent = `Quantum VQC Confidence: ${(winner.quantumConfidence * 100).toFixed(1)}%`;
    if (winSeason) winSeason.textContent = planting.season;
    if (winPrice) winPrice.textContent = `₹${winner.price_inr_q.toLocaleString()}/q`;
    if (winDuration) winDuration.textContent = `${planting.duration} Days`;

    // 2. Podium Cards (Top 3)
    const podiumContainer = document.getElementById('res-podium-grid');
    if (podiumContainer) {
      const ranks = ['🥇 BEST RECOMMENDED', '🥈 2ND ALTERNATIVE', '🥉 3RD ALTERNATIVE'];
      podiumContainer.innerHTML = top3.map((c, i) => `
        <div class="podium-card ${i === 0 ? 'rank-1' : ''}">
          <div class="podium-rank">${ranks[i]}</div>
          <div class="podium-emoji">${c.emoji}</div>
          <div class="podium-crop">${c.crop}</div>
          <div class="podium-score">${(c.blendedScore * 100).toFixed(1)}% Match</div>
          <div class="prog-track">
            <div class="prog-fill" style="width: ${(c.blendedScore * 100).toFixed(0)}%"></div>
          </div>
          <div style="font-size:0.75rem;color:var(--text-muted);margin-top:8px">
            ML: ${(c.mlConfidence * 100).toFixed(1)}% &nbsp;·&nbsp; Agronomic: ${c.agronomicScore}%
          </div>
        </div>
      `).join('');
    }

    // 3. Multi-Model Comparison Table
    const tableBody = document.getElementById('res-models-table-body');
    if (tableBody) {
      tableBody.innerHTML = models.map(m => `
        <tr class="${m.isQ ? 'quantum-row' : ''}">
          <td>${m.name}</td>
          <td style="color:#ffffff;font-weight:600">${m.crop}</td>
          <td style="font-family:var(--font-mono);font-weight:700;color:${m.isQ ? 'var(--quantum-gold)' : 'var(--emerald)'}">${m.accuracy}</td>
          <td><span class="badge-chip" style="font-size:0.7rem">${m.type}</span></td>
          <td style="font-family:var(--font-mono);color:${m.rank === 1 ? '#34d399' : 'var(--text-muted)'}">Rank #${m.rank}</td>
        </tr>
      `).join('');
    }

    // 4. Environmental Diagnostics Chips
    const diagContainer = document.getElementById('res-diagnostics-grid');
    if (diagContainer) {
      diagContainer.innerHTML = diags.map(d => `
        <div style="background:var(--bg-input);border:1px solid ${d.ok ? 'rgba(16,185,129,0.3)' : 'rgba(245,158,11,0.3)'};border-radius:var(--radius-sm);padding:10px 14px;display:flex;justify-content:space-between;align-items:center">
          <div>
            <div style="font-size:0.72rem;color:var(--text-dim);text-transform:uppercase">${d.param}</div>
            <div style="font-family:var(--font-mono);font-size:0.95rem;font-weight:700;color:#ffffff">${d.val}</div>
          </div>
          <span style="font-size:0.72rem;font-weight:700;padding:2px 8px;border-radius:4px;background:${d.ok ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)'};color:${d.ok ? '#34d399' : '#fbbf24'}">
            ${d.status}
          </span>
        </div>
      `).join('');
    }

    // 5. Planting Calendar
    const calSow = document.getElementById('res-cal-sow');
    const calHarvest = document.getElementById('res-cal-harvest');
    const calSeason = document.getElementById('res-cal-season');
    if (calSow) calSow.textContent = planting.sow;
    if (calHarvest) calHarvest.textContent = planting.harvest;
    if (calSeason) calSeason.textContent = planting.season;

    // Show result wrapper
    const resWrap = document.getElementById('prediction-results-wrap');
    if (resWrap) {
      resWrap.style.display = 'block';
      resWrap.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  // Toast Notification Helper
  function showToast(msg) {
    let toast = document.getElementById('app-toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'app-toast';
      toast.className = 'toast-notice';
      document.body.appendChild(toast);
    }
    toast.innerHTML = `<span>⚛️</span> <span>${msg}</span>`;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 3200);
  }
  window.showToast = showToast;

  // Demo Auth Modal Logic
  const loginModal = document.getElementById('login-modal');
  const userBtn = document.getElementById('nav-user-btn');
  const closeModalBtn = document.getElementById('modal-close');

  if (userBtn && loginModal) {
    userBtn.addEventListener('click', () => {
      loginModal.classList.add('active');
    });
  }

  if (closeModalBtn && loginModal) {
    closeModalBtn.addEventListener('click', () => {
      loginModal.classList.remove('active');
    });
  }

  // Initial prediction on load
  setTimeout(() => {
    triggerPrediction();
    if (window.QuantumLab) window.QuantumLab.init();
  }, 400);
});
