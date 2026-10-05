/**
 * Quantum AgriAI — Interactive Quantum Algorithm & Circuit Visualizer
 * VQC (Variational Quantum Classifier), QAOA, and Quantum Kernel SVM
 */

const QuantumLab = (function() {
  let theta = [0.85, 1.25, 0.45, 1.95];
  let qaoaGamma = 1.15;
  let qaoaBeta = 0.75;

  function init() {
    renderCircuit();
    updateStatevector();
    updateQAOA();
    setupEventListeners();
  }

  function setupEventListeners() {
    // VQC Ansatz Sliders
    for (let i = 0; i < 4; i++) {
      const slider = document.getElementById(`vqc-theta-${i}`);
      const valLabel = document.getElementById(`vqc-theta-val-${i}`);
      if (slider) {
        slider.addEventListener('input', (e) => {
          theta[i] = parseFloat(e.target.value);
          if (valLabel) valLabel.textContent = theta[i].toFixed(2);
          updateStatevector();
        });
      }
    }

    // QAOA Sliders
    const gSlider = document.getElementById('qaoa-gamma');
    const gVal = document.getElementById('qaoa-gamma-val');
    if (gSlider) {
      gSlider.addEventListener('input', (e) => {
        qaoaGamma = parseFloat(e.target.value);
        if (gVal) gVal.textContent = qaoaGamma.toFixed(2);
        updateQAOA();
      });
    }

    const bSlider = document.getElementById('qaoa-beta');
    const bVal = document.getElementById('qaoa-beta-val');
    if (bSlider) {
      bSlider.addEventListener('input', (e) => {
        qaoaBeta = parseFloat(e.target.value);
        if (bVal) bVal.textContent = qaoaBeta.toFixed(2);
        updateQAOA();
      });
    }
  }

  /**
   * Render Scalable Vector Graphic (SVG) of the 4-Qubit VQC Circuit
   */
  function renderCircuit() {
    const container = document.getElementById('vqc-circuit-svg-container');
    if (!container) return;

    const svg = `
    <svg viewBox="0 0 740 240" class="circuit-svg" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="gateGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#1e3a5f" />
          <stop offset="100%" stop-color="#0f172a" />
        </linearGradient>
        <linearGradient id="rotGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#f59e0b" />
          <stop offset="100%" stop-color="#b45309" />
        </linearGradient>
        <linearGradient id="hGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#10b981" />
          <stop offset="100%" stop-color="#047857" />
        </linearGradient>
      </defs>

      <!-- Quantum Wire lines -->
      <line x1="70" y1="40" x2="680" y2="40" stroke="#334155" stroke-width="2" />
      <line x1="70" y1="90" x2="680" y2="90" stroke="#334155" stroke-width="2" />
      <line x1="70" y1="140" x2="680" y2="140" stroke="#334155" stroke-width="2" />
      <line x1="70" y1="190" x2="680" y2="190" stroke="#334155" stroke-width="2" />

      <!-- Qubit Labels -->
      <text x="20" y="45" fill="#38bdf8" font-family="JetBrains Mono" font-size="13" font-weight="700">|q₀⟩</text>
      <text x="20" y="95" fill="#38bdf8" font-family="JetBrains Mono" font-size="13" font-weight="700">|q₁⟩</text>
      <text x="20" y="145" fill="#38bdf8" font-family="JetBrains Mono" font-size="13" font-weight="700">|q₂⟩</text>
      <text x="20" y="195" fill="#38bdf8" font-family="JetBrains Mono" font-size="13" font-weight="700">|q₃⟩</text>

      <!-- Layer 1: Hadamard Superposition -->
      <rect x="90" y="25" width="30" height="30" rx="5" fill="url(#hGrad)" stroke="#34d399" stroke-width="1.5" />
      <text x="100" y="45" fill="#fff" font-family="JetBrains Mono" font-size="13" font-weight="700">H</text>

      <rect x="90" y="75" width="30" height="30" rx="5" fill="url(#hGrad)" stroke="#34d399" stroke-width="1.5" />
      <text x="100" y="95" fill="#fff" font-family="JetBrains Mono" font-size="13" font-weight="700">H</text>

      <rect x="90" y="125" width="30" height="30" rx="5" fill="url(#hGrad)" stroke="#34d399" stroke-width="1.5" />
      <text x="100" y="145" fill="#fff" font-family="JetBrains Mono" font-size="13" font-weight="700">H</text>

      <rect x="90" y="175" width="30" height="30" rx="5" fill="url(#hGrad)" stroke="#34d399" stroke-width="1.5" />
      <text x="100" y="195" fill="#fff" font-family="JetBrains Mono" font-size="13" font-weight="700">H</text>

      <!-- Layer 2: ZZ-Feature Map (Input Features Encoding) -->
      <rect x="145" y="24" width="46" height="32" rx="6" fill="url(#gateGrad)" stroke="#38bdf8" stroke-width="1.5" />
      <text x="151" y="44" fill="#38bdf8" font-family="JetBrains Mono" font-size="10" font-weight="700">Rz(x₀)</text>

      <rect x="145" y="74" width="46" height="32" rx="6" fill="url(#gateGrad)" stroke="#38bdf8" stroke-width="1.5" />
      <text x="151" y="94" fill="#38bdf8" font-family="JetBrains Mono" font-size="10" font-weight="700">Rz(x₁)</text>

      <rect x="145" y="124" width="46" height="32" rx="6" fill="url(#gateGrad)" stroke="#38bdf8" stroke-width="1.5" />
      <text x="151" y="144" fill="#38bdf8" font-family="JetBrains Mono" font-size="10" font-weight="700">Rz(x₂)</text>

      <rect x="145" y="174" width="46" height="32" rx="6" fill="url(#gateGrad)" stroke="#38bdf8" stroke-width="1.5" />
      <text x="151" y="194" fill="#38bdf8" font-family="JetBrains Mono" font-size="10" font-weight="700">Rz(x₃)</text>

      <!-- Layer 3: CNOT Entanglement Cascade -->
      <!-- CNOT q0 -> q1 -->
      <line x1="225" y1="40" x2="225" y2="90" stroke="#f43f5e" stroke-width="2" />
      <circle cx="225" cy="40" r="5" fill="#f43f5e" />
      <circle cx="225" cy="90" r="9" fill="none" stroke="#f43f5e" stroke-width="2" />
      <line x1="225" y1="83" x2="225" y2="97" stroke="#f43f5e" stroke-width="2" />
      <line x1="218" y1="90" x2="232" y2="90" stroke="#f43f5e" stroke-width="2" />

      <!-- CNOT q1 -> q2 -->
      <line x1="265" y1="90" x2="265" y2="140" stroke="#f43f5e" stroke-width="2" />
      <circle cx="265" cy="90" r="5" fill="#f43f5e" />
      <circle cx="265" cy="140" r="9" fill="none" stroke="#f43f5e" stroke-width="2" />
      <line x1="265" y1="133" x2="265" y2="147" stroke="#f43f5e" stroke-width="2" />
      <line x1="258" y1="140" x2="272" y2="140" stroke="#f43f5e" stroke-width="2" />

      <!-- CNOT q2 -> q3 -->
      <line x1="305" y1="140" x2="305" y2="190" stroke="#f43f5e" stroke-width="2" />
      <circle cx="305" cy="140" r="5" fill="#f43f5e" />
      <circle cx="305" cy="190" r="9" fill="none" stroke="#f43f5e" stroke-width="2" />
      <line x1="305" y1="183" x2="305" y2="197" stroke="#f43f5e" stroke-width="2" />
      <line x1="298" y1="190" x2="312" y2="190" stroke="#f43f5e" stroke-width="2" />

      <!-- Layer 4: Variational Parameterized Ansatz (RealAmplitudes Ry(theta)) -->
      <rect x="345" y="24" width="56" height="32" rx="6" fill="url(#rotGrad)" stroke="#fbbf24" stroke-width="1.5" />
      <text x="351" y="44" fill="#fff" font-family="JetBrains Mono" font-size="10" font-weight="700">Ry(θ₀)</text>

      <rect x="345" y="74" width="56" height="32" rx="6" fill="url(#rotGrad)" stroke="#fbbf24" stroke-width="1.5" />
      <text x="351" y="94" fill="#fff" font-family="JetBrains Mono" font-size="10" font-weight="700">Ry(θ₁)</text>

      <rect x="345" y="124" width="56" height="32" rx="6" fill="url(#rotGrad)" stroke="#fbbf24" stroke-width="1.5" />
      <text x="351" y="144" fill="#fff" font-family="JetBrains Mono" font-size="10" font-weight="700">Ry(θ₂)</text>

      <rect x="345" y="174" width="56" height="32" rx="6" fill="url(#rotGrad)" stroke="#fbbf24" stroke-width="1.5" />
      <text x="351" y="194" fill="#fff" font-family="JetBrains Mono" font-size="10" font-weight="700">Ry(θ₃)</text>

      <!-- Layer 5: Entanglement Round 2 -->
      <line x1="435" y1="40" x2="435" y2="190" stroke="#f43f5e" stroke-width="2" stroke-dasharray="4,3" />
      <circle cx="435" cy="40" r="5" fill="#f43f5e" />
      <circle cx="435" cy="190" r="9" fill="none" stroke="#f43f5e" stroke-width="2" />

      <!-- Layer 6: Z-Measurement -->
      <g transform="translate(520, 25)">
        <rect x="0" y="0" width="34" height="30" rx="5" fill="#0f172a" stroke="#94a3b8" stroke-width="1.5" />
        <path d="M7 22 A10 10 0 0 1 27 22 M17 22 L24 10" stroke="#94a3b8" stroke-width="1.5" fill="none" />
      </g>
      <g transform="translate(520, 75)">
        <rect x="0" y="0" width="34" height="30" rx="5" fill="#0f172a" stroke="#94a3b8" stroke-width="1.5" />
        <path d="M7 22 A10 10 0 0 1 27 22 M17 22 L24 10" stroke="#94a3b8" stroke-width="1.5" fill="none" />
      </g>
      <g transform="translate(520, 125)">
        <rect x="0" y="0" width="34" height="30" rx="5" fill="#0f172a" stroke="#94a3b8" stroke-width="1.5" />
        <path d="M7 22 A10 10 0 0 1 27 22 M17 22 L24 10" stroke="#94a3b8" stroke-width="1.5" fill="none" />
      </g>
      <g transform="translate(520, 175)">
        <rect x="0" y="0" width="34" height="30" rx="5" fill="#0f172a" stroke="#94a3b8" stroke-width="1.5" />
        <path d="M7 22 A10 10 0 0 1 27 22 M17 22 L24 10" stroke="#94a3b8" stroke-width="1.5" fill="none" />
      </g>

      <!-- Classical Registers output line -->
      <line x1="560" y1="40" x2="660" y2="40" stroke="#94a3b8" stroke-width="2" stroke-dasharray="2,2" />
      <line x1="560" y1="90" x2="660" y2="90" stroke="#94a3b8" stroke-width="2" stroke-dasharray="2,2" />
      <line x1="560" y1="140" x2="660" y2="140" stroke="#94a3b8" stroke-width="2" stroke-dasharray="2,2" />
      <line x1="560" y1="190" x2="660" y2="190" stroke="#94a3b8" stroke-width="2" stroke-dasharray="2,2" />
    </svg>
    `;
    container.innerHTML = svg;
  }

  /**
   * Calculate 4-Qubit Statevector Probabilities based on Ry(theta)
   */
  function updateStatevector() {
    const list = document.getElementById('vqc-statevector-list');
    if (!list) return;

    // Single-qubit amplitudes: cos(theta/2) |0> + sin(theta/2) |1>
    const p0 = theta.map(t => Math.cos(t / 2) ** 2);
    const p1 = theta.map(t => Math.sin(t / 2) ** 2);

    // 16 Computational Basis States
    const states = [];
    for (let i = 0; i < 16; i++) {
      const b3 = (i >> 3) & 1;
      const b2 = (i >> 2) & 1;
      const b1 = (i >> 1) & 1;
      const b0 = i & 1;

      const prob = (b0 ? p1[0] : p0[0]) *
                   (b1 ? p1[1] : p0[1]) *
                   (b2 ? p1[2] : p0[2]) *
                   (b3 ? p1[3] : p0[3]);

      states.push({
        label: `|${b3}${b2}${b1}${b0}⟩`,
        prob: prob
      });
    }

    // Sort to highlight top 6 states
    const topStates = [...states].sort((a, b) => b.prob - a.prob).slice(0, 6);

    list.innerHTML = topStates.map(s => {
      const pct = (s.prob * 100).toFixed(1);
      return `
        <div class="q-state-row">
          <span class="q-state-label">${s.label}</span>
          <div class="q-state-track">
            <div class="q-state-bar" style="width: ${pct}%"></div>
          </div>
          <span class="q-state-pct">${pct}%</span>
        </div>
      `;
    }).join('');
  }

  /**
   * Update QAOA Energy Expectation <H_C>
   */
  function updateQAOA() {
    const eVal = document.getElementById('qaoa-energy-val');
    const cutVal = document.getElementById('qaoa-cut-val');
    if (!eVal || !cutVal) return;

    // Simple Ising expectation simulation: E(g, b) = -sum(sin(2*b) * sin(2*g))
    const energy = -1.4 * Math.sin(2 * qaoaBeta) * Math.sin(qaoaGamma) - 0.8 * Math.cos(qaoaBeta * 1.5);
    const cutRatio = Math.min(0.985, Math.max(0.65, (energy + 2.5) / 3.8));

    eVal.textContent = energy.toFixed(3);
    cutVal.textContent = (cutRatio * 100).toFixed(1) + '%';
  }

  return {
    init: init,
    renderCircuit: renderCircuit
  };
})();

if (typeof window !== 'undefined') {
  window.QuantumLab = QuantumLab;
}
