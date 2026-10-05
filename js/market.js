/**
 * Quantum AgriAI — Market Intelligence & Price Forecasting
 * Bollinger Bands (+/- 1.96 sigma), MSP Baselines, and Automated Selling Recommendations
 */

const MarketForecaster = (function() {
  let marketData = {};

  async function init() {
    try {
      const res = await fetch('data/market.json');
      if (res.ok) {
        marketData = await res.json();
      }
    } catch (e) {
      console.warn('Could not load market.json:', e);
    }
    populateCropSelector();
  }

  function populateCropSelector() {
    const sel = document.getElementById('market-crop-select');
    if (!sel) return;

    sel.innerHTML = '';
    const crops = Object.keys(marketData).sort();

    crops.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c;
      opt.textContent = `${marketData[c].emoji || '🌾'} ${c}`;
      sel.appendChild(opt);
    });

    sel.addEventListener('change', (e) => {
      renderMarket(e.target.value);
    });

    if (crops.length > 0) {
      sel.value = crops.includes("Wheat") ? "Wheat" : (crops.includes("Rice") ? "Rice" : crops[0]);
      renderMarket(sel.value);
    }
  }

  function renderMarket(cropName) {
    if (!cropName || !marketData[cropName]) return;
    const data = marketData[cropName];

    // Update Metric Cards
    const avgEl = document.getElementById('market-avg-price');
    const mspEl = document.getElementById('market-msp-price');
    const volEl = document.getElementById('market-volatility');
    const sigEl = document.getElementById('market-signal-badge');

    if (avgEl) avgEl.textContent = `₹${data.avg_price.toLocaleString()}`;
    if (mspEl) mspEl.textContent = `₹${data.msp.toLocaleString()}`;
    if (volEl) volEl.textContent = `${data.volatility_pct}%`;

    if (sigEl) {
      sigEl.textContent = data.signal;
      sigEl.className = 'winner-tag';
      if (data.signal.includes("SELL")) {
        sigEl.style.background = "rgba(16, 185, 129, 0.15)";
        sigEl.style.borderColor = "rgba(16, 185, 129, 0.4)";
        sigEl.style.color = "#34d399";
      } else if (data.signal.includes("STORE")) {
        sigEl.style.background = "rgba(245, 158, 11, 0.15)";
        sigEl.style.borderColor = "rgba(245, 158, 11, 0.4)";
        sigEl.style.color = "#fbbf24";
      } else {
        sigEl.style.background = "rgba(59, 130, 246, 0.15)";
        sigEl.style.borderColor = "rgba(59, 130, 246, 0.4)";
        sigEl.style.color = "#60a5fa";
      }
    }

    renderChart(data);
  }

  function renderChart(data) {
    const chartContainer = document.getElementById('market-svg-chart');
    if (!chartContainer) return;

    const trend = data.trend_12m || [];
    const upper = data.upper_band;
    const lower = data.lower_band;
    const msp = data.msp;

    const minVal = Math.min(...trend, lower, msp) * 0.92;
    const maxVal = Math.max(...trend, upper, msp) * 1.08;
    const range = maxVal - minVal || 1;

    const width = 760;
    const height = 280;
    const padding = { top: 30, right: 40, bottom: 40, left: 60 };

    const plotW = width - padding.left - padding.right;
    const plotH = height - padding.top - padding.bottom;

    const getY = (val) => padding.top + plotH - ((val - minVal) / range) * plotH;
    const getX = (idx) => padding.left + (idx / (trend.length - 1)) * plotW;

    // Generate Path for 12-Month Trend Line
    let pathD = '';
    trend.forEach((p, i) => {
      const x = getX(i);
      const y = getY(p);
      pathD += (i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
    });

    const upperY = getY(upper);
    const lowerY = getY(lower);
    const mspY = getY(msp);

    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

    const svg = `
    <svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: auto; font-family: 'DM Sans', sans-serif;">
      <defs>
        <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#10b981" stop-opacity="0.25"/>
          <stop offset="100%" stop-color="#10b981" stop-opacity="0.0"/>
        </linearGradient>
      </defs>

      <!-- Background Grid -->
      <line x1="${padding.left}" y1="${padding.top}" x2="${width - padding.right}" y2="${padding.top}" stroke="#1e293b" stroke-width="1" />
      <line x1="${padding.left}" y1="${padding.top + plotH / 2}" x2="${width - padding.right}" y2="${padding.top + plotH / 2}" stroke="#1e293b" stroke-width="1" />
      <line x1="${padding.left}" y1="${padding.top + plotH}" x2="${width - padding.right}" y2="${padding.top + plotH}" stroke="#1e293b" stroke-width="1" />

      <!-- Upper Bollinger Band -->
      <line x1="${padding.left}" y1="${upperY}" x2="${width - padding.right}" y2="${upperY}" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4,4" />
      <text x="${width - padding.right + 6}" y="${upperY + 4}" fill="#f59e0b" font-size="10" font-family="JetBrains Mono">+2σ Upper</text>

      <!-- Lower Bollinger Band -->
      <line x1="${padding.left}" y1="${lowerY}" x2="${width - padding.right}" y2="${lowerY}" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4,4" />
      <text x="${width - padding.right + 6}" y="${lowerY + 4}" fill="#f59e0b" font-size="10" font-family="JetBrains Mono">-2σ Lower</text>

      <!-- MSP Baseline Line -->
      <line x1="${padding.left}" y1="${mspY}" x2="${width - padding.right}" y2="${mspY}" stroke="#38bdf8" stroke-width="2" />
      <text x="${padding.left + 8}" y="${mspY - 6}" fill="#38bdf8" font-size="10" font-family="JetBrains Mono" font-weight="700">Govt MSP Baseline (₹${msp})</text>

      <!-- Area Under Price Curve -->
      <path d="${pathD} L ${getX(trend.length - 1)} ${padding.top + plotH} L ${getX(0)} ${padding.top + plotH} Z" fill="url(#areaGrad)" />

      <!-- Price Trend Curve -->
      <path d="${pathD}" fill="none" stroke="#10b981" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />

      <!-- Trend Dots & Month Labels -->
      ${trend.map((p, i) => {
        const x = getX(i);
        const y = getY(p);
        return `
          <circle cx="${x}" cy="${y}" r="4" fill="#34d399" stroke="#064e3b" stroke-width="2" />
          <text x="${x}" y="${padding.top + plotH + 20}" fill="#64748b" font-size="10" text-anchor="middle">${months[i]}</text>
        `;
      }).join('')}

      <!-- Y-Axis Values -->
      <text x="${padding.left - 10}" y="${padding.top + 4}" fill="#94a3b8" font-size="10" font-family="JetBrains Mono" text-anchor="end">₹${Math.round(maxVal)}</text>
      <text x="${padding.left - 10}" y="${padding.top + plotH / 2 + 4}" fill="#94a3b8" font-size="10" font-family="JetBrains Mono" text-anchor="end">₹${Math.round((maxVal + minVal) / 2)}</text>
      <text x="${padding.left - 10}" y="${padding.top + plotH + 4}" fill="#94a3b8" font-size="10" font-family="JetBrains Mono" text-anchor="end">₹${Math.round(minVal)}</text>
    </svg>
    `;

    chartContainer.innerHTML = svg;
  }

  return {
    init: init,
    renderMarket: renderMarket
  };
})();

if (typeof window !== 'undefined') {
  window.MarketForecaster = MarketForecaster;
  document.addEventListener('DOMContentLoaded', () => {
    MarketForecaster.init();
  });
}
