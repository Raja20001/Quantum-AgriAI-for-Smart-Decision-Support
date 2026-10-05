/**
 * Quantum AgriAI — India Agriculture Explorer
 * Geospatial crop distribution, yields, and prices across 37 States/UTs and 310 Districts
 */

const IndiaExplorer = (function() {
  let indiaData = {};

  async function init() {
    try {
      const res = await fetch('data/india_agri.json');
      if (res.ok) {
        indiaData = await res.json();
      }
    } catch (e) {
      console.warn('Could not load india_agri.json:', e);
    }
    populateStates();
  }

  function populateStates() {
    const stateSelect = document.getElementById('explorer-state-select');
    if (!stateSelect) return;

    stateSelect.innerHTML = '<option value="">— Select an Indian State / UT —</option>';
    const states = Object.keys(indiaData).sort();

    states.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s;
      opt.textContent = `${s} (${indiaData[s].district_count} Districts)`;
      stateSelect.appendChild(opt);
    });

    stateSelect.addEventListener('change', (e) => {
      renderState(e.target.value);
    });

    // Default to a rich agricultural state if available
    const defaultState = states.includes("Punjab") ? "Punjab" : (states.includes("Tamil Nadu") ? "Tamil Nadu" : states[0]);
    if (defaultState) {
      stateSelect.value = defaultState;
      renderState(defaultState);
    }
  }

  function renderState(stateName) {
    const summaryCard = document.getElementById('state-summary-card');
    const cropGrid = document.getElementById('state-crop-grid');
    const districtSelect = document.getElementById('explorer-district-select');

    if (!stateName || !indiaData[stateName]) {
      if (summaryCard) summaryCard.innerHTML = '<div style="color:var(--text-dim);padding:20px;text-align:center">Select a state above to view agricultural records.</div>';
      if (cropGrid) cropGrid.innerHTML = '';
      return;
    }

    const data = indiaData[stateName];

    // Populate districts
    if (districtSelect) {
      districtSelect.innerHTML = '<option value="">All Districts (' + data.districts.length + ')</option>';
      data.districts.forEach(d => {
        const opt = document.createElement('option');
        opt.value = d;
        opt.textContent = d;
        districtSelect.appendChild(opt);
      });
      districtSelect.style.display = 'block';
    }

    // Render Summary Card
    if (summaryCard) {
      summaryCard.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;margin-bottom:1.5rem">
          <div>
            <h3 style="font-size:1.6rem;color:#ffffff;display:flex;align-items:center;gap:8px">
              🇮🇳 ${stateName}
            </h3>
            <div style="font-size:0.82rem;color:var(--text-muted);margin-top:2px">
              ${data.records.toLocaleString()} Verified Records &nbsp;·&nbsp; ${data.districts.length} Agricultural Districts
            </div>
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap">
            <span class="badge-chip">Seasons: <strong>${data.seasons.join(', ') || 'Kharif, Rabi'}</strong></span>
            <span class="badge-chip">Top Crop: <strong>${Object.keys(data.top_crops)[0] || 'Rice'}</strong></span>
          </div>
        </div>
      `;
    }

    // Render Crop Stats Grid
    if (cropGrid) {
      const topCrops = Object.entries(data.top_crops);
      if (topCrops.length === 0) {
        cropGrid.innerHTML = '<div style="color:var(--text-dim)">No crop records found for this state.</div>';
        return;
      }

      cropGrid.innerHTML = topCrops.map(([crop, count]) => {
        const yld = data.avg_yield[crop] || '—';
        const prc = data.avg_price[crop] || '—';
        return `
          <div class="crop-stat-card">
            <div class="crop-stat-header">
              <span class="crop-stat-name">${crop}</span>
              <span style="font-size:0.75rem;color:var(--emerald);font-weight:600">${count} records</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-top:8px">
              <div>
                <div style="font-size:0.68rem;color:var(--text-dim);text-transform:uppercase">Avg Yield</div>
                <div class="crop-stat-yield">${yld} <span style="font-size:0.75rem;color:var(--text-muted)">kg/ha</span></div>
              </div>
              <div style="text-align:right">
                <div style="font-size:0.68rem;color:var(--text-dim);text-transform:uppercase">Mandi Price</div>
                <div class="crop-stat-price">₹${prc} <span style="font-size:0.72rem;color:var(--text-muted)">/q</span></div>
              </div>
            </div>
          </div>
        `;
      }).join('');
    }
  }

  return {
    init: init,
    renderState: renderState
  };
})();

if (typeof window !== 'undefined') {
  window.IndiaExplorer = IndiaExplorer;
  document.addEventListener('DOMContentLoaded', () => {
    IndiaExplorer.init();
  });
}
