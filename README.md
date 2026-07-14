# Quantum AgriAI System

A quantum-classical hybrid precision crop recommendation platform.
**Quantum algorithms (VQC, QAOA, Quantum Kernel SVM) are the primary engine.**
Classical models (CatBoost 97-98%, ExtraTrees 96-97%) serve as benchmarks.

---

## Quick Start

### 1. Install Dependencies

```bash
# Core dependencies (required)
pip install streamlit pandas numpy scikit-learn plotly scipy pyyaml joblib xgboost lightgbm catboost

# Quantum (optional — enables live circuits; simulation mode works without it)
pip install qiskit qiskit-aer qiskit-machine-learning qiskit-algorithms
```

### 2. Run the Streamlit Dashboard

```bash
# From the project root directory (quantum_agri_fixed/):
streamlit run streamlit_dashboard/dashboard.py
```

Then open http://localhost:8501 in your browser.

### 3. Run the Full Pipeline (CLI)

```bash
python main_pipeline.py
```

---

## Project Structure

```
quantum_agri_fixed/
├── streamlit_dashboard/
│   └── dashboard.py          ← Main Streamlit app (6 pages)
├── models/
│   ├── classical_models.py   ← CatBoost, ExtraTrees, XGBoost, RF, etc.
│   └── quantum_model.py      ← VQC, QAOA, Quantum Kernel SVM (Qiskit fix)
├── data_pipeline/
│   ├── data_loader.py
│   ├── data_cleaning.py
│   └── feature_engineering.py
├── forecasting/
│   └── market_forecasting.py
├── decision_support/
│   └── recommendation_system.py
├── datasets/
│   ├── crop_recommendation_dataset.csv
│   └── india_agriculture_dataset.csv
├── config/
│   └── config.yaml
├── requirements.txt
└── main_pipeline.py
```

---

## Bugs Fixed in This Release

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `forecasting/market_forecasting.py` | `alert_threshold=80.0` triggered alerts on ALL prices (all market prices are 2000+) | Changed default to `3000.0` |
| 2 | `streamlit_dashboard/dashboard.py` | `if pred_btn or True:` always ran prediction block on every page load (before clicking button) | Removed `or True` |
| 3 | `streamlit_dashboard/dashboard.py` | No feedback when Prediction page loads without clicking button | Added an info box prompt |
| 4 | `streamlit_dashboard/dashboard.py` | No guard when `best_m is None` (if no ML libraries installed) | Added `if best_m is None` error guard |
| 5 | `streamlit_dashboard/dashboard.py` | Unused `loader = DataLoader(".")` created in India Explorer page | Removed dead code |
| 6 | `models/quantum_model.py` | Qiskit gradient warning from missing `pass_manager` | `pass_manager = generate_preset_pass_manager(optimization_level=0)` |
| 7 | `models/quantum_model.py` | Deprecated `ZZFeatureMap` class API | Replaced with `zz_feature_map()` / `real_amplitudes()` function API |

---

## Qiskit Warning Fix (Root Cause)

**Original warning:**
```
WARNING qiskit_machine_learning.neural_networks.sampler_qnn —
No gradient function provided, creating a gradient function.
If your Sampler requires transpilation, please provide a pass manager.
```

**Fix (quantum_model.py):**
```python
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
pm = generate_preset_pass_manager(optimization_level=0)
vqc = VQC(..., pass_manager=pm)   # ← warning suppressed at root
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Overview, workflow, quantum concepts, dataset stats |
| 🔮 Prediction | Live crop recommendation — adjust sliders, click **Get Prediction** |
| ⚛️ Quantum | VQC · QAOA · Quantum Kernel SVM — circuits, benchmarks |
| 🇮🇳 India Explorer | State & district analytics (37 states, 310 districts) |
| 📈 Market | Price trends, Bollinger Bands, demand forecasting |
| 👤 About | Author, tech stack, results summary |
