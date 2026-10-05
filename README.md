# Quantum AgriAI System

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-emerald?style=for-the-badge&logo=github)](https://raja20001.github.io/Quantum-AgriAI-for-Smart-Decision-Support/)
[![Vercel](https://img.shields.io/badge/Vercel-Deployed-black?style=for-the-badge&logo=vercel)](https://vercel.com/)
[![Accuracy](https://img.shields.io/badge/VQC%20Accuracy-98.7%25-gold?style=for-the-badge)](https://github.com/Raja20001/Quantum-AgriAI-for-Smart-Decision-Support)

Agricultural decision-making in India and other developing countries faces challenges. Choosing the right crop for the appropriate season is not easy for farmers, especially in areas where soil quality changes quickly and rainfall becomes less predictable each year. Price fluctuations at harvest time add more uncertainty, and traditional decision-support tools have had a hard time addressing this issue effectively. 

This research introduces the **Quantum AgriAI System**. The system combines a Variational Quantum Classifier (VQC), a Quantum Kernel Support Vector Machine (QKernel SVM), and a QAOA-based optimizer, all operating via Qiskit. It also includes traditional ensemble learners like HistGradientBoosting and SVM-RBF, allowing for a direct comparison under the same conditions. Soil and climate data (N, P, K, temperature, humidity, pH, and rainfall) are first summarized into six specific indices, represented in quantum states using angle and amplitude encoding. The VQC achieved **98.7% classification accuracy**, which is 4.7 percentage points higher than the best classical competitor. 

- **Primary Engine:** Quantum Machine Learning (VQC 98.7%, QAOA 96.2%, Quantum Kernel SVM 93.5%).
- **Classical Baselines:** HistGradientBoosting (94.0%), Bagging+DecisionTree (91.5%), SVM-RBF (88.2%), CART (85.3%).
- **Publication Ready:** Pre-configured for instantaneous hosting on **GitHub Pages** and **Vercel**.

---

## 🌐 Live Web Deployment (Publication)

The web dashboard is fully configured for public web deployment with zero dependencies:

- **Live GitHub Pages URL:** `https://raja20001.github.io/Quantum-AgriAI-for-Smart-Decision-Support/`
- **Deployment Guide:** See [**`DEPLOYMENT.md`**](DEPLOYMENT.md) for 1-click step-by-step instructions for GitHub Pages and Vercel.
- **Local Web Preview:**
  ```bash
  python -m http.server 8000
  # Open http://localhost:8000 in your browser
  ```

---

## ⚡ Quick Start (Local Python Pipelines)

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

## 🚀 Mini Project

### 📦 Install Required Libraries

```bash
# Install required Python packages
pip install pandas numpy scikit-learn streamlit matplotlib qiskit qiskit-machine-learning

# Or install all dependencies from requirements.txt
pip install -r requirements.txt
```

### ▶️ Run the Project

```bash
python main_pipeline.py
```
## 📊 Execution Results

![Classical Models](./images/img_1.png)
![Classical Models](./images/img_2.png)
![Classical Models](./images/img_3.png)
![Classical Models](./images/img_4.png)


### 📊 Launch the Streamlit Dashboard

```bash
streamlit run streamlit_dashboard/dashboard.py
```
![streamlit_dashboard](./images/streamlit.png)
### 🌐 Run the Web Dashboard

```bash
python web_dashboard/app.py
```

![web_dashboard](./images/python.png)
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
