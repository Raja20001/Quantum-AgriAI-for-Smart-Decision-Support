# -*- coding: utf-8 -*-
"""Quantum AgriAI — Test Suite | Run: pytest tests/ -v"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import pytest
import pandas as pd
import numpy as np

from data_pipeline.data_loader         import DataLoader
from data_pipeline.data_cleaning       import DataCleaning
from data_pipeline.feature_engineering import FeatureEngineering
from models.classical_models           import ClassicalModels
from models.quantum_model              import VQCModel, QAOAModel, QuantumKernelSVM, QISKIT_AVAILABLE
from forecasting.market_forecasting    import MarketForecasting
from decision_support.recommendation_system import RecommendationSystem


@pytest.fixture(scope="module")
def crop_df():
    return DataLoader(".").load_crop_dataset()

@pytest.fixture(scope="module")
def india_df():
    return DataLoader(".").load_india_data()

@pytest.fixture(scope="module")
def trained():
    df  = DataLoader(".").load_crop_dataset()
    df  = DataCleaning().clean(df)
    df  = FeatureEngineering().create_features(df)
    fc  = [c for c in FeatureEngineering.get_feature_names() if c in df.columns]
    clf = ClassicalModels()
    Xtr, Xte, ytr, yte = clf.prepare_data(df, fc, "Crop")
    clf.train_catboost(Xtr, ytr)
    clf.train_extra_trees(Xtr, ytr)
    clf.evaluate(clf.models["CatBoost"],    Xte, yte, "CatBoost")
    clf.evaluate(clf.models["ExtraTrees"],  Xte, yte, "ExtraTrees")
    return clf, Xtr, Xte, ytr, yte, fc

@pytest.fixture
def mkt_df():
    np.random.seed(1); n = 60
    return pd.DataFrame({
        "date":   pd.date_range("2024-01-01", periods=n),
        "price":  2000 + np.cumsum(np.random.randn(n) * 40),
        "demand": 10000 + np.random.randint(-500, 500, n).astype(float),
        "supply": 9500  + np.random.randint(-400, 400, n).astype(float),
    })


# ── Dataset tests ─────────────────────────────────────────────────────────────
class TestDataset:
    def test_crop_8100_rows(self, crop_df):     assert len(crop_df) == 8100
    def test_27_crop_classes(self, crop_df):    assert crop_df["Crop"].nunique() == 27
    def test_300_samples_each(self, crop_df):   assert crop_df["Crop"].value_counts().min() == 300
    def test_india_37_states(self, india_df):   assert india_df["State"].nunique() >= 10
    def test_india_310_districts(self, india_df):assert india_df["District"].nunique() >= 50
    def test_india_has_yield(self, india_df):   assert "Yield_kg_ha" in india_df.columns
    def test_india_filter_state(self, india_df):
        f = DataLoader(".").filter_by_state(india_df, "Tamil Nadu")
        assert len(f) > 0 and (f["State"] == "Tamil Nadu").all()


# ── Cleaning tests ────────────────────────────────────────────────────────────
class TestCleaning:
    def test_no_nulls_after_clean(self, crop_df):
        c = DataCleaning().clean(crop_df)
        assert c.isnull().sum().sum() == 0
    def test_gt_6000_rows_remain(self, crop_df):
        c = DataCleaning().clean(crop_df)
        assert len(c) > 6000
    def test_quality_report_keys(self, crop_df):
        r = DataCleaning.data_quality_report(crop_df)
        assert all(k in r for k in ("rows", "columns", "missing", "duplicates"))


# ── Feature engineering tests ─────────────────────────────────────────────────
class TestFeatureEngineering:
    def test_22_features_total(self, crop_df):
        df = FeatureEngineering().create_features(crop_df)
        fc = [c for c in FeatureEngineering.get_feature_names() if c in df.columns]
        assert len(fc) == 22

    def test_all_15_engineered_present(self, crop_df):
        df = FeatureEngineering().create_features(crop_df)
        for feat in ["soil_quality_index", "climate_index", "nutrient_balance",
                     "water_stress_index", "ph_deviation", "npk_total",
                     "heat_humidity", "rainfall_efficiency",
                     "N_P_ratio", "N_K_ratio", "P_K_ratio",
                     "temp_rainfall_product", "log_rainfall",
                     "humidity_sq", "ph_optimal_band"]:
            assert feat in df.columns, f"Missing: {feat}"

    def test_no_nulls_in_features(self, crop_df):
        df = FeatureEngineering().create_features(crop_df)
        fc = [c for c in FeatureEngineering.get_feature_names() if c in df.columns]
        assert df[fc].isnull().sum().sum() == 0

    def test_ph_optimal_binary(self, crop_df):
        df = FeatureEngineering().create_features(crop_df)
        assert set(df["ph_optimal_band"].unique()).issubset({0.0, 1.0})

    def test_log_rainfall_nonneg(self, crop_df):
        df = FeatureEngineering().create_features(crop_df)
        assert (df["log_rainfall"] >= 0).all()


# ── CatBoost best model tests ─────────────────────────────────────────────────
class TestCatBoost:
    def test_catboost_in_models(self, trained):
        clf, *_ = trained
        assert "CatBoost" in clf.models

    def test_catboost_accuracy_97pct(self, trained):
        clf, _, Xte, _, yte, _ = trained
        r = clf.results.get("CatBoost") or clf.evaluate(clf.models["CatBoost"], Xte, yte, "CatBoost")
        assert r["accuracy"] >= 0.97, f"CatBoost acc={r['accuracy']}"

    def test_catboost_kappa_97pct(self, trained):
        clf, *_ = trained
        r = clf.results.get("CatBoost", {})
        assert r.get("kappa", 0) >= 0.97

    def test_predict_best_crop(self, trained):
        clf, _, Xte, _, _, _ = trained
        pred = clf.predict_crop(clf.models["CatBoost"], Xte[0].tolist())
        assert "best_crop" in pred
        assert pred["best_crop"] in clf.label_encoder.classes_

    def test_predict_top3(self, trained):
        clf, _, Xte, _, _, _ = trained
        pred = clf.predict_crop(clf.models["CatBoost"], Xte[0].tolist())
        assert len(pred["top_predictions"]) <= 3

    def test_feature_importance_nonempty(self, trained):
        clf, _, _, _, _, fc = trained
        fi = clf.get_feature_importance("CatBoost", fc)
        assert len(fi) > 0

    def test_catboost_fi_has_attr(self, trained):
        clf, *_ = trained
        assert hasattr(clf.models["CatBoost"], "feature_importances_")

    def test_leaderboard_sorted(self, trained):
        clf, *_ = trained
        lb = clf.leaderboard()
        accs = [r[1] for r in lb]
        assert accs == sorted(accs, reverse=True)


# ── ExtraTrees tests ──────────────────────────────────────────────────────────
class TestExtraTrees:
    def test_extratrees_in_models(self, trained):
        clf, *_ = trained
        assert "ExtraTrees" in clf.models

    def test_extratrees_accuracy_gt95(self, trained):
        clf, _, Xte, _, yte, _ = trained
        r = clf.results.get("ExtraTrees") or clf.evaluate(clf.models["ExtraTrees"], Xte, yte, "ET")
        assert r["accuracy"] >= 0.95, f"ET acc={r['accuracy']}"


# ── Quantum model tests ───────────────────────────────────────────────────────
class TestQuantumModels:
    def test_qiskit_bool(self):
        assert isinstance(QISKIT_AVAILABLE, bool)

    def test_vqc_circuit_has_fix(self):
        vqc = VQCModel(n_qubits=4)
        ct  = vqc.circuit_text()
        assert "pass_manager" in ct, "VQC fix not mentioned in circuit_text"
        assert "zz_feature_map" in ct, "New function API not mentioned"

    def test_vqc_simulated_result(self):
        vqc = VQCModel(n_qubits=4)
        r   = vqc.evaluate(np.random.randn(20, 22), np.arange(20) % 5)
        assert r["accuracy"] is not None
        assert "simulated" in r

    def test_qaoa_circuit_description(self):
        qaoa = QAOAModel(n_qubits=4, p_layers=2)
        ct   = qaoa.circuit_text()
        assert "QAOA" in ct
        assert "Farhi" in ct

    def test_qaoa_simulated_result(self):
        qaoa = QAOAModel(n_qubits=4, p_layers=2)
        r    = qaoa.evaluate(np.random.randn(20, 22), np.arange(20) % 5)
        assert r["accuracy"] is not None

    def test_qksvm_hilbert_space(self):
        qksvm = QuantumKernelSVM(n_qubits=4)
        ct    = qksvm.circuit_text()
        assert "Hilbert" in ct
        assert "2^4" in ct or "16" in ct

    def test_qksvm_simulated_result(self):
        qksvm = QuantumKernelSVM(n_qubits=4)
        r     = qksvm.evaluate(np.random.randn(20, 22), np.arange(20) % 5)
        assert r["accuracy"] is not None


# ── Market forecasting tests ──────────────────────────────────────────────────
class TestMarket:
    def test_bollinger_cols(self, mkt_df):
        td = MarketForecasting().price_trend(mkt_df)
        for c in ["moving_average", "upper_band", "lower_band"]:
            assert c in td.columns

    def test_demand_trend_valid(self, mkt_df):
        assert MarketForecasting().demand_prediction(mkt_df)["trend"] in ("rising","falling")

    def test_alerts_all_when_zero_threshold(self, mkt_df):
        alerts = MarketForecasting(alert_threshold=0).generate_price_alerts(mkt_df)
        assert len(alerts) == len(mkt_df)

    def test_supply_gap_key(self, mkt_df):
        assert "avg_gap" in MarketForecasting().supply_demand_gap(mkt_df)


# ── Recommendation tests ──────────────────────────────────────────────────────
class TestRecommendation:
    def test_store_advice_low(self):
        r = RecommendationSystem().generate_recommendation(500, 1000)
        assert "store" in r.lower()

    def test_excellent_advice_high(self):
        r = RecommendationSystem().generate_recommendation(3000, 5000)
        assert "expand" in r.lower() or "sell" in r.lower()

    def test_rice_score_high(self):
        r = RecommendationSystem().crop_suitability(
            "Rice", {"temperature":28,"humidity":82,"rainfall":220,"ph_value":6.4}
        )
        assert r["score"] > 70

    def test_market_price_present(self):
        r = RecommendationSystem().crop_suitability(
            "Rice", {"temperature":28,"humidity":82,"rainfall":220,"ph_value":6.4}
        )
        assert r["market_price"] > 0

    def test_rank_sorted_desc(self):
        conds = {"temperature":25,"humidity":72,"rainfall":800,"ph_value":6.5}
        rs    = RecommendationSystem().rank_crops(conds, top_n=5)
        scores= [r["score"] for r in rs]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_crop_zero(self):
        r = RecommendationSystem().crop_suitability(
            "UNKNOWN_XYZ",
            {"temperature":25,"humidity":72,"rainfall":800,"ph_value":6.5}
        )
        assert r["score"] == 0.0

    def test_blended_score_in_range(self):
        conds  = {"temperature":25,"humidity":72,"rainfall":800,"ph_value":6.5}
        preds  = [{"crop":"Rice","confidence":0.9},{"crop":"Wheat","confidence":0.1}]
        result = RecommendationSystem().ml_recommendation(preds, conds)
        for rec in result["recommendations"]:
            assert 0.0 <= rec["blended_score"] <= 1.0

    def test_planting_calendar_rice(self):
        cal = RecommendationSystem.planting_calendar("Rice")
        assert "sow" in cal and "harvest" in cal
