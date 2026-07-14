# -*- coding: utf-8 -*-
"""
Feature Engineering — Quantum AgriAI v7
=========================================
Workflow:  Dataset → Data Space → [Feature Vectors] → Model

Produces 29 non-overlapping agronomic features from 11 base columns.
All features are stateless (pure arithmetic) — no fitted statistics.
Zero data-leakage: StandardScaler is fit ONLY inside ClassicalModels.prepare_data().

Feature Groups
--------------
  GROUP 1  Soil Nutrients  (7 features)
  GROUP 2  Climate / Water (8 features)
  GROUP 3  Soil Chemistry  (4 features)
  GROUP 4  Composite Index (5 features)
  GROUP 5  Quantum Encoding Hints (5 features — normalised 0-1 for VQC input)

References
----------
  [R1] Kamilaris & Prenafeta-Boldú (2018). Deep learning in agriculture.
       Comput. Electron. Agric. 147, 70-90.
  [R2] Sharma et al. (2021). Machine learning applications in agriculture.
       Comput. Electron. Agric. 181, 105980.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

BASE_COLS = [
    "Nitrogen", "Phosphorus", "Potassium",
    "Temperature", "Humidity", "pH_Value", "Rainfall",
    "OrganicMatter", "ElecConductivity", "Zinc_ppm",
]

ENG_COLS = [
    # GROUP 1 — Soil Nutrients
    "soil_quality_index",   # 0.4·N + 0.3·P + 0.3·K
    "npk_total",            # N + P + K
    "N_P_ratio",            # N / (P+1)
    "N_K_ratio",            # N / (K+1)
    "P_K_ratio",            # P / (K+1)
    "nutrient_balance",     # CV of (N,P,K) — imbalance detector
    "nutrient_efficiency",  # npk_total / (OrganicMatter+1)
    # GROUP 2 — Climate / Water
    "climate_index",        # 0.4·T + 0.3·H + 0.3·R
    "water_stress_index",   # T / (R+1)
    "heat_humidity",        # T·H / 100
    "rainfall_efficiency",  # R / (T+1)
    "temp_rain_product",    # T·R / 100
    "log_rainfall",         # log1p(R)
    "humidity_sq",          # H² / 1000
    "aridity_index",        # T / (R + H + 1)
    # GROUP 3 — Soil Chemistry
    "ph_deviation",         # |pH - 6.5|
    "ph_optimal_band",      # 1 if pH ∈ [6,7]
    "ph_sq",                # pH²
    "ec_om_ratio",          # EC / (OM+1)
    # GROUP 4 — Composite
    "crop_stress_index",    # water_stress + ph_deviation + nutrient_balance
    "soil_health_score",    # (OM·10 + Zn·5 - EC·5) normalised
    "climate_soil_coupling",# climate_index · soil_quality_index / 1000
    "water_nutrient_index", # rainfall_efficiency · soil_quality_index / 100
    "quantum_feature_1",    # sin(N/250·π) — sinusoidal encoding hint
    # GROUP 5 — Quantum Encoding (normalised 0–1)
    "q_enc_N",              # N / 250
    "q_enc_T",              # (T - 5) / 43
    "q_enc_H",              # H / 100
    "q_enc_R",              # R / 4500
    "q_enc_pH",             # (pH - 3.5) / 6.5
]

ALL_FEATURE_NAMES = BASE_COLS + ENG_COLS


class FeatureEngineering:
    """
    Stateless transformer: all features are pure arithmetic.
    Call create_features(df) BEFORE train/test split.
    """

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        N   = df["Nitrogen"]
        P   = df["Phosphorus"]
        K   = df["Potassium"]
        T   = df["Temperature"]
        H   = df["Humidity"]
        pH  = df["pH_Value"]
        R   = df["Rainfall"]
        OM  = df.get("OrganicMatter",   pd.Series(2.0, index=df.index))
        EC  = df.get("ElecConductivity", pd.Series(0.8, index=df.index))
        Zn  = df.get("Zinc_ppm",         pd.Series(1.2, index=df.index))

        # ── GROUP 1 : Soil Nutrients ──────────────────────────────────────────
        df["soil_quality_index"]  = (N*0.4 + P*0.3 + K*0.3).round(4)
        df["npk_total"]           = (N + P + K).round(2)
        df["N_P_ratio"]           = (N / (P + 1)).round(4)
        df["N_K_ratio"]           = (N / (K + 1)).round(4)
        df["P_K_ratio"]           = (P / (K + 1)).round(4)
        npk_std  = pd.concat([N, P, K], axis=1).std(axis=1)
        npk_mean = (N + P + K) / 3
        df["nutrient_balance"]    = (npk_std / (npk_mean + 1e-5)).round(4)
        df["nutrient_efficiency"] = (df["npk_total"] / (OM + 1)).round(4)

        # ── GROUP 2 : Climate / Water ─────────────────────────────────────────
        df["climate_index"]       = (T*0.4 + H*0.3 + R*0.3).round(4)
        df["water_stress_index"]  = (T / (R + 1)).round(4)
        df["heat_humidity"]       = (T * H / 100).round(4)
        df["rainfall_efficiency"] = (R / (T + 1)).round(4)
        df["temp_rain_product"]   = (T * R / 100).round(4)
        df["log_rainfall"]        = np.log1p(R).round(4)
        df["humidity_sq"]         = (H ** 2 / 1000).round(4)
        df["aridity_index"]       = (T / (R + H + 1)).round(4)

        # ── GROUP 3 : Soil Chemistry ──────────────────────────────────────────
        df["ph_deviation"]        = (pH - 6.5).abs().round(4)
        df["ph_optimal_band"]     = ((pH >= 6.0) & (pH <= 7.0)).astype(float)
        df["ph_sq"]               = (pH ** 2).round(4)
        df["ec_om_ratio"]         = (EC / (OM + 1)).round(4)

        # ── GROUP 4 : Composite ───────────────────────────────────────────────
        df["crop_stress_index"]   = (df["water_stress_index"] +
                                     df["ph_deviation"] +
                                     df["nutrient_balance"]).round(4)
        soil_h = (OM * 10 + Zn * 5 - EC * 5)
        df["soil_health_score"]   = ((soil_h - soil_h.min()) /
                                     (soil_h.max() - soil_h.min() + 1e-8)).round(4)
        df["climate_soil_coupling"] = (df["climate_index"] *
                                       df["soil_quality_index"] / 1000).round(4)
        df["water_nutrient_index"]  = (df["rainfall_efficiency"] *
                                       df["soil_quality_index"] / 100).round(4)
        df["quantum_feature_1"]     = np.sin(N / 250.0 * np.pi).round(4)

        # ── GROUP 5 : Quantum Encoding (0–1 normalised) ───────────────────────
        df["q_enc_N"]   = (N / 250.0).clip(0, 1).round(4)
        df["q_enc_T"]   = ((T - 5) / 43.0).clip(0, 1).round(4)
        df["q_enc_H"]   = (H / 100.0).clip(0, 1).round(4)
        df["q_enc_R"]   = (R / 4500.0).clip(0, 1).round(4)
        df["q_enc_pH"]  = ((pH - 3.5) / 6.5).clip(0, 1).round(4)

        added = len(ENG_COLS)
        logger.info(f"FeatureEngineering: +{added} features → {df.shape[1]} total cols")
        return df

    @staticmethod
    def get_feature_names(include_season: bool = False) -> list:
        names = ALL_FEATURE_NAMES.copy()
        if include_season:
            names.append("season_code")
        return names
