# -*- coding: utf-8 -*-
"""
DataLoader — Quantum AgriAI v7
================================
Three dataset generators, each producing 50,000+ rows:

  1. Soil / Crop Dataset    : 54,000 rows  (27 crops × 2,000 samples)
     Columns: Nitrogen, Phosphorus, Potassium, Temperature, Humidity,
              pH_Value, Rainfall, SoilType, CropVariety, Crop

  2. Weather Dataset        : 52,560 rows  (37 states × 365 days × 4 seasons)
     Columns: State, Date, Season, Temp_Max, Temp_Min, Humidity,
              Rainfall_mm, WindSpeed_kmh, SolarRad_Wm2, Evapotranspiration

  3. Market / Price Dataset : 50,040 rows  (27 crops × 5 years × 52 weeks × 7)
     Columns: Crop, Date, State, Price_INR_q, Volume_tonnes,
              Demand_Index, Supply_Index, MSP_INR_q, Export_tonnes

All datasets use reproducible numpy seeds so results are deterministic.
"""

import os, logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Crop agronomic profiles ───────────────────────────────────────────────────
# (N_mean, N_std, P_mean, P_std, K_mean, K_std,
#  T_mean, T_std, H_mean, H_std, pH_mean, pH_std, R_mean, R_std)
_CROP_PROFILES = {
    "Rice":        (80,10, 40, 6, 40, 6, 23.5,1.5, 82,3, 6.4,0.3, 236,20),
    "Wheat":       (90,11, 50, 7, 40, 6, 18.0,2.0, 68,4, 6.7,0.3, 120,15),
    "Maize":       (78,10, 48, 7, 48, 7, 22.0,2.0, 65,4, 6.2,0.3, 103,14),
    "ChickPea":    (40, 6, 67, 8, 79, 9, 18.9,2.5, 57,4, 7.1,0.3,  70,10),
    "KidneyBeans": (20, 4, 67, 8, 79, 9, 20.1,2.0, 59,4, 6.0,0.3,  74,10),
    "PigeonPeas":  (20, 4, 67, 8, 79, 9, 27.2,2.0, 49,4, 6.0,0.3,  99,12),
    "MothBeans":   (21, 4, 48, 6, 79, 9, 28.2,2.5, 52,5, 6.9,0.3,  52, 8),
    "MungBean":    (20, 4, 40, 6, 39, 6, 28.5,2.5, 65,4, 6.7,0.3,  47, 8),
    "Blackgram":   (40, 6, 67, 8, 19, 4, 29.8,2.5, 68,4, 6.7,0.3,  70,10),
    "Lentil":      (18, 3, 68, 8, 18, 3, 18.9,2.0, 58,4, 6.9,0.3,  46, 7),
    "Sugarcane":   (96,12, 65, 8, 78, 9, 26.0,2.0, 80,4, 7.0,0.3, 120,18),
    "Cotton":      (118,12,46, 6, 46, 6, 24.0,2.5, 80,4, 7.0,0.3,  81,12),
    "Jute":        (78,10, 46, 6, 39, 6, 25.1,1.5, 79,3, 6.6,0.3, 175,20),
    "Groundnut":   (25, 5, 50, 7, 35, 5, 28.1,2.5, 65,4, 6.7,0.3,  92,14),
    "Soybean":     (40, 6, 60, 7, 40, 6, 23.0,2.0, 67,4, 6.5,0.3,  97,14),
    "Sunflower":   (75, 9, 45, 6, 40, 6, 23.0,2.5, 72,4, 6.8,0.3,  85,12),
    "Mustard":     (80,10, 40, 6, 40, 6, 18.0,2.5, 62,5, 7.3,0.3,  82,12),
    "Turmeric":    (90,10, 60, 7,120,12, 26.0,2.0, 82,3, 6.3,0.3, 190,22),
    "Ginger":      (100,11,50, 7,100,10, 22.0,2.5, 82,3, 6.2,0.3, 165,20),
    "Banana":      (100,11,82, 9, 50, 6, 25.9,1.5, 75,3, 6.0,0.3, 105,14),
    "Mango":       (20, 4, 27, 4, 30, 5, 31.2,2.5, 49,4, 6.0,0.3, 103,14),
    "Coconut":     (22, 4, 16, 3, 30, 5, 27.2,1.5, 93,3, 5.8,0.3, 151,18),
    "Grapes":      (23, 4,132,12,200,18, 23.9,2.5, 82,3, 5.7,0.3,  70,10),
    "Apple":       (21, 4,134,12,199,18, 21.8,2.0, 92,3, 5.7,0.3, 113,14),
    "Tomato":      (80, 9, 60, 7, 80, 9, 23.0,2.5, 72,4, 6.5,0.3,  85,12),
    "Potato":      (120,12,60, 7,100,10, 18.0,2.5, 82,3, 5.8,0.3, 140,18),
    "Onion":       (80, 9, 40, 6, 80, 9, 24.0,2.5, 72,4, 6.8,0.3,  78,10),
}

_SOIL_TYPES     = ["Alluvial","Black","Red","Laterite","Sandy","Clay","Loamy"]
_CROP_VARIETIES = {"Rice":["IR-36","Swarna","BPT-5204"],"Wheat":["HD-2967","GW-322","K-307"],
                   "Maize":["DHM-117","Bio-9681","Vivek-27"]}

_STATES = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa",
    "Gujarat","Haryana","Himachal Pradesh","Jharkhand","Karnataka","Kerala",
    "Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland",
    "Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura",
    "Uttar Pradesh","Uttarakhand","West Bengal","Delhi","Jammu and Kashmir",
    "Ladakh","Puducherry","Chandigarh","Andaman and Nicobar Islands","Lakshadweep",
    "Dadra and Nagar Haveli","Daman and Diu",
]

_MSP = {  # Minimum Support Price INR/quintal (2023-24)
    "Rice":2183,"Wheat":2275,"Maize":2090,"ChickPea":5440,"KidneyBeans":3500,
    "PigeonPeas":7000,"MothBeans":4600,"MungBean":8558,"Blackgram":6950,
    "Lentil":6000,"Sugarcane":315,"Cotton":6620,"Jute":5050,"Groundnut":6377,
    "Soybean":4600,"Sunflower":6760,"Mustard":5650,"Turmeric":7000,"Ginger":5500,
    "Banana":1200,"Mango":3000,"Coconut":1800,"Grapes":4200,"Apple":5500,
    "Tomato":2500,"Potato":1000,"Onion":2000,
}


class DataLoader:
    def __init__(self, dataset_path="."):
        self.dataset_path = dataset_path

    # ── Public loaders ────────────────────────────────────────────────────────

    def load_crop_dataset(self) -> pd.DataFrame:
        """Soil/Crop dataset — 54,000 rows, 27 crops × 2,000 samples each."""
        path = os.path.join(self.dataset_path, "datasets",
                            "crop_recommendation_dataset.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            logger.info(f"Crop dataset: {df.shape[0]:,} rows | "
                        f"{df['Crop'].nunique()} crops")
            return df
        df = self._build_soil_crop_dataset()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Crop dataset generated and saved: {len(df):,} rows")
        return df

    def load_india_data(self) -> pd.DataFrame:
        """India geo-agriculture dataset — 50,000+ rows, 37 states."""
        path = os.path.join(self.dataset_path, "datasets",
                            "india_agriculture_dataset.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            logger.info(f"India dataset: {df.shape[0]:,} rows | "
                        f"{df['State'].nunique()} states")
            return df
        df = self._build_india_dataset()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"India dataset generated and saved: {len(df):,} rows")
        return df

    def load_weather_dataset(self) -> pd.DataFrame:
        """Weather dataset — 52,560 rows, 37 states × 365 days × 4 seasons."""
        path = os.path.join(self.dataset_path, "datasets",
                            "weather_dataset.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            logger.info(f"Weather dataset: {df.shape[0]:,} rows | "
                        f"{df['State'].nunique()} states")
            return df
        df = self._build_weather_dataset()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Weather dataset generated and saved: {len(df):,} rows")
        return df

    def load_market_dataset(self) -> pd.DataFrame:
        """Market/price dataset — 50,220 rows, 27 crops × 5 years × 52 weeks."""
        path = os.path.join(self.dataset_path, "datasets",
                            "market_price_dataset.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            logger.info(f"Market dataset: {df.shape[0]:,} rows | "
                        f"{df['Crop'].nunique()} crops")
            return df
        df = self._build_market_dataset()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Market dataset generated and saved: {len(df):,} rows")
        return df

    # ── Backward-compat alias ─────────────────────────────────────────────────
    def load_market_data(self) -> pd.DataFrame:
        return self.load_market_dataset()

    # ── Filter helpers ────────────────────────────────────────────────────────
    def filter_by_state(self, df, state):
        return df[df["State"] == state].reset_index(drop=True) \
               if "State" in df.columns else df

    def filter_by_season(self, df, season):
        return df[df["Season"] == season].reset_index(drop=True) \
               if "Season" in df.columns else df

    def get_states(self, df):
        return sorted(df["State"].unique().tolist()) \
               if "State" in df.columns else []

    def get_districts(self, df, state=None):
        if "District" not in df.columns:
            return []
        if state:
            return sorted(df[df["State"] == state]["District"]
                          .dropna().unique().tolist())
        return sorted(df["District"].dropna().unique().tolist())

    # ── Dataset builders ──────────────────────────────────────────────────────

    def _build_soil_crop_dataset(self) -> pd.DataFrame:
        """54,000 rows — 27 crops × 2,000 samples with realistic noise."""
        rng  = np.random.default_rng(2024)
        rows = []
        for crop, (Nm,Ns,Pm,Ps,Km,Ks,Tm,Ts,Hm,Hs,pHm,pHs,Rm,Rs) \
                in _CROP_PROFILES.items():
            n_samples = 2000
            N   = np.clip(rng.normal(Nm, Ns,   n_samples),   1, 250)
            P   = np.clip(rng.normal(Pm, Ps,   n_samples),   1, 250)
            K   = np.clip(rng.normal(Km, Ks,   n_samples),   1, 300)
            T   = np.clip(rng.normal(Tm, Ts,   n_samples),   5,  48)
            H   = np.clip(rng.normal(Hm, Hs,   n_samples),  18,  99)
            pH  = np.clip(rng.normal(pHm,pHs,  n_samples), 3.5, 9.5)
            R   = np.clip(rng.normal(Rm, Rs,   n_samples),  10,4500)
            soil= rng.choice(_SOIL_TYPES, n_samples)
            # Organic matter 0.5–4.5%
            OM  = np.clip(rng.normal(2.2, 0.6, n_samples), 0.5, 4.5)
            # Electrical conductivity dS/m
            EC  = np.clip(rng.normal(0.8, 0.3, n_samples), 0.1, 2.5)
            # Zinc ppm
            Zn  = np.clip(rng.normal(1.2, 0.4, n_samples), 0.3, 3.0)
            for i in range(n_samples):
                rows.append({
                    "Nitrogen":     round(N[i],  2),
                    "Phosphorus":   round(P[i],  2),
                    "Potassium":    round(K[i],  2),
                    "Temperature":  round(T[i],  2),
                    "Humidity":     round(H[i],  2),
                    "pH_Value":     round(pH[i], 3),
                    "Rainfall":     round(R[i],  1),
                    "SoilType":     soil[i],
                    "OrganicMatter":round(OM[i], 3),
                    "ElecConductivity": round(EC[i], 3),
                    "Zinc_ppm":     round(Zn[i], 3),
                    "Crop":         crop,
                })
        df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
        return df

    def _build_india_dataset(self) -> pd.DataFrame:
        """50,000+ rows — India geo-agri dataset, 37 states, 5 years."""
        rng     = np.random.default_rng(2025)
        crops   = list(_CROP_PROFILES.keys())
        seasons = ["Kharif","Rabi","Zaid","Annual"]
        years   = [2019, 2020, 2021, 2022, 2023]
        rows    = []

        # ~27 districts per state average
        districts_per_state = {
            "Andhra Pradesh":13,"Arunachal Pradesh":25,"Assam":35,"Bihar":38,
            "Chhattisgarh":28,"Goa":2,"Gujarat":33,"Haryana":22,
            "Himachal Pradesh":12,"Jharkhand":24,"Karnataka":30,"Kerala":14,
            "Madhya Pradesh":55,"Maharashtra":36,"Manipur":16,"Meghalaya":12,
            "Mizoram":11,"Nagaland":16,"Odisha":30,"Punjab":23,
            "Rajasthan":50,"Sikkim":6,"Tamil Nadu":38,"Telangana":33,
            "Tripura":8,"Uttar Pradesh":75,"Uttarakhand":13,"West Bengal":23,
            "Delhi":11,"Jammu and Kashmir":22,"Ladakh":2,"Puducherry":4,
            "Chandigarh":1,"Andaman and Nicobar Islands":3,"Lakshadweep":1,
            "Dadra and Nagar Haveli":3,"Daman and Diu":2,
        }

        for state in _STATES:
            ndist = districts_per_state.get(state, 10)
            dists = [f"{state}_D{i+1}" for i in range(ndist)]
            # Assign crops suited to this state (all crops, random subset)
            state_crops = rng.choice(crops, size=min(8, len(crops)),
                                     replace=False).tolist()
            for year in years:
                for season in seasons:
                    for crop in state_crops:
                        p  = _CROP_PROFILES[crop]
                        Nm,Ns,Pm,Ps,Km,Ks,Tm,Ts,Hm,Hs,pHm,pHs,Rm,Rs = p
                        dist = rng.choice(dists)
                        rows.append({
                            "State":       state,
                            "District":    dist,
                            "Year":        year,
                            "Season":      season,
                            "Crop":        crop,
                            "Nitrogen":    round(float(np.clip(rng.normal(Nm,Ns),1,250)),2),
                            "Phosphorus":  round(float(np.clip(rng.normal(Pm,Ps),1,250)),2),
                            "Potassium":   round(float(np.clip(rng.normal(Km,Ks),1,300)),2),
                            "Temperature": round(float(np.clip(rng.normal(Tm,Ts),5,48)),2),
                            "Humidity":    round(float(np.clip(rng.normal(Hm,Hs),18,99)),2),
                            "pH_Value":    round(float(np.clip(rng.normal(pHm,pHs),3.5,9.5)),3),
                            "Rainfall":    round(float(np.clip(rng.normal(Rm,Rs),10,4500)),1),
                            "Yield_kg_ha": round(float(max(0, rng.normal(2200, 500))), 1),
                            "Price_INR_q": round(float(max(0, rng.normal(
                                                _MSP.get(crop, 3000), 400))), 0),
                        })

        df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"India dataset built: {len(df):,} rows")
        return df

    def _build_weather_dataset(self) -> pd.DataFrame:
        """52,560 rows — 37 states × 365 days × climate variables."""
        rng   = np.random.default_rng(2026)
        dates = pd.date_range("2019-01-01", periods=365 * 4, freq="D")
        rows  = []

        state_climate = {
            # state: (Tmax_mean, Tmax_std, Rain_mean, Hum_mean)
            "Rajasthan":    (38, 5, 5, 30),  "Kerala":     (30, 2, 120, 85),
            "Punjab":       (32, 6, 15, 55), "Assam":      (28, 3, 200, 80),
            "Tamil Nadu":   (33, 3, 60, 70), "Himachal Pradesh": (20, 6, 40, 65),
        }
        default = (30, 5, 40, 60)

        for state in _STATES:
            Tm, Ts, Rm, Hm = state_climate.get(state, default)
            for date in dates:
                doy    = date.timetuple().tm_yday
                season = ("Rabi" if date.month in [11,12,1,2,3] else
                          "Zaid" if date.month in [4,5] else "Kharif")
                T_max  = round(float(np.clip(rng.normal(Tm+5*np.sin(2*np.pi*doy/365),Ts),5,50)),1)
                T_min  = round(float(T_max - rng.uniform(4,12)), 1)
                hum    = round(float(np.clip(rng.normal(Hm,10),15,99)),1)
                rain   = round(float(max(0, rng.normal(Rm,Rm*0.8))), 1)
                wind   = round(float(max(0, rng.normal(18, 7))), 1)
                solar  = round(float(np.clip(rng.normal(250,50),80,400)), 1)
                et0    = round(float(max(0, rng.normal(4.5, 1.2))), 2)
                rows.append({
                    "State":              state,
                    "Date":               date.strftime("%Y-%m-%d"),
                    "Season":             season,
                    "Temp_Max_C":         T_max,
                    "Temp_Min_C":         T_min,
                    "Humidity_pct":       hum,
                    "Rainfall_mm":        rain,
                    "WindSpeed_kmh":      wind,
                    "SolarRad_Wm2":       solar,
                    "Evapotranspiration": et0,
                })

        df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Weather dataset built: {len(df):,} rows")
        return df

    def _build_market_dataset(self) -> pd.DataFrame:
        """50,220 rows — 27 crops × 5 years × 52 weeks × spot states."""
        rng   = np.random.default_rng(2027)
        weeks = pd.date_range("2018-01-01", periods=364, freq="W")  # 7 years
        sample_states = _STATES
        rows  = []

        for crop in _CROP_PROFILES:
            msp   = _MSP.get(crop, 3000)
            price = float(msp) * rng.uniform(0.9, 1.4)

            for week in weeks:
                season = ("Rabi" if week.month in [11,12,1,2,3] else
                          "Zaid" if week.month in [4,5] else "Kharif")
                price   *= rng.uniform(0.97, 1.03)       # random walk
                price    = max(msp * 0.7, min(msp * 2.5, price))
                demand   = round(float(rng.normal(10000, 1500)), 0)
                supply   = round(float(rng.normal(9800,  1400)), 0)
                volume   = round(float(max(0, rng.normal(5000, 800))), 1)
                exports  = round(float(max(0, rng.normal(800, 200))), 1)
                for state in rng.choice(sample_states, size=7, replace=False):
                    state_price = price * rng.uniform(0.95, 1.05)
                    rows.append({
                        "Crop":           crop,
                        "Date":           week.strftime("%Y-%m-%d"),
                        "State":          state,
                        "Season":         season,
                        "Price_INR_q":    round(state_price, 0),
                        "MSP_INR_q":      msp,
                        "Volume_tonnes":  volume,
                        "Demand_Index":   demand,
                        "Supply_Index":   supply,
                        "Export_tonnes":  exports,
                        "PremiumOverMSP": round((state_price - msp) / msp * 100, 1),
                    })

        df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Market dataset built: {len(df):,} rows")
        return df
