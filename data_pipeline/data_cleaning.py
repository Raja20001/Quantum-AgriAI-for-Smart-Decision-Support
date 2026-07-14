# -*- coding: utf-8 -*-
"""
DataCleaning — Quantum AgriAI v7
==================================
6-step pipeline:
  1. fix_dtypes          — coerce numerics, strip strings
  2. remove_missing      — drop rows with missing core features
  3. deduplicate         — drop exact duplicates
  4. validate_ranges     — hard biological/physical limits
  5. remove_outliers     — 1.5·IQR on core soil/climate columns
  6. encode_categoricals — label-encode SoilType, Season if present

Safe for Soil, Weather, Market and India datasets.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

_CORE_RANGES = {
    "Nitrogen":         (0,   250),
    "Phosphorus":       (0,   250),
    "Potassium":        (0,   300),
    "Temperature":      (-5,   50),
    "Humidity":         (10,   99),
    "pH_Value":         (3.0, 9.9),
    "Rainfall":         (5,  4500),
    "OrganicMatter":    (0.1,  6.0),
    "ElecConductivity": (0.0,  5.0),
    "Zinc_ppm":         (0.0,  5.0),
}
_STRING_COLS = {"Crop","State","District","Season","SoilType","CropVariety","Date"}
_SOIL_ENCODE = {"Alluvial":0,"Black":1,"Red":2,"Laterite":3,"Sandy":4,"Clay":5,"Loamy":6}
_SEASON_ENCODE = {"Kharif":1,"Rabi":2,"Zaid":3,"Annual":4}


class DataCleaning:
    """6-step deterministic cleaning pipeline."""

    def fix_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if col in _STRING_COLS:
                df[col] = df[col].astype(str).str.strip()
            elif df[col].dtype == object:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().mean() > 0.80:
                    df[col] = converted
        return df

    def remove_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        core = [c for c in _CORE_RANGES if c in df.columns]
        before = len(df)
        df = df.dropna(subset=core).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.info(f"Missing values: {dropped} rows dropped")
        return df

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        dups = before - len(df)
        if dups:
            logger.info(f"Deduplication: {dups} duplicates removed")
        return df

    def validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        for col, (lo, hi) in _CORE_RANGES.items():
            if col in df.columns:
                df = df[(df[col] >= lo) & (df[col] <= hi)]
        removed = before - len(df)
        if removed:
            logger.info(f"Range validation: {removed} rows removed")
        return df.reset_index(drop=True)

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """IQR method on core numeric columns only."""
        before = len(df)
        for col in [c for c in _CORE_RANGES if c in df.columns]:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr    = q3 - q1
            df     = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]
        logger.info(f"Outlier removal: {before - len(df)} rows removed, "
                    f"{len(df)} remaining")
        return df.reset_index(drop=True)

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode SoilType and Season into numeric codes."""
        if "SoilType" in df.columns:
            df["SoilType_code"] = (df["SoilType"]
                                   .map(_SOIL_ENCODE)
                                   .fillna(-1).astype(int))
        if "Season" in df.columns:
            df["season_code"] = (df["Season"]
                                 .map(_SEASON_ENCODE)
                                 .fillna(0).astype(int))
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.fix_dtypes(df)
        df = self.remove_missing(df)
        df = self.deduplicate(df)
        df = self.validate_ranges(df)
        df = self.remove_outliers(df)
        df = self.encode_categoricals(df)
        logger.info(f"Cleaning complete. Final shape: {df.shape}")
        return df

    @staticmethod
    def data_quality_report(df: pd.DataFrame) -> dict:
        return {
            "rows":       len(df),
            "columns":    len(df.columns),
            "missing":    int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "dtypes":     {c: str(t) for c, t in df.dtypes.items()},
        }
