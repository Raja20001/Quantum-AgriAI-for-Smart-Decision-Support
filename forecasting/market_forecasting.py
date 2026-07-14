# -*- coding: utf-8 -*-
"""
Market Forecasting — Quantum AgriAI v7
========================================
Workflow:  Market Dataset → Price Trend → Bollinger Bands → Alerts → Seasonal Summary

FIX: Handles both old column names (price/demand/supply) and new dataset
     column names (Price_INR_q / Demand_Index / Supply_Index).
     Robust to missing columns — no KeyError crashes.

Algorithms:
  - Bollinger Bands  : MA ± 2σ price envelope  [John Bollinger, 1983]
  - EMA Forecast     : Exponential moving average (α=0.3) demand trend
  - Supply-Demand Gap: Deficit / surplus analysis
  - Price Alerts     : Threshold-based market signals

Reference:
  [M1] Bollinger, J. (2002) Bollinger on Bollinger Bands. McGraw-Hill.
  [M2] Brown, R.G. (1963) Smoothing, Forecasting and Prediction. Prentice-Hall.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Column alias map: maps new dataset names → internal names
_COL_ALIASES = {
    "Price_INR_q":   "price",
    "Demand_Index":  "demand",
    "Supply_Index":  "supply",
    "Date":          "date",
    "MSP_INR_q":     "msp",
}


class MarketForecasting:
    """
    Market intelligence module.
    Accepts both raw market_df (new column names) and pre-renamed DataFrames.
    """

    def __init__(self, window: int = 7, alert_threshold: float = 3000.0):
        self.window          = window
        self.alert_threshold = alert_threshold

    def _normalise_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename new dataset column names to internal standard names."""
        df = df.copy()
        rename = {k: v for k, v in _COL_ALIASES.items() if k in df.columns}
        df = df.rename(columns=rename)
        return df

    def price_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Band price envelope: MA ± 2σ."""
        df = self._normalise_cols(df)
        if "price" not in df.columns:
            logger.warning("price_trend: 'price' column not found.")
            return df
        df["moving_average"] = df["price"].rolling(window=self.window, min_periods=1).mean()
        df["rolling_std"]    = df["price"].rolling(window=self.window, min_periods=1).std().fillna(0)
        df["upper_band"]     = df["moving_average"] + 2 * df["rolling_std"]
        df["lower_band"]     = df["moving_average"] - 2 * df["rolling_std"]
        df["price_signal"]   = (df["price"] > df["upper_band"]).astype(int)
        df["pct_change"]     = df["price"].pct_change().round(4).fillna(0)
        logger.info("Bollinger Bands computed.")
        return df

    def demand_prediction(self, df: pd.DataFrame) -> dict:
        """EMA demand forecast (α=0.3)."""
        df = self._normalise_cols(df)
        if "demand" not in df.columns:
            return {"trend": "N/A", "note": "No demand column"}
        demand = df["demand"].dropna()
        if len(demand) == 0:
            return {"trend": "N/A"}
        ema   = demand.ewm(alpha=0.3, adjust=False).mean()
        diff  = ema.diff().iloc[-5:].mean() if len(ema) >= 5 else 0
        trend = "rising" if diff > 0 else "falling"
        return {
            "mean_demand":   round(float(demand.mean()),  2),
            "std_demand":    round(float(demand.std()),   2),
            "forecast_next": round(float(ema.iloc[-1]),   2),
            "trend":         trend,
        }

    def supply_demand_gap(self, df: pd.DataFrame) -> dict:
        """Deficit/surplus analysis."""
        df = self._normalise_cols(df)
        if "supply" not in df.columns or "demand" not in df.columns:
            return {"note": "No supply/demand columns"}
        gap = df["demand"] - df["supply"]
        return {
            "avg_gap":      round(float(gap.mean()), 2),
            "max_shortage": round(float(gap.max()),  2),
            "max_surplus":  round(float(gap.min()),  2),
        }

    def generate_price_alerts(self, df: pd.DataFrame) -> list:
        """Threshold-based price alerts."""
        df     = self._normalise_cols(df)
        alerts = []
        if "price" not in df.columns:
            return alerts
        for _, row in df.iterrows():
            p = float(row.get("price", 0) or 0)
            if p > self.alert_threshold:
                alerts.append({
                    "date":  str(row.get("date", "N/A")),
                    "price": round(p, 2),
                    "alert": "Price above threshold — consider selling now.",
                })
        logger.info(f"{len(alerts)} price alerts generated.")
        return alerts

    def seasonal_summary(self, df: pd.DataFrame) -> dict:
        """Monthly average price summary."""
        df = self._normalise_cols(df)
        if "price" not in df.columns or "date" not in df.columns:
            return {"note": "Insufficient columns for seasonal summary"}
        try:
            df2          = df.copy()
            df2["date"]  = pd.to_datetime(df2["date"], errors="coerce")
            df2          = df2.dropna(subset=["date"])
            df2["month"] = df2["date"].dt.month
            monthly      = df2.groupby("month")["price"].mean().round(2)
            if len(monthly) == 0:
                return {}
            return {
                "monthly_avg": monthly.to_dict(),
                "peak_month":  int(monthly.idxmax()),
                "low_month":   int(monthly.idxmin()),
            }
        except Exception as e:
            logger.warning(f"seasonal_summary: {e}")
            return {}

    def crop_price_summary(self, df: pd.DataFrame) -> dict:
        """Per-crop price statistics from market dataset."""
        df = self._normalise_cols(df)
        if "price" not in df.columns or "Crop" not in df.columns:
            return {}
        grouped = df.groupby("Crop")["price"].agg(["mean","std","min","max"])
        return grouped.round(0).to_dict("index")

    def full_forecast(self, df: pd.DataFrame) -> dict:
        """Run full market forecasting pipeline."""
        df_t   = self.price_trend(df)
        demand = self.demand_prediction(df)
        alerts = self.generate_price_alerts(df_t)
        season = self.seasonal_summary(df_t)
        gap    = self.supply_demand_gap(df)
        return {
            "trend_data":    df_t,
            "demand_info":   demand,
            "alerts":        alerts,
            "seasonal":      season,
            "supply_demand": gap,
        }
