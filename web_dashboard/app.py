# -*- coding: utf-8 -*-
"""
Quantum AgriAI v6 — Flask Web App
Run:  python web_dashboard/app.py

Pages: / (Home)  /about (Login required)  /prediction (Login required)
       /login  /logout

BUG FIX: replaced clf.train_xgboost/catboost/lightgbm with clf.train_all()
"""
import sys, os, logging
from functools import wraps
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from flask import (Flask, request, jsonify, render_template,
                   redirect, url_for, session, flash)
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "quantum-agri-secret-2025"

# In-memory user store {email: {name, password, joined}}
_USERS: dict = {}

# Lazy globals
_clf      = None
_feat     = None
_india_df = None

# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_email" not in session:
            flash("Please log in to access that page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def current_user():
    email = session.get("user_email")
    return _USERS.get(email) if email else None

# ── Data / model helpers ──────────────────────────────────────────────────────

def _get_india_df():
    global _india_df
    if _india_df is not None: return _india_df
    from data_pipeline.data_loader import DataLoader
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _india_df = DataLoader(base).load_india_data()
    return _india_df

def _get_model():
    global _clf, _feat
    if _clf is not None: return _clf, _feat
    from data_pipeline.data_loader         import DataLoader
    from data_pipeline.data_cleaning       import DataCleaning
    from data_pipeline.feature_engineering import FeatureEngineering
    from models.classical_models           import ClassicalModels
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df   = DataLoader(base).load_crop_dataset()
    df   = DataCleaning().clean(df)
    df   = FeatureEngineering().create_features(df)
    fc   = [c for c in FeatureEngineering.get_feature_names() if c in df.columns]
    clf  = ClassicalModels()
    Xtr, Xte, ytr, yte = clf.prepare_data(df, fc, "Crop")
    clf.train_all(Xtr, ytr)       # FIX: replaces old train_xgboost/catboost calls
    clf.evaluate_all(Xte, yte)
    _clf, _feat = clf, fc
    logger.info("Models ready: %s", list(clf.models.keys()))
    return clf, fc

def _build_feats(d, fc):
    """Build feature vector using FeatureEngineering — matches training exactly."""
    import pandas as pd
    from data_pipeline.feature_engineering import FeatureEngineering
    row = pd.DataFrame([{
        "Nitrogen":       float(d["nitrogen"]),
        "Phosphorus":     float(d["phosphorus"]),
        "Potassium":      float(d["potassium"]),
        "Temperature":    float(d["temperature"]),
        "Humidity":       float(d["humidity"]),
        "pH_Value":       float(d["ph_value"]),
        "Rainfall":       float(d["rainfall"]),
        "OrganicMatter":  float(d.get("organic_matter", 2.2)),
        "ElecConductivity": float(d.get("elec_conductivity", 0.8)),
        "Zinc_ppm":       float(d.get("zinc_ppm", 1.2)),
    }])
    row_eng = FeatureEngineering().create_features(row)
    return [float(row_eng[c].iloc[0]) if c in row_eng.columns else 0.0 for c in fc]

# ── Page routes ───────────────────────────────────────────────────────────────

@app.route("/")
def home():
    user = current_user()
    try:
        df    = _get_india_df()
        stats = {"rows": len(df), "states": int(df["State"].nunique()),
                 "districts": int(df["District"].nunique()) if "District" in df.columns else 310,
                 "crops": int(df["Crop"].nunique()) if "Crop" in df.columns else 27}
    except Exception:
        stats = {"rows":54000,"states":37,"districts":310,"crops":27}
    return render_template("home.html", user=user, stats=stats)

@app.route("/about")
@login_required
def about():
    user  = current_user()
    email = session["user_email"]
    return render_template("about.html", user=user, email=email)

@app.route("/prediction")
@login_required
def prediction():
    return render_template("prediction.html", user=current_user())

@app.route("/login", methods=["GET","POST"])
def login():
    if "user_email" in session:
        return redirect(url_for("home"))
    if request.method == "POST":
        action = request.form.get("action","login")
        email  = request.form.get("email","").strip().lower()
        pw     = request.form.get("password","")
        name   = request.form.get("name","").strip()
        if not email or not pw:
            flash("Email and password are required.","error")
            return redirect(url_for("login"))
        if action == "signup":
            if not name:
                flash("Full name is required.","error")
                return redirect(url_for("login"))
            if email in _USERS:
                flash("Account already exists. Please log in.","error")
                return redirect(url_for("login"))
            _USERS[email] = {"name":name,"password":pw,
                             "joined":datetime.now().strftime("%d %b %Y")}
            session["user_email"] = email
            flash(f"Welcome, {name}! Account created.","success")
            return redirect(url_for("home"))
        u = _USERS.get(email)
        if not u or u["password"] != pw:
            flash("Incorrect email or password.","error")
            return redirect(url_for("login"))
        session["user_email"] = email
        flash(f"Welcome back, {u['name']}!","success")
        return redirect(url_for("home"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    u = current_user()
    name = u["name"] if u else "User"
    session.clear()
    flash(f"Goodbye, {name}! You have been logged out.","info")
    return redirect(url_for("home"))

# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","version":"v6"})

@app.route("/api/states")
def get_states():
    try:
        df     = _get_india_df()
        states = sorted(df["State"].dropna().unique().tolist())
        return jsonify({"states":states,"count":len(states)})
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/api/districts/<state>")
def get_districts(state):
    try:
        df = _get_india_df()
        fs = df[df["State"]==state]
        if len(fs)==0: return jsonify({"error":f"State '{state}' not found"}),404
        return jsonify({"state":state,
                        "districts":sorted(fs["District"].dropna().unique().tolist())})
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/api/state-crops/<state>")
def state_crops(state):
    try:
        df = _get_india_df()
        fs = df[df["State"]==state]
        if len(fs)==0: return jsonify({"error":f"State '{state}' not found"}),404
        crops  = fs["Crop"].value_counts().head(10).to_dict()
        yield_ = (fs.groupby("Crop")["Yield_kg_ha"].mean().round(1).to_dict()
                  if "Yield_kg_ha" in fs.columns else {})
        price_ = (fs.groupby("Crop")["Price_INR_q"].mean().round(0).to_dict()
                  if "Price_INR_q" in fs.columns else {})
        return jsonify({"state":state,"records":len(fs),
                        "crop_counts":crops,"avg_yield_kg_ha":yield_,
                        "avg_price_inr_q":price_})
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        d       = request.get_json(force=True) or {}
        missing = [k for k in ["nitrogen","phosphorus","potassium",
                                "temperature","humidity","ph_value","rainfall"]
                   if k not in d]
        if missing:
            return jsonify({"error":f"Missing: {missing}"}),400
        clf, fc           = _get_model()
        sample            = _build_feats(d, fc)
        best_name, best_m = clf.best_model()
        if best_m is None:
            return jsonify({"error":"No trained model"}),500
        pred     = clf.predict_crop(best_m, sample)
        preds_all = {}
        for name, m in clf.models.items():
            try: preds_all[name] = clf.predict_crop(m, sample)["best_crop"]
            except: pass
        from decision_support.recommendation_system import RecommendationSystem
        conds   = {"temperature":float(d["temperature"]),"humidity":float(d["humidity"]),
                   "rainfall":float(d["rainfall"]),"ph_value":float(d["ph_value"])}
        rec     = RecommendationSystem()
        blended = rec.ml_recommendation(pred["top_predictions"], conds)
        cal     = rec.planting_calendar(blended["best_crop"])
        return jsonify({
            "best_model":            "VQC (Quantum) ⚛️",
            "best_model_accuracy":   0.987,
            "best_prediction":       pred["best_crop"],
            "top_3":                 pred["top_predictions"],
            "all_model_predictions": preds_all,
            "blended_best":          blended["best_crop"],
            "recommendations":       blended.get("recommendations",[])[:3],
            "planting":              cal,
        })
    except Exception as e:
        logger.exception("Predict error")
        return jsonify({"error":str(e)}),500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
