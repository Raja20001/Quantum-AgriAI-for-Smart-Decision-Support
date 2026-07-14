# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np, os, logging

logger = logging.getLogger(__name__)

DARK = {"figure.facecolor":"#0f1117","axes.facecolor":"#0f1117","axes.edgecolor":"#334155",
        "text.color":"#e2e8f0","axes.labelcolor":"#e2e8f0","xtick.color":"#94a3b8",
        "ytick.color":"#94a3b8","grid.color":"#1e293b","axes.grid":True,
        "grid.linestyle":"--","grid.alpha":0.5}

COLORS = {"ExtraTrees":"#34d399","XGBoost":"#60a5fa","LightGBM":"#f59e0b",
          "CatBoost":"#a78bfa","Voting Ensemble":"#ec4899",
          "Random Forest":"#003366","Gradient Boosting":"#0D9E75",
          "SVM":"#5B8DB8","Naive Bayes":"#9B59B6","KNN":"#1ABC9C",
          "VQC":"#F59E0B","QAOA":"#E8593C","Quantum Kernel SVM":"#FF6B6B"}

class GraphGenerator:
    def __init__(self, output_dir="outputs/charts"):
        self.out = output_dir; os.makedirs(output_dir, exist_ok=True)

    def _s(self): plt.rcParams.update(DARK)

    def _save(self, fig, name):
        path = os.path.join(self.out, name)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig); logger.info(f"Chart: {path}"); return path

    def model_accuracy_bar(self, results, quantum_results=None):
        self._s()
        all_r = dict(results)
        if quantum_results:
            for k,v in quantum_results.items():
                if v.get("accuracy"): all_r[k] = v
        names = list(all_r.keys())
        accs  = [all_r[n].get("accuracy",0) or 0 for n in names]
        kaps  = [all_r[n].get("kappa",0)    or 0 for n in names]
        colors= [COLORS.get(n,"#888888") for n in names]
        x = np.arange(len(names)); w = 0.38
        fig, ax = plt.subplots(figsize=(14,6), facecolor="#0f1117")
        b1 = ax.bar(x-w/2, accs, w, color=colors, alpha=0.92, label="Accuracy")
        b2 = ax.bar(x+w/2, kaps, w, color=colors, alpha=0.55, label="Kappa")
        for b in list(b1)+list(b2):
            h = b.get_height()
            if h > 0.01:
                ax.text(b.get_x()+b.get_width()/2, h+0.005, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=7, color="#e2e8f0")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=25, ha="right")
        ax.set_ylim(0,1.12); ax.set_ylabel("Score"); ax.legend(fontsize=9)
        ax.axhline(0.95, color="#34d399", ls="--", lw=1.2, alpha=0.7, label="95% target")
        ax.set_title("All Model Performance: Accuracy & Kappa", fontsize=13, fontweight="bold")
        ax.axvline(x=4.5, color="#475569", ls=":", lw=1.2)
        ax.text(2.0, 1.08, "TRENDING", fontsize=8, color="#34d399", ha="center")
        ax.text(7.5, 1.08, "CLASSIC",  fontsize=8, color="#94a3b8", ha="center")
        fig.tight_layout(); return self._save(fig, "model_accuracy.png")

    def feature_importance_h(self, importance, title="Feature Importance"):
        self._s()
        items = list(importance.items())[:18]
        feats = [i[0] for i in items]; vals = [i[1] for i in items]
        colors = cm.plasma(np.linspace(0.2,0.9,len(feats)))
        fig, ax = plt.subplots(figsize=(10,7), facecolor="#0f1117")
        ax.barh(feats[::-1], vals[::-1], color=colors, edgecolor="none")
        ax.set_xlabel("Importance"); ax.set_title(title, fontsize=12, fontweight="bold")
        fig.tight_layout(); return self._save(fig, f"fi_{title[:8].lower().replace(' ','_')}.png")

    def confusion_matrix_chart(self, cm_data, class_names):
        self._s()
        arr = np.array(cm_data)
        fig, ax = plt.subplots(figsize=(14,12), facecolor="#0f1117")
        im = ax.imshow(arr, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout(); return self._save(fig, "confusion_matrix.png")

    def crop_distribution(self, df, target_col="Crop"):
        self._s()
        counts = df[target_col].value_counts()
        colors = cm.viridis(np.linspace(0.2,0.9,len(counts)))
        fig, ax = plt.subplots(figsize=(12,7), facecolor="#0f1117")
        bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="none")
        for b,v in zip(bars, counts.values):
            ax.text(v+0.3, b.get_y()+b.get_height()/2, str(v), va="center", fontsize=8, color="#94a3b8")
        ax.set_xlabel("Count"); ax.set_title("Crop Distribution", fontsize=12, fontweight="bold")
        fig.tight_layout(); return self._save(fig, "crop_distribution.png")

    def correlation_heatmap(self, df):
        self._s()
        num  = df.select_dtypes(include="number").columns[:15]
        corr = df[num].corr()
        fig, ax = plt.subplots(figsize=(11,9), facecolor="#0f1117")
        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                ax.text(j,i,f"{corr.values[i,j]:.2f}",ha="center",va="center",
                        fontsize=6.5,color="white" if abs(corr.values[i,j])>0.5 else "#64748b")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        ax.set_title("Feature Correlation Heatmap (Pearson r)", fontsize=12, fontweight="bold")
        fig.tight_layout(); return self._save(fig, "correlation_heatmap.png")

    def market_price_trend(self, df):
        self._s()
        fig, ax = plt.subplots(figsize=(13,5), facecolor="#0f1117")
        ax.plot(df["price"].values, color="#f59e0b", lw=1.8, label="Price")
        if "moving_average" in df.columns:
            ax.plot(df["moving_average"].values, color="#60a5fa", lw=2, ls="--", label="Moving Avg")
        if "upper_band" in df.columns:
            ax.fill_between(range(len(df)), df["lower_band"].values, df["upper_band"].values,
                            alpha=0.12, color="#a78bfa", label="Bollinger Bands")
        ax.set_title("Market Price Trend & Bollinger Bands", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); fig.tight_layout()
        return self._save(fig, "market_price_trend.png")

    def quantum_circuit_diagram(self, n_qubits=4, p_layers=2):
        self._s()
        fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(16,6), facecolor="#0f1117")
        for ax in (ax1,ax2,ax3):
            ax.axis("off"); ax.set_facecolor("#0a0e1a"); ax.patch.set_facecolor("#0a0e1a")
        vqc_txt = f"VQC - Variational Quantum Classifier\n{'='*40}\n\n"
        for i in range(n_qubits):
            vqc_txt += f"q{i}: -[H]-[ZZ]-[Ry(t{i})]-[Rz(t{i+n_qubits})]-[M]-\n"
        vqc_txt += f"\nZZFeatureMap: x -> |phi(x)>\nRealAmplitudes: V(theta) reps=2\nCOBYLA gradient-free opt\nStatevectorSampler (Qiskit 1.x)"
        ax1.text(0.05,0.95,vqc_txt,transform=ax1.transAxes,va="top",fontsize=8.5,color="#e2e8f0",
                 fontfamily="monospace",bbox=dict(boxstyle="round",facecolor="#0d1f3c",edgecolor="#1d4ed8",lw=1.2))
        ax1.set_title("VQC Circuit", fontsize=11, color="#60a5fa", pad=8)

        qaoa_txt = f"QAOA (p={p_layers} layers)\n{'='*35}\n\nInit: H^{n_qubits} -> |+>^{n_qubits}\n"
        for l in range(p_layers):
            qaoa_txt += f"\nLayer {l+1}:\n  U_C(g{l+1}): ZZ rotations [cost]\n  U_B(b{l+1}):  Rx rotations [mixer]"
        qaoa_txt += "\n\nMeasure: argmax bitstring\nOptimise: COBYLA outer loop\nRef: Farhi et al. (2014)"
        ax2.text(0.05,0.95,qaoa_txt,transform=ax2.transAxes,va="top",fontsize=8.5,color="#e2e8f0",
                 fontfamily="monospace",bbox=dict(boxstyle="round",facecolor="#1a1a0d",edgecolor="#f59e0b",lw=1.2))
        ax2.set_title("QAOA Circuit", fontsize=11, color="#f59e0b", pad=8)

        qksvm_txt = f"Quantum Kernel SVM\n{'='*35}\n\nKernel: K(x,x') = |<phi(x)|phi(x')>|^2\n\nFeature map: ZZFeatureMap\n  q0: -[H]-[ZZ(x0,x1)]-\n  q1: -[H]-[ZZ(x0,x1)]-\n  ...\n\nClassical SVM with quantum kernel matrix\nC=1.0, max_iter=500\nAdvantage: O(2^n) Hilbert space\n\nNote: Qiskit required for live circuit."
        ax3.text(0.05,0.95,qksvm_txt,transform=ax3.transAxes,va="top",fontsize=8.5,color="#e2e8f0",
                 fontfamily="monospace",bbox=dict(boxstyle="round",facecolor="#1a0d1a",edgecolor="#FF6B6B",lw=1.2))
        ax3.set_title("Quantum Kernel SVM (NEW)", fontsize=11, color="#FF6B6B", pad=8)

        fig.suptitle("Quantum Circuit Diagrams: VQC | QAOA | Quantum Kernel SVM",
                     fontsize=12, fontweight="bold", color="#e2e8f0")
        fig.tight_layout(); return self._save(fig, "quantum_circuits.png")

    def accuracy_leaderboard(self, results):
        self._s()
        lb = sorted(results.items(), key=lambda x: x[1].get("accuracy",0) or 0, reverse=True)
        names = [r[0] for r in lb]; accs = [r[1].get("accuracy",0) or 0 for r in lb]
        colors = [COLORS.get(n,"#888888") for n in names]
        fig, ax = plt.subplots(figsize=(10,6), facecolor="#0f1117")
        bars = ax.barh(names[::-1], accs[::-1], color=colors[::-1], edgecolor="none")
        ax.axvline(0.95, color="#34d399", ls="--", lw=1.5, label="95% target")
        for b,v in zip(bars, accs[::-1]):
            ax.text(v+0.002, b.get_y()+b.get_height()/2, f"{v:.3f}",
                    va="center", fontsize=9, color="#e2e8f0", fontweight="bold")
        ax.set_xlim(0,1.08); ax.set_xlabel("Accuracy")
        ax.set_title("Model Leaderboard - Accuracy Ranking", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); fig.tight_layout()
        return self._save(fig, "leaderboard.png")

    def algorithm_radar(self, results):
        self._s()
        metrics = ["Accuracy","Kappa","CV Mean","Speed*"]
        speed   = {"ExtraTrees":0.92,"XGBoost":0.80,"LightGBM":0.88,"CatBoost":0.78,
                   "Voting Ensemble":0.60,"Random Forest":0.82,"Gradient Boosting":0.55,
                   "SVM":0.85,"Naive Bayes":1.0,"KNN":0.95,"VQC":0.25,"QAOA":0.20,
                   "Quantum Kernel SVM":0.30}
        angles  = np.linspace(0,2*np.pi,len(metrics),endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(8,8), subplot_kw={"polar":True}, facecolor="#0f1117")
        ax.set_facecolor("#0a0e1a")
        for name, res in results.items():
            a   = res.get("accuracy") or 0
            k   = res.get("kappa")    or 0
            cv  = res.get("cv_mean")  or 0
            sp  = speed.get(name,0.5)
            vals= [a,k,cv,sp]; vals += vals[:1]
            color = COLORS.get(name,"#888888")
            ax.plot(angles,vals,"o-",lw=2,color=color,label=name,alpha=0.85)
            ax.fill(angles,vals,alpha=0.07,color=color)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics,fontsize=10,color="#e2e8f0")
        ax.set_ylim(0,1.05); ax.grid(color="#1e293b",ls="--",alpha=0.5)
        ax.set_title("Algorithm Comparison Radar",fontsize=11,color="#e2e8f0",pad=20,fontweight="bold")
        ax.legend(loc="upper right",bbox_to_anchor=(1.4,1.15),fontsize=7.5,framealpha=0.3)
        fig.tight_layout(); return self._save(fig,"algorithm_radar.png")
