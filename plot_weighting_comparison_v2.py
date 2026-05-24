from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
TOP_N = 10
OUTPUT_FIG = BASE_DIR / "figure_weighting_comparison.png"

SEARCH_CANDIDATES = {
    "PI": [
        BASE_DIR / "weights_pi.csv",
        BASE_DIR / "weights" / "weights_pi.csv",
        BASE_DIR / "outputs_full_compare" / "weights" / "weights_pi.csv",
        BASE_DIR / "outputs_quick_compare" / "weights" / "weights_pi.csv",
    ],
    "SHAP": [
        BASE_DIR / "weights_shap.csv",
        BASE_DIR / "weights" / "weights_shap.csv",
        BASE_DIR / "outputs_full_compare" / "weights" / "weights_shap.csv",
        BASE_DIR / "outputs_quick_compare" / "weights" / "weights_shap.csv",
    ],
    "Entropy": [
        BASE_DIR / "weights_entropy.csv",
        BASE_DIR / "weights" / "weights_entropy.csv",
        BASE_DIR / "outputs_full_compare" / "weights" / "weights_entropy.csv",
        BASE_DIR / "outputs_quick_compare" / "weights" / "weights_entropy.csv",
    ],
    "CRITIC": [
        BASE_DIR / "weights_critic.csv",
        BASE_DIR / "weights" / "weights_critic.csv",
        BASE_DIR / "outputs_full_compare" / "weights" / "weights_critic.csv",
        BASE_DIR / "outputs_quick_compare" / "weights" / "weights_critic.csv",
    ],
}

FEATURE_LABELS = {
    "Sicaklik": "Temperature",
    "Ruzgar": "Wind",
    "Nem": "Humidity",
    "Yagis": "Rainfall",
    "BitkiRiskSkoru": "Vegetation risk score",
    "Biyokutle": "Biomass",
    "Yukselti": "Elevation",
    "Guneylilik": "Southness",
    "Egim": "Slope",
    "Toprak_Nem": "Soil moisture",
    "Toprak_Kil": "Soil clay",
    "Toplam_Yagis": "Cumulative rainfall",
    "Toplam_Buharlasma": "Cumulative evaporation",
    "NDVI": "NDVI",
    "NDWI": "NDWI",
    "NDMI": "NDMI",
    "Yerlesim_Uzaklik": "Distance to settlements",
    "Yol_Uzaklik": "Distance to roads",
    "Elektrik_Hatti_Var": "Electricity-line presence",
}


def resolve_file(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def read_weight_file(path, method_name):
    df = pd.read_csv(path)
    feature_col = find_column(df, ["feature", "criterion", "variable", "name", "Feature"])
    weight_col = find_column(df, ["weight", "value", "score", "Weight"])
    if feature_col is None or weight_col is None:
        raise ValueError(
            f"Could not detect feature/weight columns in {path.name}. Columns found: {list(df.columns)}"
        )
    out = df[[feature_col, weight_col]].copy()
    out.columns = ["feature", "weight"]
    out["method"] = method_name
    return out


all_dfs = []
resolved = {}
for method, candidates in SEARCH_CANDIDATES.items():
    path = resolve_file(candidates)
    if path is None:
        tried = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"Could not find {method} weights file. Tried:\n{tried}"
        )
    resolved[method] = path
    all_dfs.append(read_weight_file(path, method))

print("Resolved files:")
for method, path in resolved.items():
    print(f"- {method}: {path}")


df_all = pd.concat(all_dfs, ignore_index=True)
pivot = df_all.pivot_table(index="feature", columns="method", values="weight", fill_value=0.0)
for col in ["PI", "SHAP", "Entropy", "CRITIC"]:
    if col not in pivot.columns:
        pivot[col] = 0.0
pivot = pivot[["PI", "SHAP", "Entropy", "CRITIC"]]
pivot["max_weight"] = pivot.max(axis=1)
pivot = pivot.sort_values("max_weight", ascending=False)
top_df = pivot.head(TOP_N).drop(columns=["max_weight"]).copy()
top_df = top_df.sort_values(by=["PI", "SHAP"], ascending=False)

x = np.arange(len(top_df))
width = 0.2
fig, ax = plt.subplots(figsize=(14, 7))
methods = ["PI", "SHAP", "Entropy", "CRITIC"]
offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

for method, offset in zip(methods, offsets):
    ax.bar(x + offset, top_df[method].values, width=width, label=method)

labels = [FEATURE_LABELS.get(f, f) for f in top_df.index.tolist()]
ax.set_title("Comparative weighting patterns across PI, SHAP, Entropy, and CRITIC", fontsize=16, pad=14)
ax.set_ylabel("Weight", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha="right")
ax.legend(title="Weighting scheme")
ax.grid(axis="y", alpha=0.3)
fig.text(0.5, 0.01, "Top criteria selected by maximum weight across alternative weighting schemes.", ha="center", fontsize=11)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure saved to: {OUTPUT_FIG}")
