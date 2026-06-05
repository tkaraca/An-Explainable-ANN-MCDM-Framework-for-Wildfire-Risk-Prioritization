# Comparative ANN + MCDM Package for the Muğla Wildfire Dataset (v3)

This package builds a **comparative MLP (ANN) + multi-MCDM ranking workflow** on `Mugla_Orman_Yangini_Data.csv`.

Compared with previous versions, this release adds:
- **multiple weighting sources**: Permutation Importance, SHAP, equal, entropy, and CRITIC
- **multiple ranking approaches**: direct MLP probability, TOPSIS, and VIKOR
- **comparison tables** at both test-subset and all-grid levels
- **rank-correlation** and **top-N overlap** outputs across methods
- **full-grid scoring** for selected method combinations
- **response-center candidate generation** from wildfire risk scores

---

## 1) The most important conceptual distinction

### What is the “alternative” in the main pipeline?
In the main pipeline, the **alternative = a grid cell / one data row**.

In other words, the package first:
- learns wildfire risk for each grid cell,
- derives criterion weights from the ANN or from objective weighting schemes,
- ranks cells using TOPSIS / VIKOR.

### Are response centers the alternatives?
**No.**
Response-center location is a **second-stage problem**.

The workflow in this package is:
1. **Generate a wildfire risk map** → alternatives = grid cells
2. **Derive center candidates** → select clustered representative points from the highest-risk cells

For this reason, `propose_centers.py` runs **after** the main risk model.

---

## 2) Which weighting sources are available in this version?

### a) `pi`
After training the MLP, **Permutation Importance** is computed on the validation set.
This is the **primary recommended weighting source**.

### b) `shap`
Sample-based **global SHAP importance** is computed for the MLP.
This is included mainly for **robustness analysis**.

### c) `equal`
All criteria receive equal weight.
This is the most basic **baseline**.

### d) `entropy`
Generates objective weights based on the information diversity / dispersion of each criterion in the data.

### e) `critic`
Generates objective weights using both criterion variance and inter-criterion correlation.

---

## 3) Which ranking scores are available?

### `score_probability_mlp`
The direct wildfire probability output of the ANN.

### `score_topsis_<weight_source>`
A TOPSIS-based wildfire priority score computed using the selected weighting source.
Example: `score_topsis_pi`

### `score_vikor_<weight_source>`
A VIKOR-based wildfire priority score computed using the selected weighting source.
Example: `score_vikor_pi`

---

## 4) Recommended core method combinations

Recommended core combinations for the paper:
- **MLP probability ranking** → `score_probability_mlp`
- **PI-weighted TOPSIS** → `score_topsis_pi`
- **PI-weighted VIKOR** → `score_vikor_pi`
- **SHAP-weighted TOPSIS** → `score_topsis_shap`
- **SHAP-weighted VIKOR** → `score_vikor_shap`

Additional baselines are also compared on the test subset:
- `score_topsis_equal`
- `score_topsis_entropy`
- `score_topsis_critic`
- `score_vikor_equal`
- `score_vikor_entropy`
- `score_vikor_critic`

By default, all-grid scoring is limited to selected combinations in order to keep runtime manageable.
You can also score **all method combinations** if needed.

---

## 5) Columns excluded because of leakage risk

The following columns are **excluded from the predictive model**:
- `brightness`
- `scan`
- `track`
- `acq_date`
- `acq_time`
- `satellite`
- `instrument`
- `confidence`
- `version`
- `bright_t31`
- `frp`
- `daynight`
- `type`

---

## 6) Clean criteria used in the model

- `Biyokutle`
- `NDVI`
- `NDWI`
- `NDMI`
- `Yukselti`
- `Egim`
- `Yerlesim_Uzaklik`
- `Yol_Uzaklik`
- `Toprak_Kil`
- `Toprak_Nem`
- `Toplam_Yagis`
- `Toplam_Buharlasma`
- `Elektrik_Hatti_Var`
- `Sicaklik`
- `Nem`
- `Ruzgar`
- `Yagis`
- `Guneylilik` (derived from `Baki` / aspect)
- `BitkiRiskSkoru` (derived from `Bitki_Turu` using **training data only**)

By default, `Boylam` and `Enlem` are **not included as predictive features**; they are retained only for spatial identity, splitting, and mapping.

---

## 7) File structure

### Main scripts
- `run_pipeline.py` → comparative ANN + MCDM workflow
- `propose_centers.py` → response-center candidate generation from wildfire risk scores

### Modules
- `src/data_utils.py` → data loading, cleaning, splitting, and feature engineering
- `src/modeling.py` → MLP, PI, SHAP, equal, entropy, and CRITIC weighting
- `src/mcdm.py` → TOPSIS and VIKOR
- `src/comparison.py` → comparative method-performance tables
- `src/location.py` → response-center candidate generation

### Convenience run scripts
- `run_quick_compare.sh`
- `run_full_compare_spatial.sh`
- `run_centers_from_pi_topsis.sh`
- `run_quick_compare.ps1`
- `run_full_compare_spatial.ps1`
- `run_centers_from_pi_topsis.ps1`

### Notes / documentation files
- `RUN_WINDOWS_TR.md`
- `DATA_AUDIT.md`
- `PAPER_OUTLINE_TR.md`
- `FACILITY_LOCATION_NOTE_TR.md`
- `METHOD_COMPARISON_NOTE_TR.md`

> Note: some supporting notes still use Turkish filenames, even though the main README is now in English.

---

## 8) Recommended execution order

### Step 1 — Install requirements
```bash
pip install -r requirements.txt
```

### Step 2 — Run a quick sanity check
```bash
python run_pipeline.py \
  --csv ./Mugla_Orman_Yangini_Data.csv \
  --output-dir outputs_quick_compare \
  --split random \
  --max-rows 200000 \
  --top-n 200
```

### Step 3 — Run the full comparative spatial experiment
```bash
python run_pipeline.py \
  --csv ./Mugla_Orman_Yangini_Data.csv \
  --output-dir outputs_full_compare \
  --split spatial \
  --top-n 1000
```

### Step 4 — Optionally score all methods on the full grid
```bash
python run_pipeline.py \
  --csv ./Mugla_Orman_Yangini_Data.csv \
  --output-dir outputs_full_all_methods \
  --split spatial \
  --top-n 1000 \
  --all-grid-all-methods
```

### Step 5 — Generate response-center candidates
```bash
python propose_centers.py \
  --ranked-csv ./outputs_full_compare/all_grid_scored.csv \
  --k 10 \
  --top-n 5000 \
  --weight-col score_topsis_pi
```

---

## 9) Main output files

### Basic quality and data-audit outputs
- `prototype_data_audit.json`
- `missing_profile.csv`
- `leakage_report.csv`
- `split_summary.csv`
- `prototype_metrics.json`
- `run_metadata.json`

### Weight files
- `feature_directions.csv`
- `feature_weights_all_sources.csv`
- `feature_weights_pivot.csv`
- `weights/weights_pi.csv`
- `weights/weights_shap.csv`
- `weights/weights_equal.csv`
- `weights/weights_entropy.csv`
- `weights/weights_critic.csv`
- `feature_schema_all_sources.csv`

### Comparison files
- `test_ranked_comparison.csv`
- `method_comparison_test.csv`
- `method_spearman_test.csv`
- `method_topn_overlap_test.csv`

### Full-grid scoring outputs
- `all_grid_scored.csv`
- `method_comparison_all_grid.csv`
- `risk_blocks_1000m.csv`
- `all_grid_views/top_<N>_<score_column>.csv`

### Response-center candidate outputs
- files such as `proposed_response_centers_k10_score_topsis_pi.csv`

---

## 10) How to read `all_grid_scored.csv`

Each row in this file corresponds to **one grid cell**.

Important columns include:
- `Boylam`
- `Enlem`
- `YANGIN_DURUMU`
- `score_probability_mlp`
- `rank_probability_mlp`
- `score_topsis_pi`
- `rank_topsis_pi`
- `score_vikor_pi`
- `rank_vikor_pi`
- `score_topsis_shap`
- `rank_topsis_shap`
- ...

### Interpretation
- `score_probability_mlp` → the direct ANN probability score
- `score_topsis_pi` → the primary recommended MCDM wildfire-priority score
- `score_vikor_pi` → an alternative compromise-ranking score
- in all `rank_*` columns, **1 = highest priority / highest wildfire risk**

---

## 11) Recommended comparison set for the paper

For the **main comparison table**:
- `MLP_probability`
- `TOPSIS__pi`
- `VIKOR__pi`
- `TOPSIS__shap`
- `VIKOR__shap`

For an **additional baseline table**:
- `TOPSIS__equal`
- `TOPSIS__entropy`
- `TOPSIS__critic`
- `VIKOR__equal`
- `VIKOR__entropy`
- `VIKOR__critic`

This gives a strong robustness analysis without making the core manuscript overly crowded.

---

## 12) Logic of response-center candidate generation

`propose_centers.py` performs the following steps:
1. selects the top-`n` highest-risk grid cells according to the chosen score column
2. partitions them into `k` clusters using **weighted KMeans**
3. finds the weighted center of each cluster
4. snaps that center to the nearest real grid cell

In other words, this script is **not a full location-allocation optimizer**; it is a **risk-based candidate-center generation module**.

---

## 13) Which score should I use for center generation?

Recommended priority order:
1. `score_topsis_pi`
2. `score_vikor_pi`
3. `score_topsis_shap`
4. `score_probability_mlp`

Recommended first run:
```bash
python propose_centers.py \
  --ranked-csv ./outputs_full_compare/all_grid_scored.csv \
  --k 10 \
  --top-n 5000 \
  --weight-col score_topsis_pi
```

---

## 14) Final note

This package provides a **comparative hybrid decision-support backbone** suitable for a journal article.
The next natural step for real response-center siting would be to extend it with explicit optimization models such as:
- p-median
- MCLP
- set covering

In other words, the current package produces a strong risk-prioritization and candidate-generation framework, while full facility-location optimization can be added in the next stage.
