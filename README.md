# Muğla Orman Yangını Verisi için Karşılaştırmalı YSA + MCDM Paketi (v3)

Bu paket, `Mugla_Orman_Yangini_Data.csv` üzerinde **MLP (YSA) + çoklu MCDM sıralama** akışı kurar.

Bu sürümde önceki sürümlere ek olarak şunlar vardır:
- **birden fazla ağırlık kaynağı**: Permutation Importance, SHAP, equal, entropy, CRITIC
- **birden fazla sıralama yaklaşımı**: saf MLP olasılığı, TOPSIS, VIKOR
- **karşılaştırma tabloları**: test ve all-grid düzeyinde yöntem performansı
- **yöntemler arası sıra korelasyonu** ve **top-N örtüşme** çıktıları
- **tüm grid** üzerinde seçilmiş yöntem kombinasyonları için skor üretimi
- risk skorlarından **müdahale merkezi adayı** üretimi

---

## 1) En önemli kavramsal ayrım

### Ana pipeline'da alternatif nedir?
Ana pipeline'da **alternatif = grid hücresi / veri satırıdır**.

Yani bu paket önce şunu yapar:
- her grid hücresi için yangın riski öğrenir,
- YSA'dan veya objektif yöntemlerden kriter ağırlıkları üretir,
- TOPSIS / VIKOR ile hücreleri risk sırasına koyar.

### Müdahale merkezleri alternatif miydi?
**Hayır.**
Müdahale merkezi lokasyonu, ikinci aşama bir problemdir.

Bu paketteki akış:
1. **Risk haritası üret** → alternatifler = grid hücreleri
2. **Merkez adayları türet** → riskli hücrelerden kümelenmiş temsilci noktalar seç

Bu yüzden `propose_centers.py`, ana risk modelinden **sonra** çalışır.

---

## 2) Bu sürümde hangi ağırlık kaynakları var?

### a) `pi`
MLP eğitildikten sonra validation set üzerinde **Permutation Importance** hesaplanır.
Bu, ana önerilen ağırlık kaynağıdır.

### b) `shap`
MLP için örneklem tabanlı **SHAP global importance** hesaplanır.
Bu, sağlamlık kontrolü için eklenmiştir.

### c) `equal`
Tüm kriterler eşit ağırlık alır.
Bu, temel baseline’dır.

### d) `entropy`
Kriterlerin veri içindeki bilgi çeşitliliğine göre objektif ağırlık üretir.

### e) `critic`
Kriter varyansı ve kriterler arası korelasyonu birlikte kullanır.

---

## 3) Bu sürümde hangi sıralama skorları var?

### `score_probability_mlp`
YSA’nın doğrudan verdiği yangın olasılığıdır.

### `score_topsis_<weight_source>`
Seçilen ağırlık kaynağı ile hesaplanan TOPSIS risk öncelik skorudur.
Örnek: `score_topsis_pi`

### `score_vikor_<weight_source>`
Seçilen ağırlık kaynağı ile hesaplanan VIKOR risk öncelik skorudur.
Örnek: `score_vikor_pi`

---

## 4) Varsayılan önerilen ana kombinasyonlar

Makale için önerilen çekirdek kombinasyonlar:
- **MLP olasılık sıralaması** → `score_probability_mlp`
- **PI ağırlıklı TOPSIS** → `score_topsis_pi`
- **PI ağırlıklı VIKOR** → `score_vikor_pi`
- **SHAP ağırlıklı TOPSIS** → `score_topsis_shap`
- **SHAP ağırlıklı VIKOR** → `score_vikor_shap`

Ek baseline’lar test setinde ayrıca karşılaştırılır:
- `score_topsis_equal`
- `score_topsis_entropy`
- `score_topsis_critic`
- `score_vikor_equal`
- `score_vikor_entropy`
- `score_vikor_critic`

Varsayılan all-grid scoring daha hızlı olması için sadece seçili kombinasyonları üretir.
Tüm kombinasyonları da istersen üretebilirsin.

---

## 5) Leakage nedeniyle dışlanan sütunlar

Aşağıdaki kolonlar modele alınmaz:
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

## 6) Kullanılan temiz kriterler

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
- `Guneylilik` (`Baki`'den türetilir)
- `BitkiRiskSkoru` (`Bitki_Turu`ndan yalnızca eğitim verisiyle türetilir)

Varsayılan ayarda `Boylam` ve `Enlem` feature olarak modele verilmez; sadece konumsal kimlik için tutulur.

---

## 7) Dosya yapısı

### Ana scriptler
- `run_pipeline.py` → karşılaştırmalı YSA + MCDM akışı
- `propose_centers.py` → risk skorlarından müdahale merkezi adayı üretimi

### Modüller
- `src/data_utils.py` → veri okuma, temizlik, split, feature engineering
- `src/modeling.py` → MLP, PI, SHAP, equal, entropy, CRITIC
- `src/mcdm.py` → TOPSIS ve VIKOR
- `src/comparison.py` → yöntem karşılaştırma tabloları
- `src/location.py` → merkez adayı üretimi

### Çalıştırma kolaylaştırıcı dosyalar
- `run_quick_compare.sh`
- `run_full_compare_spatial.sh`
- `run_centers_from_pi_topsis.sh`
- `run_quick_compare.ps1`
- `run_full_compare_spatial.ps1`
- `run_centers_from_pi_topsis.ps1`

### Not dosyaları
- `RUN_WINDOWS_TR.md`
- `DATA_AUDIT.md`
- `PAPER_OUTLINE_TR.md`
- `FACILITY_LOCATION_NOTE_TR.md`
- `METHOD_COMPARISON_NOTE_TR.md`

---

## 8) En pratik çalışma sırası

### Adım 1 — Paketleri kur
```bash
pip install -r requirements.txt
```

### Adım 2 — Hızlı deneme
```bash
python run_pipeline.py \
  --csv ./Mugla_Orman_Yangini_Data.csv \
  --output-dir outputs_quick_compare \
  --split random \
  --max-rows 200000 \
  --top-n 200
```

### Adım 3 — Asıl karşılaştırmalı tam koşu
```bash
python run_pipeline.py \
  --csv ./Mugla_Orman_Yangini_Data.csv \
  --output-dir outputs_full_compare \
  --split spatial \
  --top-n 1000
```

### Adım 4 — İstersen tüm yöntemleri tüm gridde de skorla
```bash
python run_pipeline.py \
  --csv ./Mugla_Orman_Yangini_Data.csv \
  --output-dir outputs_full_all_methods \
  --split spatial \
  --top-n 1000 \
  --all-grid-all-methods
```

### Adım 5 — Müdahale merkezi adayları üret
```bash
python propose_centers.py \
  --ranked-csv ./outputs_full_compare/all_grid_scored.csv \
  --k 10 \
  --top-n 5000 \
  --weight-col score_topsis_pi
```

---

## 9) Ana çıktı dosyaları

### Temel kalite ve audit
- `prototype_data_audit.json`
- `missing_profile.csv`
- `leakage_report.csv`
- `split_summary.csv`
- `prototype_metrics.json`
- `run_metadata.json`

### Ağırlık dosyaları
- `feature_directions.csv`
- `feature_weights_all_sources.csv`
- `feature_weights_pivot.csv`
- `weights/weights_pi.csv`
- `weights/weights_shap.csv`
- `weights/weights_equal.csv`
- `weights/weights_entropy.csv`
- `weights/weights_critic.csv`
- `feature_schema_all_sources.csv`

### Karşılaştırma dosyaları
- `test_ranked_comparison.csv`
- `method_comparison_test.csv`
- `method_spearman_test.csv`
- `method_topn_overlap_test.csv`

### Tüm grid skorları
- `all_grid_scored.csv`
- `method_comparison_all_grid.csv`
- `risk_blocks_1000m.csv`
- `all_grid_views/top_<N>_<score_column>.csv`

### Müdahale merkezi adayları
- `proposed_response_centers_k10_score_topsis_pi.csv` gibi dosyalar

---

## 10) `all_grid_scored.csv` nasıl okunur?

Bu dosyada her satır bir grid hücresidir.

Önemli sütunlar:
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

### Yorum
- `score_probability_mlp` → YSA’nın doğrudan olasılığı
- `score_topsis_pi` → önerilen ana MCDM risk puanı
- `score_vikor_pi` → alternatif uzlaşık sıralama puanı
- `rank_*` sütunlarında **1 = en yüksek öncelik / en riskli**

---

## 11) Makale için önerilen çekirdek karşılaştırma seti

Ana tablo için:
- `MLP_probability`
- `TOPSIS__pi`
- `VIKOR__pi`
- `TOPSIS__shap`
- `VIKOR__shap`

Ek tablo için:
- `TOPSIS__equal`
- `TOPSIS__entropy`
- `TOPSIS__critic`
- `VIKOR__equal`
- `VIKOR__entropy`
- `VIKOR__critic`

Makale ana omurgasını fazla dağıtmadan iyi bir sağlamlık analizi verir.

---

## 12) Müdahale merkezi adayı üretme mantığı

`propose_centers.py` şunu yapar:
1. seçilen skor sütununa göre en riskli `top-n` grid hücresini alır
2. bunları **ağırlıklı KMeans** ile `k` kümeye ayırır
3. her kümenin ağırlıklı merkezini bulur
4. bu merkezi en yakın gerçek grid hücresine snap eder

Yani bu script, tam location-allocation optimizasyonu değil; **risk tabanlı aday merkez üretimidir**.

---

## 13) Hangi skorla merkez adayı üretmeliyim?

Önerilen sıra:
1. `score_topsis_pi`
2. `score_vikor_pi`
3. `score_topsis_shap`
4. `score_probability_mlp`

İlk çalıştırma için önerilen komut:
```bash
python propose_centers.py \
  --ranked-csv ./outputs_full_compare/all_grid_scored.csv \
  --k 10 \
  --top-n 5000 \
  --weight-col score_topsis_pi
```

---

## 14) Not

Bu paket, makaleye uygun bir **karşılaştırmalı hibrit karar destek iskeleti** verir. 
Gerçek müdahale merkezi yer seçimi için bir sonraki doğal adım:
- p-median
- MCLP
- set covering

gibi optimizasyon modellerini eklemektir.
