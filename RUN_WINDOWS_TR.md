# Windows Kullanım Kılavuzu (Adım Adım)

Aşağıdaki komutlar PowerShell içindir.

## 1) Proje klasörüne gir
Örnek:
```powershell
cd C:\Users\Admin\Downloads\mugla_ysa_mcdm_project_v3\mugla_ysa_mcdm_project_v3
```

## 2) Gerekli paketleri kur
```powershell
& C:/Users/Admin/.conda/envs/Ormanyangn/python.exe -m pip install -r .\requirements.txt
```

## 3) Hızlı test koşusu
Bu adım her şeyin düzgün çalıştığını görmek içindir.
```powershell
& C:/Users/Admin/.conda/envs/Ormanyangn/python.exe .\run_pipeline.py --csv ".\Mugla_Orman_Yangini_Data.csv" --output-dir outputs_quick_compare --split random --max-rows 200000 --top-n 200
```

## 4) Asıl tam koşu
Bu, makale için kullanman gereken ana çalıştırmadır.
```powershell
& C:/Users/Admin/.conda/envs/Ormanyangn/python.exe .\run_pipeline.py --csv ".\Mugla_Orman_Yangini_Data.csv" --output-dir outputs_full_compare --split spatial --top-n 1000
```

## 5) İstersen tüm yöntemleri tüm gridde skorla
Bu daha uzun sürer ama bütün yöntem kombinasyonlarını all-grid düzeyinde de yazar.
```powershell
& C:/Users/Admin/.conda/envs/Ormanyangn/python.exe .\run_pipeline.py --csv ".\Mugla_Orman_Yangini_Data.csv" --output-dir outputs_full_all_methods --split spatial --top-n 1000 --all-grid-all-methods
```

## 6) Müdahale merkezi adayları üret
Önerilen ilk deneme:
```powershell
& C:/Users/Admin/.conda/envs/Ormanyangn/python.exe .\propose_centers.py --ranked-csv ".\outputs_full_compare\all_grid_scored.csv" --k 10 --top-n 5000 --weight-col score_topsis_pi
```

---

# Çıktıları nasıl kontrol edeceksin?

## A) Ana klasörde hangi dosyalar var?
```powershell
dir .\outputs_full_compare\
```

## B) Test karşılaştırma özeti
```powershell
Get-Content .\outputs_full_compare\method_comparison_test.csv -TotalCount 20
```

## C) All-grid karşılaştırma özeti
```powershell
Get-Content .\outputs_full_compare\method_comparison_all_grid.csv -TotalCount 20
```

## D) Ağırlık tabloları
```powershell
Get-Content .\outputs_full_compare\feature_weights_pivot.csv -TotalCount 30
```

## E) Ana all-grid skor dosyası
```powershell
Get-Content .\outputs_full_compare\all_grid_scored.csv -TotalCount 20
```

## F) En riskli hücrelerin ayrı dosyaları
```powershell
dir .\outputs_full_compare\all_grid_views\
```

## G) Merkez adayları
```powershell
Get-Content .\outputs_full_compare\proposed_response_centers_k10_score_topsis_pi.csv -TotalCount 20
```

---

# Çıktı dosyaları ne işe yarıyor?

## `prototype_metrics.json`
MLP modelinin temel başarı metriklerini verir.

## `feature_weights_pivot.csv`
Aynı feature için farklı ağırlık kaynaklarının yan yana karşılaştırmasını verir.

## `method_comparison_test.csv`
Örneklenmiş test setinde hangi yöntem daha iyi sıralama yapıyor, bunu gösterir.
Özellikle bak:
- `pr_auc`
- `roc_auc`
- `top_1pct_positive_rate`
- `top_5pct_positive_rate`
- `top_100_positive_rate`

## `method_comparison_all_grid.csv`
Tüm grid düzeyinde yöntem karşılaştırması.

## `test_ranked_comparison.csv`
Test örneklemindeki bütün hücreler ve bütün skor sütunları.

## `all_grid_scored.csv`
Tüm geçerli grid hücreleri için skorlar.
Bu dosya, haritalama ve merkez adayı üretiminin temel girdisidir.

## `risk_blocks_1000m.csv`
1 km blok düzeyinde özet risk skorları.

## `proposed_response_centers_k10_score_topsis_pi.csv`
Riskli alanlardan türetilmiş 10 merkez adayı.

---

# Hangi sütunla merkez önerisi üretmeliyim?

İlk sırada şunu kullan:
```text
score_topsis_pi
```

Daha sonra istersen şu alternatifleri de dene:
- `score_vikor_pi`
- `score_topsis_shap`
- `score_vikor_shap`
- `score_probability_mlp`

Örnek:
```powershell
& C:/Users/Admin/.conda/envs/Ormanyangn/python.exe .\propose_centers.py --ranked-csv ".\outputs_full_compare\all_grid_scored.csv" --k 10 --top-n 5000 --weight-col score_vikor_pi
```

---

# Sık görülen durumlar

## 1) `ConvergenceWarning`
MLP bazen tam yakınsamadan da kullanılabilir sonuç üretir. Bu fatal hata değildir.
İstersen iterasyonu artırabilirsin:
```powershell
& C:/Users/Admin/.conda/envs/Ormanyangn/python.exe .\run_pipeline.py --csv ".\Mugla_Orman_Yangini_Data.csv" --output-dir outputs_full_compare --split spatial --top-n 1000 --mlp-max-iter 250
```

## 2) SHAP yavaş çalışıyor
SHAP örneklem büyüklüğünü küçült:
```powershell
& C:/Users/Admin/.conda/envs/Ormanyangn/python.exe .\run_pipeline.py --csv ".\Mugla_Orman_Yangini_Data.csv" --output-dir outputs_fast_shap --split spatial --top-n 1000 --shap-background-size 30 --shap-explain-size 100 --shap-nsamples 100
```

## 3) Sadece hızlı bir PI karşılaştırması istiyorum
```powershell
& C:/Users/Admin/.conda/envs/Ormanyangn/python.exe .\run_pipeline.py --csv ".\Mugla_Orman_Yangini_Data.csv" --output-dir outputs_pi_only --split spatial --top-n 1000 --weight-sources pi,equal,entropy,critic --all-grid-combos probability,topsis:pi,vikor:pi
```

---

# En kısa önerilen çalışma sırası

1. `pip install -r .\requirements.txt`
2. hızlı test: `outputs_quick_compare`
3. tam koşu: `outputs_full_compare`
4. `method_comparison_test.csv` ve `method_comparison_all_grid.csv` incele
5. `score_topsis_pi` ile merkez adayı üret
