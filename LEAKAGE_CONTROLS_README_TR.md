# Leakage Control Experiments

Bu script üç ek negatif kontrol üretir:

1. **Label permutation**
   - Train etiketleri karıştırılır.
   - Beklenen sonuç: ROC-AUC yaklaşık 0.50, PR-AUC yaklaşık prevalence.

2. **Coordinate-only baseline**
   - Sadece `Boylam` ve `Enlem` ile MLP eğitilir.
   - Beklenen sonuç: tam modele göre belirgin biçimde daha düşük performans.

3. **Deliberately leaky model comparison**
   - Güvenli feature setine ek olarak leaky sütunlar (ve bunların varlık göstergeleri) eklenir.
   - Beklenen sonuç: performansın yapay biçimde yükselmesi.

## Kullanım

### Tam spatial deney
```powershell
& C:/Users/Admin/.conda/envs/Ormanyangn/python.exe .\run_leakage_controls.py --csv ".\Mugla_Orman_Yangini_Data.csv" --output-dir outputs_leakage_controls --split spatial
```

### Hızlı test
```powershell
& C:/Users/Admin/.conda/envs/Ormanyangn/python.exe .\run_leakage_controls.py --csv ".\Mugla_Orman_Yangini_Data.csv" --output-dir outputs_leakage_controls_quick --split random --max-rows 200000
```

## Üretilen dosyalar
- `leakage_controls_summary.csv`
- `leakage_controls_summary.json`
- `leakage_controls_roc_pr.png`
- `baseline_safe_features.csv`
- `deliberately_leaky_features.csv`
- `leakage_controls_notes.json`
