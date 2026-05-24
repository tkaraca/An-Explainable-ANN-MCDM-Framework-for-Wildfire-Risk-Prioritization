# Yöntem Karşılaştırma Notu

Bu sürümde karşılaştırma iki eksende yapılır:

## 1) Aynı ağırlık, farklı sıralama
Örnek:
- `score_topsis_pi`
- `score_vikor_pi`

Bu karşılaştırma şunu gösterir:
**Aynı bilgi yapısını farklı MCDM mantıklarıyla sıraladığımızda sonuç değişiyor mu?**

## 2) Aynı sıralama mantığı, farklı ağırlık kaynağı
Örnek:
- `score_topsis_pi`
- `score_topsis_shap`
- `score_topsis_equal`
- `score_topsis_entropy`
- `score_topsis_critic`

Bu karşılaştırma şunu gösterir:
**Karar kalitesi ağırlık kaynağına ne kadar duyarlı?**

## 3) Saf YSA ile hibrit yaklaşım karşılaştırması
Örnek:
- `score_probability_mlp`
- `score_topsis_pi`
- `score_vikor_pi`

Bu karşılaştırma şunu gösterir:
**MCDM katmanı eklemek gerçekten değer katıyor mu?**

---

# Önerilen yorum sırası

## Ana tablo
- `MLP_probability`
- `TOPSIS__pi`
- `VIKOR__pi`
- `TOPSIS__shap`
- `VIKOR__shap`

## Ek tablo
- `TOPSIS__equal`
- `TOPSIS__entropy`
- `TOPSIS__critic`
- `VIKOR__equal`
- `VIKOR__entropy`
- `VIKOR__critic`

---

# Hangi metriklere bakacaksın?

Öncelik sırası:
1. `pr_auc`
2. `top_1pct_positive_rate`
3. `top_5pct_positive_rate`
4. `top_100_positive_rate`
5. `roc_auc`

Yangın olayı dengesiz olduğu için, özellikle **PR-AUC** ve **top-N positive rate** daha anlamlıdır.

---

# Makalede kullanılabilecek kısa yorum örneği

"Karşılaştırma sonuçları, PI-ağırlıklı TOPSIS ve VIKOR yaklaşımlarının hem saf MLP olasılık sıralamasına hem de objektif/naif ağırlıklandırma tabanlı baseline’lara karşı daha yüksek risk yoğunlaşması sağladığını göstermektedir. Özellikle top-%1 ve top-%5 dilimlerinde pozitif olay yoğunluğunun artması, önerilen hibrit yapının müdahale önceliklendirmesi açısından yararlı olduğunu desteklemektedir."
