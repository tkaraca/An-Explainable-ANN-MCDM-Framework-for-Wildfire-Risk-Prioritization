# Skor Sütunları Referansı

## Ana skorlar

### `score_probability_mlp`
MLP modelinin doğrudan yangın olasılığı.

### `score_topsis_pi`
Permutation importance ağırlıkları ile hesaplanan TOPSIS risk puanı.
Önerilen ana skor budur.

### `score_vikor_pi`
Permutation importance ağırlıkları ile hesaplanan VIKOR risk puanı.
Ana sağlamlık karşılaştırması için önerilir.

### `score_topsis_shap`
SHAP ağırlıkları ile hesaplanan TOPSIS risk puanı.

### `score_vikor_shap`
SHAP ağırlıkları ile hesaplanan VIKOR risk puanı.

---

## Baseline skorlar

### `score_topsis_equal`
Eşit ağırlıklı TOPSIS.

### `score_topsis_entropy`
Entropy ağırlıklı TOPSIS.

### `score_topsis_critic`
CRITIC ağırlıklı TOPSIS.

### `score_vikor_equal`
Eşit ağırlıklı VIKOR.

### `score_vikor_entropy`
Entropy ağırlıklı VIKOR.

### `score_vikor_critic`
CRITIC ağırlıklı VIKOR.

---

## Hangi sütunla merkez önerisi üretmeliyim?

Öncelik sırası:
1. `score_topsis_pi`
2. `score_vikor_pi`
3. `score_topsis_shap`
4. `score_probability_mlp`
