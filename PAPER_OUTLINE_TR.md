# Makale Taslağı

## Seçenek 1: Risk önceliklendirme odaklı başlık

**Muğla'da Orman Yangını Risk Önceliklendirmesi için Açıklanabilir Yapay Sinir Ağı ve Çoklu MCDM Tabanlı Hibrit Bir Karar Destek Modeli**

## Seçenek 2: İki aşamalı lokasyon başlığı

**Muğla'da Orman Yangını Müdahale Merkezi Adaylarının Belirlenmesi için YSA Tabanlı Risk Haritası, Çoklu MCDM Karşılaştırması ve Ağırlıklı Kümeleme Yaklaşımı**

---

## Kısa katkı cümlesi

Bu çalışma, yangın-sonrası sensör alanlarını dışlayarak çevresel ve meteorolojik kriterlerden öğrenen bir MLP modelinden türetilen veri tabanlı kriter ağırlıklarını çok kriterli sıralama yöntemlerine aktaran, TOPSIS ve VIKOR sonuçlarını objektif ağırlıklandırma ve saf YSA sıralamasıyla karşılaştıran, hücre bazlı yangın riskini önceliklendiren ve isteğe bağlı olarak müdahale merkezi aday bölgeleri üreten iki aşamalı bir karar destek çerçevesi önermektedir.

---

## Araştırma soruları

1. Leakage içermeyen çevresel ve meteorolojik kriterler kullanılarak yangın riski YSA ile güvenilir biçimde öğrenilebilir mi?
2. YSA'dan türetilen kriter önemleri MCDM tarafında anlamlı risk sıralaması üretiyor mu?
3. Aynı ağırlık yapısı altında TOPSIS ve VIKOR sıralamaları nasıl farklılaşıyor?
4. Permutation importance ve SHAP gibi farklı ağırlık kaynakları, risk önceliklendirmesini ne ölçüde değiştiriyor?
5. Elde edilen risk yüzeyi operasyonel önceliklendirme ve merkez adayı üretimi için kullanılabilir mi?

---

## Yöntem akışı

1. Ham veri denetimi
2. Leakage kolonlarının çıkarılması
3. `-9999` kodlu alanların temizlenmesi
4. `Baki -> Guneylilik` dönüşümü
5. `Bitki_Turu -> BitkiRiskSkoru` dönüşümü
6. Eğitim / validation / test ayrımı
7. MLP eğitimi
8. Veri tabanlı ağırlık üretimi
   - Permutation importance
   - SHAP
9. Objektif / baseline ağırlık üretimi
   - Equal
   - Entropy
   - CRITIC
10. Sıralama yöntemleri
   - Saf MLP olasılığı
   - TOPSIS
   - VIKOR
11. Test örneklemi üzerinde yöntem karşılaştırması
12. Tüm grid üzerinde risk haritası üretimi
13. İsteğe bağlı: yüksek riskli hücrelerden müdahale merkezi aday üretimi
14. Duyarlılık ve sağlamlık analizi

---

## Kullanılabilecek hipotezler

- **H1:** Leakage içermeyen çevresel ve meteorolojik kriterlerle eğitilen MLP, yangın riskini ayırt etmede anlamlı performans sağlar.
- **H2:** Veri tabanlı ağırlıklarla çalışan hibrit MCDM yöntemleri, eşit veya salt objektif ağırlıklı baseline’lara kıyasla daha yüksek risk yoğunlaşması sağlar.
- **H3:** Aynı ağırlık yapısı altında TOPSIS ve VIKOR benzer ama tam örtüşmeyen risk öncelik sıralamaları üretir.
- **H4:** SHAP ve permutation importance tabanlı ağırlıklar, baskın risk kriterlerinde genel olarak tutarlı sonuç verir.
- **H5:** Tüm grid üzerinde elde edilen yüksek risk yoğunlaşmaları, müdahale merkezi adaylarının belirlenmesinde kullanışlı talep kümeleri oluşturur.

---

## Denklemler

### MLP olasılık tahmini

\[
\hat{p}_i = f_\theta(x_i)
\]

### Kriter öneminden ağırlık üretimi

\[
w_j = \frac{I_j}{\sum_{k=1}^{m} I_k}
\]

Burada `I_j`, ilgili kriterin permutation importance veya ortalama mutlak SHAP değeridir.

### TOPSIS yakınlık katsayısı

\[
C_i = \frac{D_i^-}{D_i^+ + D_i^-}
\]

En yüksek `C_i`, en yüksek riskli hücreyi temsil eder.

### VIKOR uzlaşık sıralama skoru

\[
Q_i = v\frac{S_i - S^*}{S^- - S^*} + (1-v)\frac{R_i - R^*}{R^- - R^*}
\]

Bu projede risk önceliği için `1 - Q_i` skoru kullanılır; daha yüksek değer daha yüksek risk önceliğini gösterir.

---

## Sonuçlar bölümünde verilebilecek tablo ve şekiller

- Veri seti ve değişkenler
- Leakage kontrol özeti
- MLP performans metrikleri
- Ağırlık kaynaklarının karşılaştırması
- Yöntem karşılaştırma tablosu (`method_comparison_test.csv`)
- Yöntemler arası spearman korelasyonu
- En riskli hücreler ve top-N yoğunlaşmaları
- 1 km blok risk özeti
- Risk haritası
- Müdahale merkezi aday noktaları (opsiyonel)

---

## Tartışma başlıkları

- Neden sensör metadata alanları dışlandı?
- Neden koordinatlar varsayılan olarak kriter yapılmadı?
- Neden `BitkiRiskSkoru` yararlı oldu?
- Neden hem TOPSIS hem VIKOR kullanıldı?
- Neden PI ana yöntem, SHAP ise sağlamlık testi olarak seçildi?
- Hibrit MCDM’nin katkısı ne: açıklanabilir sıralama mı, operasyonel önceliklendirme mi?
- Risk haritasından merkez adayı çıkarmanın sınırları neler?

---

## Genişletme önerileri

1. PROMETHEE’yi block-level veya top-N alternatif kümesinde ekle.
2. SHAP ile yerel açıklamalar ve harita tabanlı feature attribution üret.
3. Hücre bazlı sıralamayı ilçe / işletme şefliği düzeyine aggregate et.
4. Yol ağı ve seyahat süresi verisi eklenirse p-median veya MCLP çöz.
5. Mevsimsellik için dönemsel modeller kur.
