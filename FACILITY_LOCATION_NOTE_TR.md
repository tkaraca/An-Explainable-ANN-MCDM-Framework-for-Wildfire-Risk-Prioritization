# Müdahale Merkezi Lokasyon Seçimi Notu

## 1. Mevcut paket ne yapıyor?

Mevcut ana paket doğrudan **müdahale merkezi seçmiyor**. Önce **risk yüzeyi / risk haritası** üretiyor.

Bu yüzden ana modelde:
- **alternatifler = grid hücreleri**
- **kriterler = çevresel, topografik ve meteorolojik değişkenler**
- **çıktı = yangın riski / öncelik puanı**

## 2. Müdahale merkezi seçimi için doğru kurgu nedir?

Merkez yeri seçimi için iki aşama önerilir.

### Aşama 1: Risk hücrelerini bul
YSA + MCDM ile her hücreye bir risk puanı ver.

### Aşama 2: Merkez adaylarını yerleştir
Bu riskli hücreleri **talep noktası** gibi düşün.
Sonra merkezleri şu yöntemlerden biriyle seç:

- p-median
- maximal covering location problem (MCLP)
- location-allocation
- ağırlıklı kümeleme tabanlı heuristik

## 3. Bu pakette eklenen merkez aday scripti ne yapıyor?

`propose_centers.py` doğrudan optimizasyon çözmüyor. Bunun yerine:
- en riskli hücreleri alıyor,
- risk puanını ağırlık kabul ediyor,
- ağırlıklı KMeans ile kümeliyor,
- her küme için bir öneri merkez noktası veriyor.

Bu çıktı:
- keşif analizi,
- ön fizibilite,
- makalede ek deney,
- saha uzmanına aday bölge gösterme

için yararlıdır.

## 4. Bu çıktı makalede nasıl sunulur?

Şu biçimde sunabilirsin:

1. Hücre bazlı risk haritası oluşturuldu.
2. En riskli 5000 hücre talep kümesi olarak seçildi.
3. Risk puanları talep ağırlığı kabul edildi.
4. K=10 için ağırlıklı kümeleme ile aday müdahale merkezi lokasyonları üretildi.
5. Bu merkezler daha sonra operasyonel, yol erişimi ve kurum altyapısı açısından uzman değerlendirmesine sunuldu.

## 5. Bilimsel dürüstlük notu

Bu heuristik merkezi aday üretimi, tam bir ağ / yol / erişim optimizasyonu değildir. Yol ağı, seyahat süresi, kapasite ve mevcut istasyonlar da girerse daha güçlü bir location-allocation modeli gerekir.

Yani:
- **mevcut ana çalışma = risk sıralaması**
- **ek script = merkez aday önerisi**
- **nihai lokasyon seçimi = ayrı optimizasyon problemi**
