# Veri Denetimi Özeti

## Temel bulgular

- Hedef değişken: `YANGIN_DURUMU`
- Yorum: `1 = yangın`, `0 = yangın yok`
- Koordinat çiftleri tekrarsız görünüyor.
- `Boylam` ve `Enlem` kolonlarında minimum pozitif fark yaklaşık **100 m**; veri hücre bazlı grid mantığına uygun.

## Kritik leakage alanları

Aşağıdaki kolonlar yangın-sonrası / sensör-tespit bilgisi taşıdığı için feature olarak kullanılmamalı:

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

## Temizlik notları

- `Toprak_Nem`, `Toplam_Yagis`, `Toplam_Buharlasma`, `Yerlesim_Uzaklik`, `Yol_Uzaklik` kolonlarında `-9999` kodlu eksik değerler bulunuyor.
- `Baki` dairesel bir değişken olduğu için doğrudan bırakılmadı; `Guneylilik` türetildi.
- `Bitki_Turu` doğrudan kategorik olarak bırakılmadı; eğitim verisinden türetilen smoothed `BitkiRiskSkoru` eklendi.

## Bu veri ile ne yapılır?

### Güçlü kullanım
- Yangın risk sınıflaması
- Hücre bazlı risk önceliklendirmesi
- Bölgesel risk haritası
- Risk haritasından müdahale merkezi adayları türetme

### Ayrı model gerektiren kullanım
- Gerçek yol süreleri ile istasyon yer seçimi
- Müdahale kapasitesi optimizasyonu
- Mevcut istasyon ağının kapsama analizi
