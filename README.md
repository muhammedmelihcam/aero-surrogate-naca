# Aerodinamik Vekil Model: NACA 4 Hane Kanat Profilleri

NACA 4 hane kanat profillerinin kaldırma (CL) ve sürükleme (CD) katsayılarını XFOIL'e göre çok daha hızlı tahmin eden vekil (surrogate) modeller. Modeller XFOIL ile üretilen bir veri tabanında eğitildi ve eğitimde hiç görülmemiş kanatlarla kör testte doğrusal enterpolasyonla karşılaştırıldı.

## Amaç

- Girdiler: kamber `m`, kamber konumu `p`, kalınlık `t`, Reynolds sayısı `Re`, hücum açısı `alfa`
- Çıktılar: `CL` ve `CD`
- Modeller: Gauss Süreç Regresyonu (GPR), Destek Vektör Regresyonu (SVR), çok katmanlı algılayıcı (MLP)
- Referans: doğrusal enterpolasyon

## Yöntem

_Doldurulacak._

## Veri

_Doldurulacak: eğitim ve test kanat sayısı, satır sayısı, elenen koşu sayısı._

## Sonuçlar

_Doldurulacak: kör test sonuç tablosu ve stall bölgesi (`alfa >= 12`) sonuçları._

## Grafikler

_Doldurulacak._

## Sınırlamalar

- XFOIL iki boyutlu, viskoz bir panel yöntemidir ve CFD'nin yerini tutmaz. Düşük Mach sayıları için geçerlidir.
- Stall bölgesinde XFOIL'in kendisi de daha az güvenilirdir (ayrılmış akış, yakınsama sorunları).
- Modeller yalnızca örneklenen tasarım aralığında geçerlidir. Kör test kanatları aynı aralığın içindedir; bu bir ekstrapolasyon testi değildir.
- Veri sentetiktir, gerçek ölçüm içermez.

## Yeniden üretme

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

XFOIL 6.99'u https://web.mit.edu/drela/Public/web/xfoil/XFOIL6.99.zip adresinden indirip `xfoil.exe` dosyasını `bin/xfoil.exe` olarak yerleştirin.

_Doldurulacak: veri üretimi, eğitim ve değerlendirme komutları._

## Klasör yapısı

```
src/        kaynak kodlar
data/       egitim.csv, kor_test.csv
results/    sonuç tabloları ve grafikler
bin/        xfoil.exe (depoya dahil değil)
```
