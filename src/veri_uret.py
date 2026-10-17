"""XFOIL ile eğitim ve kör test veri tabanını üretir.

Çalıştırma: python src/veri_uret.py
Deneme (2 kanat, dosya yazmaz): python src/veri_uret.py --deneme
"""
import os

# Her işçi süreci kendi BLAS iş parçacıklarını açarsa bellek tükeniyor; XFOIL zaten tek çekirdek
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import aerosandbox as asb
import numpy as np
import pandas as pd

from geometri import naca4
from tasarim import ALFALAR, egitim_kumesi, test_kumesi

KOK = Path(__file__).resolve().parent.parent
XFOIL = str(KOK / "bin" / "xfoil.exe")
VERI = KOK / "data"

YUKARI = ALFALAR[ALFALAR >= 0]          # 0, 1, ..., 20
ASAGI = np.sort(ALFALAR[ALFALAR <= 0])[::-1]  # 0, -1, ..., -6


def tarama(kanat, Re, alfalar):
    """Tek bir XFOIL oturumunda verilen sırayla açı taraması."""
    xf = asb.XFoil(
        airfoil=kanat, Re=Re, mach=0.0, xfoil_command=XFOIL,
        timeout=60, max_iter=100,
    )
    for deneme in range(3):
        try:
            return xf.alpha(alfalar, start_at=None)
        except OSError as hata:
            # Windows'ta geçici klasör bazen kilitli kalıyor; aynı taramayı tekrar dene
            son_hata = hata
        except Exception as hata:
            # XFOIL çökmesi: tekrar denemek aynı sonucu verir
            son_hata = hata
            break
    print(f"tarama başarısız (Re={Re:.0f}, {alfalar[0]}..{alfalar[-1]}, "
          f"{deneme + 1} deneme): {type(son_hata).__name__}: {str(son_hata)[:150]}")
    return {"alpha": np.array([]), "CL": np.array([]), "CD": np.array([])}


def eslestir(sonuc, alfa):
    # XFOIL açıyı yuvarlayarak yazar, bu yüzden küçük bir toleransla eşleştirilir
    yakin = np.abs(np.asarray(sonuc["alpha"]) - alfa) < 0.01
    if not yakin.any():
        return np.nan, np.nan
    i = np.argmax(yakin)
    return float(sonuc["CL"][i]), float(sonuc["CD"][i])


def kanat_kos(komb):
    """Bir kombinasyon için iki tarama yapar, istenen her açı için bir satır döner."""
    kanat = asb.Airfoil(
        name=f"k_{uuid.uuid4().hex}",  # paralel koşuda geçici dosyalar çakışmasın
        coordinates=naca4(komb["m"], komb["p"], komb["t"]),
    )
    bas = time.perf_counter()
    yukari = tarama(kanat, komb["Re"], YUKARI)
    asagi = tarama(kanat, komb["Re"], ASAGI)
    sure = time.perf_counter() - bas

    satirlar = []
    for alfa in ALFALAR:
        kaynak = yukari if alfa >= 0 else asagi  # 0 derece yukarı taramadan alınır
        cl, cd = eslestir(kaynak, alfa)
        satirlar.append({**komb, "alfa": alfa, "CL": cl, "CD": cd})
    return satirlar, sure


def ayikla(df):
    """Yakınsamayan ya da fiziksel olmayan satırları atar, sayıları döner."""
    nan = df["CL"].isna() | df["CD"].isna()
    cd_negatif = ~nan & (df["CD"] <= 0)
    cl_buyuk = ~nan & (df["CL"].abs() > 3)
    elenen = nan | cd_negatif | cl_buyuk
    sayilar = {
        "istenen": int(len(df)),
        "yakinsamayan": int(nan.sum()),
        "cd_sifir_alti": int(cd_negatif.sum()),
        "cl_mutlak_3_ustu": int(cl_buyuk.sum()),
        "elenen_toplam": int(elenen.sum()),
        "kalan": int((~elenen).sum()),
    }
    return df[~elenen].reset_index(drop=True), sayilar


def kume_uret(kombinasyonlar, havuz):
    bas = time.perf_counter()
    sonuclar = list(havuz.map(kanat_kos, kombinasyonlar))
    duvar = time.perf_counter() - bas
    satirlar = [s for kanat_satirlari, _ in sonuclar for s in kanat_satirlari]
    xfoil_sure = sum(sure for _, sure in sonuclar)
    return pd.DataFrame(satirlar), xfoil_sure, duvar


def main():
    deneme = "--deneme" in sys.argv
    isci = max(1, (os.cpu_count() or 2) - 2)

    kumeler = {"egitim": egitim_kumesi(), "kor_test": test_kumesi()}
    if deneme:
        kumeler = {"egitim": kumeler["egitim"][:2]}

    kayit = {"isci_sayisi": isci}
    havuz = ProcessPoolExecutor(max_workers=isci)  # iki küme için tek havuz
    for ad, kombinasyonlar in kumeler.items():
        df, xfoil_sure, duvar = kume_uret(kombinasyonlar, havuz)
        temiz, sayilar = ayikla(df)
        sayilar["kanat_sayisi"] = len(kombinasyonlar)
        sayilar["xfoil_nokta_basi_s"] = xfoil_sure / len(df)
        sayilar["duvar_suresi_s"] = duvar
        kayit[ad] = sayilar
        print(ad, json.dumps(sayilar, ensure_ascii=False))

        if deneme:
            print(temiz.to_string(max_rows=60))
        else:
            VERI.mkdir(exist_ok=True)
            temiz.to_csv(VERI / f"{ad}.csv", index=False)
    havuz.shutdown()

    if not deneme:
        with open(VERI / "uretim_ozeti.json", "w", encoding="utf-8") as f:
            json.dump(kayit, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
