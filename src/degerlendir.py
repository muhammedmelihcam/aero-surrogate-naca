"""Kör test: dört yöntem aynı test satırlarında, aynı ölçütlerle.

Çıktılar results/ içine yazılır:
  sonuclar.csv              ölçüt tablosu
  test_tahminleri.csv       her test satırı için her yöntemin tahmini
  gpr_uzunluk_olcekleri.csv GPR'ın öğrendiği uzunluk ölçekleri
  svr_en_iyi.json           SVR ızgara aramasının seçtiği değerler
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from hazirlik import OZELLIKLER, SONUC, VERI, yukle
from modeller import GPR, MLP, Enterpolasyon, SVRModel

STALL_ALFA = 12


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def olcutler(Y, T, stall):
    cd, cd_t = 10 ** Y[:, 1], 10 ** T[:, 1]
    return {
        "CL_RMSE": rmse(Y[:, 0], T[:, 0]),
        "CL_MAE": float(mean_absolute_error(Y[:, 0], T[:, 0])),
        "CL_R2": float(r2_score(Y[:, 0], T[:, 0])),
        "logCD_RMSE": rmse(Y[:, 1], T[:, 1]),
        "CD_bagil_hata": float(np.mean(np.abs(cd_t - cd) / cd)),
        "stall_CL_RMSE": rmse(Y[stall, 0], T[stall, 0]),
        "stall_logCD_RMSE": rmse(Y[stall, 1], T[stall, 1]),
    }


def main():
    v = yukle()
    X, Y, Xt, Yt = v["X_egitim"], v["Y_egitim"], v["X_test"], v["Y_test"]
    stall = v["test"]["alfa"].to_numpy() >= STALL_ALFA
    bin_nokta = np.resize(Xt, (1000, Xt.shape[1]))  # tahmin süresi için 1000 nokta

    with open(VERI / "uretim_ozeti.json", encoding="utf-8") as f:
        xfoil_nokta = json.load(f)["egitim"]["xfoil_nokta_basi_s"]

    SONUC.mkdir(exist_ok=True)
    satirlar = []
    tahminler = v["test"][["kombinasyon", "m", "p", "t", "Re", "alfa", "CL", "CD"]].copy()

    for model in (Enterpolasyon(), GPR(), SVRModel(), MLP()):
        bas = time.perf_counter()
        model.egit(X, Y, v["gruplar"])
        egitim_s = time.perf_counter() - bas

        T = model.tahmin(Xt)
        if isinstance(model, Enterpolasyon):
            zarf_disi = model.zarf_disi_oran

        bas = time.perf_counter()
        model.tahmin(bin_nokta)
        tahmin_s = time.perf_counter() - bas

        satir = {"yontem": model.ad, **olcutler(Yt, T, stall),
                 "egitim_s": egitim_s, "tahmin_1000_s": tahmin_s}
        satirlar.append(satir)
        tahminler[f"CL_{model.ad}"] = T[:, 0]
        tahminler[f"CD_{model.ad}"] = 10 ** T[:, 1]
        print(model.ad, {k: round(d, 4) for k, d in satir.items() if k != "yontem"})

        if isinstance(model, Enterpolasyon):
            print(f"  zarf dışında kalan test noktası oranı: {zarf_disi:.1%}")
        if isinstance(model, GPR):
            olcekler = pd.DataFrame(model.uzunluk_olcekleri(), index=OZELLIKLER)
            olcekler.to_csv(SONUC / "gpr_uzunluk_olcekleri.csv")
            print("  çekirdek (CL):", model.modeller[0].kernel_)
            print("  uzunluk ölçekleri:\n", olcekler.round(3))
        if isinstance(model, SVRModel):
            with open(SONUC / "svr_en_iyi.json", "w", encoding="utf-8") as f:
                json.dump(model.en_iyi, f, indent=2)
            print("  SVR en iyi:", model.en_iyi)

    tablo = pd.DataFrame(satirlar)
    tablo["xfoil_1000_s"] = 1000 * xfoil_nokta
    tablo.to_csv(SONUC / "sonuclar.csv", index=False)
    tahminler.to_csv(SONUC / "test_tahminleri.csv", index=False)
    with open(SONUC / "ek_bilgi.json", "w", encoding="utf-8") as f:
        json.dump({"zarf_disi_oran": zarf_disi, "stall_test_satiri": int(stall.sum()),
                   "test_satiri": int(len(stall)), "xfoil_nokta_basi_s": xfoil_nokta}, f, indent=2)
    print(tablo.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
