"""GPR üzerinde Sobol hassasiyet analizi.

Girdiler tasarım aralıklarında düzgün dağılımlı kabul edilir (Re logaritmik uzayda).
Model ölçekli [0, 1] girdi aldığı için örnekleme doğrudan bu uzayda yapılır.

Çıktılar:
  results/sobol.csv   S1 ve ST indeksleri, %95 güven aralıklarıyla
  results/sobol.png   S1, ST ve GPR uzunluk ölçekleri yan yana
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analiz
from SALib.sample import sobol as sobol_ornek

from hazirlik import OZELLIKLER, SONUC, yukle
from modeller import GPR, HEDEF_ADLARI

N = 4096  # toplam model çağrısı N * (D + 2) = 28672
ETIKET = {"m": "m", "p": "p", "t": "t", "logRe": "log10(Re)", "alfa": "alpha"}


def main():
    v = yukle()
    gpr = GPR()
    gpr.egit(v["X_egitim"], v["Y_egitim"])

    problem = {"num_vars": len(OZELLIKLER), "names": OZELLIKLER,
               "bounds": [[0.0, 1.0]] * len(OZELLIKLER)}
    X = sobol_ornek.sample(problem, N, calc_second_order=False, seed=0)
    Y = gpr.tahmin(X)
    print("örnek sayısı:", len(X))

    satirlar = []
    for j, hedef in enumerate(HEDEF_ADLARI):
        s = sobol_analiz.analyze(problem, Y[:, j], calc_second_order=False, seed=0)
        for i, girdi in enumerate(OZELLIKLER):
            satirlar.append({"hedef": hedef, "girdi": girdi,
                             "S1": s["S1"][i], "S1_guven": s["S1_conf"][i],
                             "ST": s["ST"][i], "ST_guven": s["ST_conf"][i]})
    tablo = pd.DataFrame(satirlar)
    olcek = pd.DataFrame(gpr.uzunluk_olcekleri(), index=OZELLIKLER)
    tablo["uzunluk_olcegi"] = [olcek.loc[g, h] for h, g in zip(tablo["hedef"], tablo["girdi"])]
    tablo.to_csv(SONUC / "sobol.csv", index=False)
    print(tablo.round(3).to_string(index=False))
    for hedef in HEDEF_ADLARI:
        t = tablo[tablo["hedef"] == hedef]
        print(f"{hedef}: S1 toplamı = {t['S1'].sum():.3f}")

    ciz(tablo)


def ciz(tablo):
    fig, eksenler = plt.subplots(2, 2, figsize=(11, 7), gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(OZELLIKLER))
    for satir, (hedef, baslik) in enumerate([("CL", "CL"), ("logCD", "log10(CD)")]):
        t = tablo[tablo["hedef"] == hedef].set_index("girdi").loc[OZELLIKLER]
        ax = eksenler[satir, 0]
        ax.bar(x - 0.2, t["S1"], 0.4, yerr=t["S1_guven"], label="S1 (first order)", color="#1f77b4")
        ax.bar(x + 0.2, t["ST"], 0.4, yerr=t["ST_guven"], label="ST (total)", color="#ff7f0e")
        ax.set_xticks(x, [ETIKET[g] for g in OZELLIKLER])
        ax.set_title(f"Sobol indices, {baslik}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        ax = eksenler[satir, 1]
        ax.bar(x, t["uzunluk_olcegi"], 0.6, color="#7f7f7f")
        ax.set_xticks(x, [ETIKET[g] for g in OZELLIKLER])
        ax.set_title(f"GPR length scale, {baslik}")
        ax.set_ylabel("smaller = more sensitive")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(SONUC / "sobol.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
