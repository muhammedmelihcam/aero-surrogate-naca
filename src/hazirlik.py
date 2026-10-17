"""Girdi ve çıktı hazırlığı.

Özellikler: m, p, t, log10(Re), alfa. Hepsi tasarım aralığına göre [0, 1]'e ölçeklenir.
Hedefler: CL ve log10(CD).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from tasarim import ALT, UST, ALFALAR

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data"
SONUC = KOK / "results"

OZELLIKLER = ["m", "p", "t", "logRe", "alfa"]
# Ölçek sınırları tasarımdan gelir, test verisine hiç bakılmaz
OLCEK_ALT = np.array(ALT + [ALFALAR.min()])
OLCEK_UST = np.array(UST + [ALFALAR.max()])


def olcekle(df):
    X = np.column_stack([df["m"], df["p"], df["t"], np.log10(df["Re"]), df["alfa"]])
    return (X - OLCEK_ALT) / (OLCEK_UST - OLCEK_ALT)


def hedefler(df):
    return np.column_stack([df["CL"], np.log10(df["CD"])])


def sizinti_kontrolu(egitim, test):
    ortak = set(egitim["kombinasyon"]) & set(test["kombinasyon"])
    geo = ["m", "p", "t", "Re"]
    ortak_geo = pd.merge(egitim[geo].drop_duplicates(), test[geo].drop_duplicates(), on=geo)
    assert not ortak, f"test kimlikleri eğitimde var: {ortak}"
    assert ortak_geo.empty, "aynı geometri iki kümede birden var"


def yukle():
    egitim = pd.read_csv(VERI / "egitim.csv")
    test = pd.read_csv(VERI / "kor_test.csv")
    sizinti_kontrolu(egitim, test)
    return {
        "egitim": egitim,
        "test": test,
        "X_egitim": olcekle(egitim),
        "Y_egitim": hedefler(egitim),
        "X_test": olcekle(test),
        "Y_test": hedefler(test),
        "gruplar": egitim["kombinasyon"].to_numpy(),
    }


if __name__ == "__main__":
    v = yukle()
    print("sızıntı kontrolü geçti")
    print("eğitim:", v["X_egitim"].shape, "test:", v["X_test"].shape)
    print("eğitim X min:", v["X_egitim"].min(0).round(3), "max:", v["X_egitim"].max(0).round(3))
    print("test X min:", v["X_test"].min(0).round(3), "max:", v["X_test"].max(0).round(3))
    print("log10(CD) aralığı:", v["Y_egitim"][:, 1].min().round(3), v["Y_egitim"][:, 1].max().round(3))
