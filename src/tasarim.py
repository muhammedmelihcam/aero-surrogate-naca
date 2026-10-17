"""Latin Hypercube ile eğitim ve kör test kanat kombinasyonları."""
import numpy as np
from scipy.stats import qmc

# Tasarım aralıkları: m, p, t, log10(Re)
ALT = [0.00, 0.2, 0.08, 5.3]
UST = [0.06, 0.6, 0.18, 6.3]
ALFALAR = np.arange(-6, 21, 1.0)  # -6 ile 20 derece, 27 değer

EGITIM_TOHUM = 1
TEST_TOHUM = 2


def kombinasyonlar(n, tohum, onek):
    ornek = qmc.LatinHypercube(d=4, seed=tohum).random(n)
    olcekli = qmc.scale(ornek, ALT, UST)
    return [
        {
            "kombinasyon": f"{onek}{i:03d}",
            "m": row[0],
            "p": row[1],
            "t": row[2],
            "Re": 10 ** row[3],  # Re logaritmik uzayda örneklendi
        }
        for i, row in enumerate(olcekli)
    ]


def egitim_kumesi():
    return kombinasyonlar(90, EGITIM_TOHUM, "E")


def test_kumesi():
    return kombinasyonlar(25, TEST_TOHUM, "T")


if __name__ == "__main__":
    egitim, test = egitim_kumesi(), test_kumesi()
    ortak_kimlik = {k["kombinasyon"] for k in egitim} & {k["kombinasyon"] for k in test}
    ortak_geometri = {(k["m"], k["p"], k["t"], k["Re"]) for k in egitim} & {
        (k["m"], k["p"], k["t"], k["Re"]) for k in test
    }
    print("eğitim:", len(egitim), "test:", len(test), "açı sayısı:", len(ALFALAR))
    print("ortak kimlik:", len(ortak_kimlik), "ortak geometri:", len(ortak_geometri))
    for ad, kume in (("eğitim", egitim), ("test", test)):
        dizi = np.array([[k["m"], k["p"], k["t"], np.log10(k["Re"])] for k in kume])
        print(ad, "min:", dizi.min(0).round(3), "max:", dizi.max(0).round(3))
