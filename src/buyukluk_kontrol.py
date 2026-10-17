"""Veri üretim hattının büyüklük kontrolü: NACA 0012 ve 2412, Re = 1e6.

veri_uret.py ile aynı geometri ve aynı tarama fonksiyonlarını kullanır.
"""
import numpy as np

from veri_uret import kanat_kos

for ad, m, p, t in (("NACA 0012", 0.0, 0.4, 0.12), ("NACA 2412", 0.02, 0.4, 0.12)):
    satirlar, _ = kanat_kos({"kombinasyon": ad, "m": m, "p": p, "t": t, "Re": 1e6})
    alfa = np.array([s["alfa"] for s in satirlar])
    cl = np.array([s["CL"] for s in satirlar])
    cd = np.array([s["CD"] for s in satirlar])
    i = np.nanargmax(cl)
    print(f"{ad}: CL(0)={cl[alfa == 0][0]:.3f}  CL(5)={cl[alfa == 5][0]:.3f}  "
          f"CLmax={cl[i]:.3f} @ {alfa[i]:.0f} derece  en düşük CD={np.nanmin(cd):.5f}  "
          f"yakınsamayan={int(np.isnan(cl).sum())}/{len(cl)}")
