"""Kurulum kontrolü: NACA 2412, Re = 1e6, alfa = 5 derece için XFOIL koşusu.

Beklenen CL yaklaşık 0,7 ile 0,85 arasında.
"""
from pathlib import Path

import aerosandbox as asb

XFOIL = Path(__file__).resolve().parent.parent / "bin" / "xfoil.exe"

kanat = asb.Airfoil("naca2412")
xf = asb.XFoil(
    airfoil=kanat,
    Re=1e6,
    mach=0.0,
    xfoil_command=str(XFOIL),
    timeout=60,
    max_iter=100,
)
sonuc = xf.alpha(5)

print("anahtarlar:", list(sonuc.keys()))
print("alfa:", sonuc["alpha"])
print("CL:", sonuc["CL"])
print("CD:", sonuc["CD"])
