"""geometri.py ile üretilen NACA 2412 koordinatlarını AeroSandbox'ınkiyle karşılaştırır."""
import aerosandbox as asb
import numpy as np

from geometri import naca4


def yuzeylere_ayir(koord):
    # Selig sırasında en küçük x hücum kenarıdır; öncesi üst, sonrası alt yüzey
    i = np.argmin(koord[:, 0])
    ust = koord[: i + 1][::-1]  # x artan sıraya çevir
    alt = koord[i:]
    return ust, alt


uretilen = naca4(0.02, 0.4, 0.12)
referans = asb.Airfoil("naca2412").coordinates

b_ust, b_alt = yuzeylere_ayir(uretilen)
r_ust, r_alt = yuzeylere_ayir(referans)

x = np.linspace(0, 1, 400)
fark_ust = np.abs(np.interp(x, b_ust[:, 0], b_ust[:, 1]) - np.interp(x, r_ust[:, 0], r_ust[:, 1]))
fark_alt = np.abs(np.interp(x, b_alt[:, 0], b_alt[:, 1]) - np.interp(x, r_alt[:, 0], r_alt[:, 1]))

print("nokta sayısı:", len(uretilen), "| referans:", len(referans))
print("ilk ve son nokta:", uretilen[0], uretilen[-1])
print(f"en büyük y farkı, üst: {fark_ust.max():.2e}  alt: {fark_alt.max():.2e}")
