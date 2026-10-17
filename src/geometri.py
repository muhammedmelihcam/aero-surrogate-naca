"""NACA 4 hane kanat profili koordinatları.

m, p, t kord oranı olarak verilir. NACA 2412 için m = 0.02, p = 0.4, t = 0.12.
"""
import numpy as np


def kalinlik(x, t):
    # Kapalı art kenarlı standart kalınlık dağılımı (son katsayı -0.1036)
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                    + 0.2843 * x**3 - 0.1036 * x**4)


def orta_cizgi(x, m, p):
    """Orta çizgi yc ve eğimi dyc/dx."""
    yc = np.zeros_like(x)
    egim = np.zeros_like(x)
    if m == 0:
        return yc, egim

    on = x < p
    yc[on] = m / p**2 * (2 * p * x[on] - x[on] ** 2)
    egim[on] = 2 * m / p**2 * (p - x[on])

    arka = ~on
    yc[arka] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[arka] - x[arka] ** 2)
    egim[arka] = 2 * m / (1 - p) ** 2 * (p - x[arka])
    return yc, egim


def naca4(m, p, t, n=100):
    """Selig sırasında koordinatlar: üst yüzey art kenardan hücum kenarına,
    sonra alt yüzey hücum kenarından art kenara. (2n - 1, 2) dizi döner."""
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))  # kosinüs aralığı: uçlarda sık nokta

    yt = kalinlik(x, t)
    yc, egim = orta_cizgi(x, m, p)
    theta = np.arctan(egim)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Hücum kenarı noktası iki yüzeyde ortak, alt yüzeyde bir kez atlanır
    ust = np.column_stack([xu[::-1], yu[::-1]])
    alt = np.column_stack([xl[1:], yl[1:]])
    return np.vstack([ust, alt])
