"""Sonuç grafikleri, results/ içine PNG olarak yazılır."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geometri import naca4
from hazirlik import SONUC, VERI

YONTEMLER = ["Enterpolasyon", "GPR", "SVR", "MLP"]
RENK = {"Enterpolasyon": "#8c8c8c", "GPR": "#1f77b4", "SVR": "#2ca02c", "MLP": "#d62728"}
# Grafik etiketleri İngilizce, sütun adları kodla aynı kalıyor
ETIKET = {"Enterpolasyon": "Interpolation", "GPR": "GPR", "SVR": "SVR", "MLP": "MLP"}


def dagilim(df):
    fig, eksenler = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True, sharey=True)
    sinir = [df["CL"].min() - 0.1, df["CL"].max() + 0.1]
    for ax, y in zip(eksenler, YONTEMLER):
        stall = df["alfa"] >= 12
        ax.scatter(df.loc[~stall, "CL"], df.loc[~stall, f"CL_{y}"], s=8, alpha=0.5,
                   color=RENK[y], label="alpha < 12")
        ax.scatter(df.loc[stall, "CL"], df.loc[stall, f"CL_{y}"], s=8, alpha=0.7,
                   color="black", marker="x", label="alpha >= 12")
        ax.plot(sinir, sinir, "k--", lw=0.8)
        ax.set_title(ETIKET[y])
        ax.set_xlabel("XFOIL CL")
    eksenler[0].set_ylabel("Predicted CL")
    eksenler[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(SONUC / "cl_tahmin_gercek.png", dpi=150)
    plt.close(fig)


def kanat_egrisi(df):
    # Stall tepesinin görünmesi için tepe CL'si en yüksek test kanadı seçilir
    kimlik = df.groupby("kombinasyon")["CL"].max().idxmax()
    k = df[df["kombinasyon"] == kimlik].sort_values("alfa")
    ilk = k.iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(k["alfa"], k["CL"], "ko-", lw=2, ms=4, label="XFOIL")
    for y in YONTEMLER:
        ax.plot(k["alfa"], k[f"CL_{y}"], "-", color=RENK[y], label=ETIKET[y])
    ax.axvspan(12, k["alfa"].max(), color="0.9", zorder=0)
    ax.set_xlabel("alpha [deg]")
    ax.set_ylabel("CL")
    ax.set_title(f"{kimlik}: m={ilk.m:.3f}, p={ilk.p:.2f}, t={ilk.t:.3f}, Re={ilk.Re:.2e}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SONUC / "cl_alfa_test_kanadi.png", dpi=150)
    plt.close(fig)


def rmse_cubuk(tablo):
    tablo = tablo.set_index("yontem").loc[YONTEMLER]
    x = np.arange(len(YONTEMLER))
    fig, eksenler = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (tum, stall, baslik) in zip(eksenler, [
        ("CL_RMSE", "stall_CL_RMSE", "CL RMSE"),
        ("logCD_RMSE", "stall_logCD_RMSE", "log10(CD) RMSE"),
    ]):
        ax.bar(x - 0.2, tablo[tum], 0.4, label="all test points", color="#1f77b4")
        ax.bar(x + 0.2, tablo[stall], 0.4, label="alpha >= 12", color="#ff7f0e")
        ax.set_xticks(x, [ETIKET[y] for y in YONTEMLER])
        ax.set_title(baslik)
        ax.grid(axis="y", alpha=0.3)
    eksenler[0].legend()
    fig.tight_layout()
    fig.savefig(SONUC / "rmse_karsilastirma.png", dpi=150)
    plt.close(fig)


def kanatlar(df):
    # Her kombinasyon için tek satır (geometri ve Re)
    return df.drop_duplicates("kombinasyon")[["kombinasyon", "m", "p", "t", "Re"]]


def tasarim_uzayi(egitim, test):
    e, t = kanatlar(egitim), kanatlar(test)
    ciftler = [("m", "p"), ("m", "t"), ("p", "t"), ("t", "Re")]
    fig, eksenler = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (a, b) in zip(eksenler, ciftler):
        ax.scatter(e[a], e[b], s=18, color="#1f77b4", label=f"training ({len(e)})")
        ax.scatter(t[a], t[b], s=30, color="#ff7f0e", marker="^", label=f"blind test ({len(t)})")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        if b == "Re":
            ax.set_yscale("log")
    eksenler[0].legend(fontsize=8)
    fig.suptitle("Latin Hypercube design: airfoil combinations")
    fig.tight_layout()
    fig.savefig(SONUC / "tasarim_uzayi.png", dpi=150)
    plt.close(fig)


def profiller(test):
    t = kanatlar(test)
    # Kamberi ve kalınlığı en farklı birkaç test kanadı
    secim = pd.concat([t.nsmallest(1, "m"), t.nlargest(1, "m"),
                       t.nsmallest(1, "t"), t.nlargest(1, "t")]).drop_duplicates("kombinasyon")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for _, k in secim.iterrows():
        xy = naca4(k.m, k.p, k.t)
        ax.plot(xy[:, 0], xy[:, 1], lw=1.5,
                label=f"{k.kombinasyon}: m={k.m:.3f}, p={k.p:.2f}, t={k.t:.3f}")
    ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SONUC / "ornek_profiller.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def aciya_gore_hata(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for y in YONTEMLER:
        hata = (df[f"CL_{y}"] - df["CL"]) ** 2
        rmse = np.sqrt(hata.groupby(df["alfa"]).mean())
        ax.plot(rmse.index, rmse.values, "o-", ms=3, color=RENK[y], label=ETIKET[y])
    ax.axvspan(12, df["alfa"].max(), color="0.9", zorder=0)
    ax.set_xlabel("alpha [deg]")
    ax.set_ylabel("CL RMSE")
    ax.set_title("Blind test CL error by angle of attack")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SONUC / "aciya_gore_hata.png", dpi=150)
    plt.close(fig)


def surukleme_polari(df):
    kimlik = df.groupby("kombinasyon")["CL"].max().idxmax()
    k = df[df["kombinasyon"] == kimlik].sort_values("alfa")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(k["CD"], k["CL"], "ko-", lw=2, ms=4, label="XFOIL")
    for y in YONTEMLER:
        ax.plot(k[f"CD_{y}"], k[f"CL_{y}"], "-", color=RENK[y], label=ETIKET[y])
    ax.set_xscale("log")
    ax.set_xlabel("CD (log)")
    ax.set_ylabel("CL")
    ax.set_title(f"Drag polar, test airfoil {kimlik}")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(SONUC / "surukleme_polari.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    tahmin = pd.read_csv(SONUC / "test_tahminleri.csv")
    dagilim(tahmin)
    kanat_egrisi(tahmin)
    rmse_cubuk(pd.read_csv(SONUC / "sonuclar.csv"))
    tasarim_uzayi(pd.read_csv(VERI / "egitim.csv"), pd.read_csv(VERI / "kor_test.csv"))
    profiller(pd.read_csv(VERI / "kor_test.csv"))
    aciya_gore_hata(tahmin)
    surukleme_polari(tahmin)
    print("grafikler yazıldı:", sorted(p.name for p in SONUC.glob("*.png")))
