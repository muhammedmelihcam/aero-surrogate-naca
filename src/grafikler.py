"""Sonuç grafikleri, results/ içine PNG olarak yazılır."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hazirlik import SONUC

YONTEMLER = ["Enterpolasyon", "GPR", "SVR", "MLP"]
RENK = {"Enterpolasyon": "#8c8c8c", "GPR": "#1f77b4", "SVR": "#2ca02c", "MLP": "#d62728"}


def dagilim(df):
    fig, eksenler = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True, sharey=True)
    sinir = [df["CL"].min() - 0.1, df["CL"].max() + 0.1]
    for ax, y in zip(eksenler, YONTEMLER):
        stall = df["alfa"] >= 12
        ax.scatter(df.loc[~stall, "CL"], df.loc[~stall, f"CL_{y}"], s=8, alpha=0.5,
                   color=RENK[y], label="alfa < 12")
        ax.scatter(df.loc[stall, "CL"], df.loc[stall, f"CL_{y}"], s=8, alpha=0.7,
                   color="black", marker="x", label="alfa >= 12")
        ax.plot(sinir, sinir, "k--", lw=0.8)
        ax.set_title(y)
        ax.set_xlabel("XFOIL CL")
    eksenler[0].set_ylabel("Tahmin CL")
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
        ax.plot(k["alfa"], k[f"CL_{y}"], "-", color=RENK[y], label=y)
    ax.axvspan(12, k["alfa"].max(), color="0.9", zorder=0)
    ax.set_xlabel("alfa [derece]")
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
        ax.bar(x - 0.2, tablo[tum], 0.4, label="tüm test", color="#1f77b4")
        ax.bar(x + 0.2, tablo[stall], 0.4, label="alfa >= 12", color="#ff7f0e")
        ax.set_xticks(x, YONTEMLER)
        ax.set_title(baslik)
        ax.grid(axis="y", alpha=0.3)
    eksenler[0].legend()
    fig.tight_layout()
    fig.savefig(SONUC / "rmse_karsilastirma.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    tahmin = pd.read_csv(SONUC / "test_tahminleri.csv")
    dagilim(tahmin)
    kanat_egrisi(tahmin)
    rmse_cubuk(pd.read_csv(SONUC / "sonuclar.csv"))
    print("grafikler yazıldı:", sorted(p.name for p in SONUC.glob("*.png")))
