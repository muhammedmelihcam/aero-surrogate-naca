"""Dört yöntem: doğrusal enterpolasyon, GPR, SVR, MLP.

Her sınıf aynı arayüzü kullanır: egit(X, Y, gruplar) ve tahmin(X).
Y'nin iki sütunu var: CL ve log10(CD).
"""
import os

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ISCI = max(1, (os.cpu_count() or 2) - 2)
HEDEF_ADLARI = ["CL", "logCD"]


class Enterpolasyon:
    ad = "Enterpolasyon"

    def egit(self, X, Y, gruplar=None):
        self.dogrusal = LinearNDInterpolator(X, Y)
        self.en_yakin = NearestNDInterpolator(X, Y)

    def tahmin(self, X):
        Y = self.dogrusal(X)
        # Dış bükey zarfın dışı NaN döner, oralarda en yakın komşu kullanılır
        disarida = np.isnan(Y).any(axis=1)
        Y[disarida] = self.en_yakin(X[disarida])
        self.zarf_disi_oran = disarida.mean()
        return Y


class GPR:
    ad = "GPR"

    def __init__(self, alt_kume=750, tohum=0):
        self.alt_kume = alt_kume
        self.tohum = tohum

    def egit(self, X, Y, gruplar=None):
        rng = np.random.default_rng(self.tohum)
        secim = rng.choice(len(X), size=min(self.alt_kume, len(X)), replace=False)
        self.modeller = []
        for j in range(Y.shape[1]):
            cekirdek = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5)
                + WhiteKernel(1e-4, (1e-8, 1e-1))
            )
            # Hiperparametreler alt kümede öğrenilir...
            ayar = GaussianProcessRegressor(
                cekirdek, normalize_y=True, n_restarts_optimizer=2, random_state=self.tohum
            ).fit(X[secim], Y[secim, j])
            # ...sonra aynı çekirdekle tüm veriye optimizasyonsuz oturtulur
            tam = GaussianProcessRegressor(ayar.kernel_, normalize_y=True, optimizer=None).fit(X, Y[:, j])
            self.modeller.append(tam)

    def tahmin(self, X):
        return np.column_stack([m.predict(X) for m in self.modeller])

    def uzunluk_olcekleri(self):
        # Çekirdek yapısı: (Constant * Matern) + White, Matern'e k1.k2 ile ulaşılır
        return {ad: m.kernel_.k1.k2.length_scale for ad, m in zip(HEDEF_ADLARI, self.modeller)}


class SVRModel:
    ad = "SVR"
    IZGARA = {
        "regressor__C": [1, 10, 100, 1000],
        "regressor__gamma": [0.1, 0.3, 1, 3, 10],
        "regressor__epsilon": [0.001, 0.01],
    }

    def egit(self, X, Y, gruplar):
        self.modeller, self.en_iyi = [], {}
        for j, ad in enumerate(HEDEF_ADLARI):
            # Hedef ölçeklenir ki epsilon her iki çıktı için aynı anlama gelsin
            temel = TransformedTargetRegressor(regressor=SVR(kernel="rbf"), transformer=StandardScaler())
            arama = GridSearchCV(
                temel, self.IZGARA, cv=GroupKFold(5),
                scoring="neg_root_mean_squared_error", n_jobs=ISCI,
            )
            # Gruplar kanat kimliği: aynı kanadın açıları hem eğitime hem doğrulamaya girmesin
            arama.fit(X, Y[:, j], groups=gruplar)
            self.modeller.append(arama.best_estimator_)
            self.en_iyi[ad] = {**arama.best_params_, "cv_rmse": -arama.best_score_}

    def tahmin(self, X):
        return np.column_stack([m.predict(X) for m in self.modeller])


class MLP:
    ad = "MLP"

    def __init__(self, tohumlar=(0, 1, 2, 3, 4)):
        self.tohumlar = tohumlar

    def egit(self, X, Y, gruplar=None):
        self.olcek = StandardScaler().fit(Y)
        Ys = self.olcek.transform(Y)
        self.aglar = [
            MLPRegressor(
                hidden_layer_sizes=(128, 128, 64), activation="relu",
                early_stopping=True, max_iter=3000, random_state=s,
            ).fit(X, Ys)
            for s in self.tohumlar
        ]

    def tahmin(self, X):
        # Birkaç tohumla eğitilen ağların ortalaması
        ort = np.mean([ag.predict(X) for ag in self.aglar], axis=0)
        return self.olcek.inverse_transform(ort)
