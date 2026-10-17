# Aerodynamic Surrogate Model: NACA 4-Digit Airfoils

**English** | [Türkçe](README.tr.md)

Computing CL and CD for one airfoil at one angle with XFOIL takes anywhere from a fraction of a second to a few seconds. That gets slow when a design sweep needs hundreds of thousands of points. In this project I generated a database with XFOIL, trained three surrogate models on it, and compared them against linear interpolation on airfoils the models never saw during training.

- Inputs: camber `m`, camber position `p`, thickness `t`, Reynolds number `Re`, angle of attack `alfa`
- Outputs: `CL` and `CD`
- Models: Gaussian Process Regression (GPR), Support Vector Regression (SVR), multilayer perceptron (MLP)
- Baseline: linear interpolation

File and column names are in Turkish: `egitim` = training, `kor_test` = blind test, `alfa` = angle of attack, `kombinasyon` = airfoil ID.

## Method

**Geometry.** NACA 4-digit profiles come from my own generator (`src/geometri.py`): 100 cosine-spaced points along the chord, coordinates in Selig order, closed trailing edge thickness coefficient (−0.1036). Compared with AeroSandbox's NACA 2412, the largest difference is 1.27e-3. All of it sits at the trailing edge and comes from AeroSandbox using the open trailing edge formula (−0.1015): the half-thickness difference at x = 1 is exactly `5 · 0.12 · 0.0021 = 1.26e-3`.

**Design of experiments.** 4D Latin Hypercube sampling (`src/tasarim.py`):

| Input | Range |
|---|---|
| m | 0 – 0.06 |
| p | 0.2 – 0.6 |
| t | 0.08 – 0.18 |
| log10(Re) | 5.3 – 6.3 |

90 training airfoils and 25 blind test airfoils were drawn with different seeds. Angle of attack goes from −6° to 20° in 1° steps (27 angles). The code checks that the two sets share no airfoil ID and no geometry (`src/hazirlik.py`).

**Data generation.** `src/veri_uret.py` runs two separate XFOIL sweeps per airfoil: 0° up to 20°, and 0° down to −6°. A single sweep from −6° to 20° tends to hurt convergence. Runs are parallel, so every airfoil gets a unique name. XFOIL rounds the angle it prints, so results are matched to the requested angle with a 0.01° tolerance. Rows with NaN, `CD <= 0` or `|CL| > 3` are dropped.

**Inputs and targets.** Features are `[m, p, t, log10(Re), alfa]`, each scaled to [0, 1] using the design ranges. The scaling bounds come from the design, the test data is never looked at. Targets are `CL` and `log10(CD)`. CD spans several orders of magnitude, so its logarithm is modelled and predictions are mapped back with `10 ** prediction`.

**Models** (`src/modeller.py`):

- **Interpolation:** `LinearNDInterpolator`, with `NearestNDInterpolator` for points outside the convex hull.
- **GPR:** kernel `Constant × Matern(ν = 2.5, one length scale per input) + White`, `normalize_y=True`. Hyperparameters are learned on a random 750-point subset, then the learned kernel is fit to all data with `optimizer=None`. Separate models for CL and log10(CD).
- **SVR:** RBF kernel, target scaled with `StandardScaler`. Grid search over `C`, `gamma` and `epsilon` with `GroupKFold(5)`, grouped by airfoil ID so neighbouring angles of the same airfoil never land in both training and validation folds. Scoring is RMSE.
- **MLP:** layers `(128, 128, 64)`, ReLU, `early_stopping=True`. One network with two outputs, averaged over 5 seeds. `early_stopping` picks its validation rows at random, so there is a small leak in the stopping decision only. The final evaluation uses the blind test, so the results are not affected.

## Data

| Set | Airfoils | Requested rows | Dropped | Kept |
|---|---|---|---|---|
| Training (`data/egitim.csv`) | 90 | 2430 | 132 (5.4%) | 2298 |
| Blind test (`data/kor_test.csv`) | 25 | 675 | 28 (4.1%) | 647 |

Every dropped row is a point where XFOIL did not converge. No row was dropped for `CD <= 0` or `|CL| > 3`. 51 of the 132 dropped training rows are between 18° and 20°, so convergence problems pile up after stall, as expected. The counts are in `data/uretim_ozeti.json`.

The data pipeline was first checked on known airfoils (`src/buyukluk_kontrol.py`, Re = 1e6):

| | CL(0°) | CL(5°) | CLmax | Min CD |
|---|---|---|---|---|
| NACA 0012 | 0.000 | 0.544 | 1.315 (16°) | 0.00533 |
| NACA 2412 | 0.226 | 0.843 | 1.433 (17°) | 0.00539 |

## Results

All four methods were evaluated on the same 647 blind test rows (`results/sonuclar.csv`). Stall region is taken as `alfa >= 12`.

| Method | CL RMSE | CL MAE | CL R² | log10(CD) RMSE | CD rel. error | Stall CL RMSE | Stall log10(CD) RMSE |
|---|---|---|---|---|---|---|---|
| Interpolation | 0.0658 | 0.0413 | 0.9879 | 0.0544 | 7.3% | 0.0684 | 0.0732 |
| GPR | **0.0248** | **0.0148** | **0.9983** | **0.0236** | **3.1%** | 0.0364 | 0.0352 |
| SVR | 0.0309 | 0.0211 | 0.9973 | 0.0308 | 4.9% | **0.0313** | 0.0358 |
| MLP | 0.0268 | 0.0194 | 0.9980 | 0.0268 | 4.3% | **0.0313** | **0.0314** |

Compute cost:

| Method | Training time | Prediction time, 1000 points |
|---|---|---|
| Interpolation | 20.9 s | 10.5 s |
| GPR | 51.3 s | 0.16 s |
| SVR | 485.7 s (incl. grid search) | 0.14 s |
| MLP | 8.4 s (5 networks) | 0.005 s |
| XFOIL | – | ≈ 637 s |

The XFOIL figure is based on the per-point time measured per process during data generation (0.64 s, both sweeps included). It was measured under parallel load, so treat it as a rough number.

**Discussion:**

- **Overall:** all three models clearly beat interpolation: CL RMSE is 2.1–2.7 times lower, CD relative error 1.5–2.4 times lower. GPR has the best overall accuracy.
- **Stall region:** GPR's stall CL error (0.0364) is about 1.5 times its overall error (0.0248). SVR and MLP do slightly better in stall (0.0313). The sudden break in CL after the stall peak is a harder shape for a smooth Matern kernel.
- **Why interpolation is weak:** 59.5% of test points fall outside the convex hull of the training points. With 90 airfoils in 5 dimensions the hull covers only a small part of the design space, and nearest neighbour is used outside it. The steps in `results/cl_alfa_test_kanadi.png` come from this. It is the main reason the baseline falls behind.
- **SVR grid search** picked `C = 100, gamma = 1, epsilon = 0.01` for CL and `C = 1000, gamma = 1, epsilon = 0.01` for log10(CD), with grouped CV RMSE of 0.0378 and 0.0342. The `C` chosen for log10(CD) is at the edge of the grid, so a wider grid might help a little.
- **Speed:** every trained model predicts 1000 points in under a second. The same job takes about ten minutes in XFOIL.

### GPR length scales

Learned values in the scaled [0, 1] input space (`results/gpr_uzunluk_olcekleri.csv`):

| Input | CL | log10(CD) |
|---|---|---|
| m | 1.40 | 1.42 |
| p | 1.90 | 1.53 |
| t | 1.43 | 1.28 |
| log10(Re) | 2.04 | 1.98 |
| alfa | **0.42** | **0.31** |

A short length scale means the output changes faster and less linearly with that input. Angle of attack is clearly the input that drives CL the most. Among the geometry inputs, camber and thickness matter more than camber position. Re has the longest scale, so within this range it affects CL the least. For CD, angle comes first again, followed by thickness.

A length scale is a sensitivity measure, not exactly an effect size. An input with a large but linear effect can still get a long length scale. Camber is an example: it shifts CL a lot, but almost linearly.

## Plots

All plots are produced by `src/grafikler.py` from the real XFOIL data and blind test predictions. Plot labels are in Turkish.

`results/tasarim_uzayi.png`: where the training and blind test airfoils sit in the design space. Test airfoils are inside the same ranges but do not overlap the training points.

![Design space](results/tasarim_uzayi.png)

`results/ornek_profiller.png`: a few test airfoils with extreme camber and thickness, drawn with `src/geometri.py`.

![Example profiles](results/ornek_profiller.png)

`results/cl_tahmin_gercek.png`: predicted vs XFOIL CL for each method, stall points marked separately.

![Predicted vs true CL](results/cl_tahmin_gercek.png)

`results/cl_alfa_test_kanadi.png`: CL–alpha curve for the test airfoil with the highest peak CL (T018), grey area is the stall region.

![CL alpha curve](results/cl_alfa_test_kanadi.png)

`results/rmse_karsilastirma.png`: RMSE on all test points and on the stall region only.

![RMSE comparison](results/rmse_karsilastirma.png)

`results/aciya_gore_hata.png`: blind test CL RMSE at each angle. Below 12°, GPR has the lowest error at almost every angle. Between 15° and 18° it has the highest error of the three models, reaching 0.052 at 18°. This is why GPR's stall RMSE is worse than SVR and MLP. Interpolation has the highest error at every angle.

![Error by angle](results/aciya_gore_hata.png)

`results/surukleme_polari.png`: CL–CD polar for the same test airfoil. The models capture both the low-drag region and the rise towards stall. The shift in the interpolation curve around CL = 0.5 comes from the nearest-neighbour fallback outside the hull.

![Drag polar](results/surukleme_polari.png)

## Limitations

- XFOIL is a 2D viscous panel method and does not replace CFD. It is valid for low Mach numbers. All data was generated with `mach = 0` and `n_crit = 9`.
- XFOIL itself is less reliable in the stall region (separated flow, convergence issues). Most dropped points are also there.
- The models are only valid inside the sampled design ranges. The blind test airfoils are inside the same ranges, so this is not an extrapolation test.
- The data is synthetic and contains no real measurements.
- The blind test has 25 airfoils. The gap between GPR and MLP is small (CL RMSE 0.0248 vs 0.0268), and the ranking could change on a different test set.
- Dropped points stay as gaps. No value is filled in for angles that did not converge.

## Reproducing

On Windows with Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download XFOIL 6.99 and place it at `bin/xfoil.exe`:

```powershell
New-Item -ItemType Directory -Force bin | Out-Null
Invoke-WebRequest -Uri "https://web.mit.edu/drela/Public/web/xfoil/XFOIL6.99.zip" -OutFile "bin\XFOIL6.99.zip"
Expand-Archive "bin\XFOIL6.99.zip" -DestinationPath "bin\xfoil_zip"
Copy-Item "bin\xfoil_zip\xfoil.exe" "bin\xfoil.exe"
```

Then, in order:

```powershell
python src\kurulum_kontrol.py    # NACA 2412, Re = 1e6, alpha = 5 deg: CL ≈ 0.81
python src\geometri_kontrol.py   # own geometry vs AeroSandbox
python src\buyukluk_kontrol.py   # NACA 0012 and 2412 sanity checks
python src\veri_uret.py          # data/egitim.csv, data/kor_test.csv (about 3-4 minutes)
python src\hazirlik.py           # leakage check
python src\degerlendir.py        # models and blind test (about 10 minutes)
python src\grafikler.py          # results/*.png
```

Run times depend on the machine. The parallel XFOIL runs use the number of CPU cores minus two.

## Project layout

```
src/        source code
data/       egitim.csv, kor_test.csv, uretim_ozeti.json
results/    result tables and plots
bin/        xfoil.exe (not in the repository)
```

## License

MIT, see [LICENSE](LICENSE).
