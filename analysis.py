import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Data (from your sheet)
# --------------------

# Complete Weighted
n_complete = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768], dtype=float)
w_complete = np.array([1.1387, 1.2319, 1.1867, 1.19, 1.209, 1.1996, 1.2018, 1.21, 1.1998], dtype=float)

# Hypercube
n_hyper = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144], dtype=float)
w_hyper = np.array([11.7325, 21.2903, 37.498, 65.9899, 122.2502, 218.6783, 397.6677, 741.6098,
                    1377.7243, 2578.2893, 4833.7059, 9117.5534], dtype=float)

# Unit Square
n_square = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768], dtype=float)
w_square = np.array([7.4734, 10.6306, 15.1533, 20.9707, 29.6168, 41.8939, 59.0393, 83.1646, 117.5198], dtype=float)

# Unit Cube
n_cube = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768], dtype=float)
w_cube = np.array([17.4287, 27.3988, 43.3727, 67.8709, 107.4156, 169.1001, 267.399, 422.5828, 669.2997], dtype=float)

# Unit Hypercube (d=4)
n_uhyper = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768], dtype=float)
w_uhyper = np.array([28.5581, 47.1232, 78.398, 130.3202, 216.6957, 360.9654, 603.7268, 1007.3652, 1688.55865], dtype=float)

# --------------------
# Regression helpers
# --------------------
def r2_score(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

def fit_constant(x, y):
    c = float(np.mean(y))
    yhat = np.full_like(y, c)
    return {"model": "W(n) = c", "params": {"c": c}, "predict": lambda t: np.full_like(np.array(t, dtype=float), c), "r2": r2_score(y, yhat)}

def fit_powerlaw(x, y):
    # y ≈ a * x^p  => log y = log a + p log x
    lx, ly = np.log(x), np.log(y)
    p, loga = np.polyfit(lx, ly, 1)
    a = float(np.exp(loga))
    predict = lambda t: a * (np.array(t, dtype=float) ** p)
    yhat = predict(x)
    return {"model": "W(n) = a * n^p", "params": {"a": a, "p": float(p)}, "predict": predict, "r2": r2_score(y, yhat)}

def fit_powerlaw_fixed_p(x, y, p_fixed):
    # y ≈ a * x^p_fixed  => log a = mean(log y - p log x)
    lx, ly = np.log(x), np.log(y)
    loga = float(np.mean(ly - p_fixed * lx))
    a = float(np.exp(loga))
    predict = lambda t: a * (np.array(t, dtype=float) ** p_fixed)
    yhat = predict(x)
    return {"model": f"W(n) = a * n^{p_fixed:.6g}", "params": {"a": a, "p": float(p_fixed)}, "predict": predict, "r2": r2_score(y, yhat)}

def fit_hypercube_n_over_log(x, y, log_base=2.0):
    # y ≈ a * n / log_b(n)
    x = np.array(x, dtype=float)
    denom = np.log(x) / np.log(log_base)
    feat = x / denom
    # a = argmin ||y - a feat||^2 = (feat·y)/(feat·feat)
    a = float(np.dot(feat, y) / np.dot(feat, feat))
    predict = lambda t: a * (np.array(t, dtype=float) / (np.log(np.array(t, dtype=float)) / np.log(log_base)))
    yhat = predict(x)
    return {"model": f"W(n) = a * n / log_{log_base:g}(n)", "params": {"a": a, "log_base": log_base}, "predict": predict, "r2": r2_score(y, yhat)}

def print_fit(name, fit):
    print(f"\n=== {name} ===")
    print(f"Model: {fit['model']}")
    for k, v in fit["params"].items():
        print(f"  {k} = {v}")
    print(f"R^2: {fit['r2']:.6f}")

# --------------------
# Fit models you likely want
# --------------------

# 0) Complete (constant)
fit_complete = fit_constant(n_complete, w_complete)

# 1) Hypercube (try both power-law and n/log n; pick best R^2)
fit_hyper_pow = fit_powerlaw(n_hyper, w_hyper)
fit_hyper_nlog = fit_hypercube_n_over_log(n_hyper, w_hyper, log_base=2.0)

# 2) Geometric: estimate exponent p from data, and also fit fixed-theory p=(d-1)/d
# Unit square (d=2 => p=1/2)
fit_square_free = fit_powerlaw(n_square, w_square)
fit_square_theory = fit_powerlaw_fixed_p(n_square, w_square, p_fixed=0.5)

# Unit cube (d=3 => p=2/3)
fit_cube_free = fit_powerlaw(n_cube, w_cube)
fit_cube_theory = fit_powerlaw_fixed_p(n_cube, w_cube, p_fixed=2/3)

# Unit hypercube (d=4 => p=3/4)
fit_uhyper_free = fit_powerlaw(n_uhyper, w_uhyper)
fit_uhyper_theory = fit_powerlaw_fixed_p(n_uhyper, w_uhyper, p_fixed=3/4)

# Print results
print_fit("Complete Weighted", fit_complete)

print_fit("Hypercube (power-law)", fit_hyper_pow)
print_fit("Hypercube (n/log2 n)", fit_hyper_nlog)

print_fit("Unit Square (free power-law)", fit_square_free)
print_fit("Unit Square (theory p=1/2)", fit_square_theory)

print_fit("Unit Cube (free power-law)", fit_cube_free)
print_fit("Unit Cube (theory p=2/3)", fit_cube_theory)

print_fit("Unit Hypercube (free power-law)", fit_uhyper_free)
print_fit("Unit Hypercube (theory p=3/4)", fit_uhyper_theory)

# --------------------
# Plot with fitted curves
# --------------------
def plot_with_fit(x, y, title, fits, xlog=True, ylog=False):
    plt.figure()
    plt.plot(x, y, marker="o", linestyle="", label="data")

    xs = np.geomspace(np.min(x), np.max(x), 400)
    for fit in fits:
        ys = fit["predict"](xs)
        label = fit["model"] + f" (R^2={fit['r2']:.3f})"
        plt.plot(xs, ys, label=label)

    plt.xlabel("n")
    plt.ylabel("Average MST weight")
    plt.title(title)
    if xlog:
        plt.xscale("log")
    if ylog:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_with_fit(n_complete, w_complete, "Complete Weighted Graph", [fit_complete], xlog=True, ylog=False)

# Hypercube: show both candidates on log-log (helps visually)
plot_with_fit(n_hyper, w_hyper, "Hypercube Graph", [fit_hyper_pow, fit_hyper_nlog], xlog=True, ylog=True)

plot_with_fit(n_square, w_square, "Unit Square Graph", [fit_square_free, fit_square_theory], xlog=True, ylog=False)
plot_with_fit(n_cube, w_cube, "Unit Cube Graph", [fit_cube_free, fit_cube_theory], xlog=True, ylog=False)
plot_with_fit(n_uhyper, w_uhyper, "Unit Hypercube Graph", [fit_uhyper_free, fit_uhyper_theory], xlog=True, ylog=False)

# --------------------
# Optional: automatically pick "best" fit by R^2 for each family
# --------------------
def best_fit(*fits):
    return max(fits, key=lambda f: f["r2"])

print("\n\n=== Best-fit summary (by R^2) ===")
print(f"Complete: {best_fit(fit_complete)['model']}")
print(f"Hypercube: {best_fit(fit_hyper_pow, fit_hyper_nlog)['model']}")
print(f"Unit Square: {best_fit(fit_square_free, fit_square_theory)['model']}")
print(f"Unit Cube: {best_fit(fit_cube_free, fit_cube_theory)['model']}")
print(f"Unit Hypercube: {best_fit(fit_uhyper_free, fit_uhyper_theory)['model']}")
