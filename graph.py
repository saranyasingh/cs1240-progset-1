import numpy as np
import matplotlib.pyplot as plt
# ---------- data ----------
complete_n = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
complete_t = [0.0016, 0.0043, 0.0154, 0.0515, 0.1894, 0.7270, 2.7938, 11.2099, 45.8443]

hypercube_n = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
hypercube_t = [0.0497, 0.0014, 0.0032, 0.0079, 0.0188, 0.0471, 0.1079, 0.2542, 0.7069, 1.7355, 6.1152, 21.3428]

sq2d_n = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
sq2d_t = [0.4108, 0.0057, 0.0184, 0.0598, 0.2117, 1.0048, 3.8614, 14.9926, 65.4167]

cube3d_n = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
cube3d_t = [0.2826, 0.0090, 0.0285, 0.1105, 0.4257, 1.9152, 8.9104, 41.1200, 194.2079]

hyper4d_n = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
hyper4d_t = [0.6767, 0.0123, 0.0415, 0.1651, 0.8388, 4.2470, 20.7915, 107.0295, 529.1853]

# ----------------------------
# Assumes you already have:
#   hypercube_n = [...]
#   hypercube_t = [...]
# ----------------------------
x = np.array(hypercube_n, dtype=float)
y = np.array(hypercube_t, dtype=float)

# ----------------------------
# Helpers
# ----------------------------
def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else float("nan")

def fit_scale_only(fx, y):
    """Fit y ≈ a * fx (no intercept)."""
    a = float(np.dot(fx, y) / np.dot(fx, fx))
    yhat = a * fx
    return a, yhat, r2(y, yhat)


# ----------------------------
# Build candidate model features
# ----------------------------
log2x = np.log2(x)

f_n_log2_sq = x * (log2x**2)                 # n log^2 n
f_n2_plus_n_log2_sq = x**2 + x * (log2x**2)  # n^2 + n log^2 n
f_n = x**2

# ----------------------------
# Fit both models (choose one: scale-only or scale+intercept)
# ----------------------------

a1, yhat1, r2_1 = fit_scale_only(f_n_log2_sq, y)

print("Model: y = a*(n log^2 n)")
print(f"  a={a1:.6g}, R^2={r2_1:.6f}")

# ----------------------------
# Plot data + fits
# ----------------------------
plt.figure()
plt.plot(x, y, "o-", label="data")

plt.plot(x, yhat1, "--", linewidth=2, label=fr"$a\,n \log^2 n$ (R$^2$={r2_1:.3f})")
# plt.plot(x, yhat1, "--", linewidth=2, label=fr"$a\,(n^2+n\log_2^2 n)$ (R$^2$={r2_1:.3f})")

plt.xscale("log")
plt.yscale("log")  # timing plots usually look best log-log
plt.xlabel("n")
plt.ylabel("time")
plt.title("Hypercube Runtime")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# Quick residual diagnostics
# ----------------------------
res1 = y - yhat1
print("\nResidual summary:")
print(f"  Model 1: mean={np.mean(res1):.6g}, std={np.std(res1, ddof=1):.6g}, max|res|={np.max(np.abs(res1)):.6g}")
