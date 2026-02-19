import numpy as np
import matplotlib.pyplot as plt

# --------------------
# Data
# --------------------
n_complete = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768], dtype=float)
w_complete = np.array([1.1758, 1.1575, 1.1997, 1.1965, 1.1993, 1.1996, 1.2032, 1.21, 1.1998], dtype=float)

n_hyper = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144], dtype=float)
w_hyper = np.array([11.7325, 21.2903, 37.498, 65.9899, 122.2502, 218.6783, 397.6677, 741.6098,
                    1377.7243, 2578.2893, 4833.7059, 9117.5534], dtype=float)

n_square = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768], dtype=float)
w_square = np.array([7.4734, 10.6306, 15.1533, 20.9707, 29.6168, 41.8939, 59.0393, 83.1646, 117.5198], dtype=float)

n_cube = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768], dtype=float)
w_cube = np.array([17.4287, 27.3988, 43.3727, 67.8709, 107.4156, 169.1001, 267.399, 422.5828, 669.2997], dtype=float)

n_uhyper = np.array([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768], dtype=float)
w_uhyper = np.array([28.5581, 47.1232, 78.398, 130.3202, 216.6957, 360.9654, 603.7268, 1007.3652, 1688.55865], dtype=float)

# --------------------
# Predicted functions (USE EXACTLY the fits you listed)
# --------------------

# Complete weighted: constant
c_complete = 1.1934888888888888
r2_complete = 0.0

# Hypercube: a * n / log2(n)
a_hyper = 0.6265510779062619
r2_hyper = 0.999995

# Unit square: a * n^p  (free power-law)
a_square = 0.6799597832880142
p_square = 0.4954278859979664
r2_square = 0.999991

# Unit cube: a * n^p  (free power-law)
a_cube = 0.7146038412829555
p_cube = 0.6576482097205681
r2_cube = 0.999974

# Unit hypercube: a * n^p  (free power-law)
a_uhyper = 0.7964584165635816
p_uhyper = 0.7359470727876873
r2_uhyper = 0.999935

def pred_complete(n):
    n = np.array(n, dtype=float)
    return np.full_like(n, c_complete)

def pred_hyper(n):
    n = np.array(n, dtype=float)
    return a_hyper * n / np.log2(n)

def pred_square(n):
    n = np.array(n, dtype=float)
    return a_square * n**p_square

def pred_cube(n):
    n = np.array(n, dtype=float)
    return a_cube * n**p_cube

def pred_uhyper(n):
    n = np.array(n, dtype=float)
    return a_uhyper * n**p_uhyper

# --------------------
# Plot helper: data + orange dotted predicted curve
# --------------------
def plot_one_with_prediction(x, y, title, pred_fn, pred_label,
                             xlog=True, ylog=False):
    plt.figure()

    # data
    plt.plot(x, y, marker="o", label="data")

    # smooth x grid for predicted curve
    xs = np.geomspace(np.min(x), np.max(x), 400)
    ys = pred_fn(xs)

    # orange dotted prediction
    plt.plot(xs, ys, linestyle=":", linewidth=2, color="orange", label=pred_label)

    plt.xlabel("n")
    plt.ylabel("Average MST weight")
    plt.title(title)

    if xlog:
        plt.xscale("log")
    if ylog:
        plt.yscale("log")

    ymin = min(np.min(y), np.min(ys)) * 0.75
    ymax = max(np.max(y), np.max(ys)) * 1.3
    plt.ylim(ymin, ymax)

    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------
# Make plots (with the predicted orange dotted line)
# --------------------
plot_one_with_prediction(
    n_complete, w_complete, "Complete Weighted Graph",
    pred_complete, fr"prediction: $f(n)=c$ (c={c_complete:.4f}, $R^2$={r2_complete:.6f})",
    xlog=True, ylog=False
)

plot_one_with_prediction(
    n_hyper, w_hyper, "Hypercube Graph",
    pred_hyper, fr"prediction: $f(n)=a\,\frac{{n}}{{\log_2 n}}$ (a={a_hyper:.4f}, $R^2$={r2_hyper:.6f})",
    xlog=True, ylog=True
)

plot_one_with_prediction(
    n_square, w_square, "Unit Square Graph",
    pred_square, fr"prediction: $f(n)=a\,n^p$ (a={a_square:.4f}, p={p_square:.4f}, $R^2$={r2_square:.6f})",
    xlog=True, ylog=False
)

plot_one_with_prediction(
    n_cube, w_cube, "Unit Cube Graph",
    pred_cube, fr"prediction: $f(n)=a\,n^p$ (a={a_cube:.4f}, p={p_cube:.4f}, $R^2$={r2_cube:.6f})",
    xlog=True, ylog=False
)

plot_one_with_prediction(
    n_uhyper, w_uhyper, "Unit Hypercube Graph",
    pred_uhyper, fr"prediction: $f(n)=a\,n^p$ (a={a_uhyper:.4f}, p={p_uhyper:.4f}, $R^2$={r2_uhyper:.6f})",
    xlog=True, ylog=False
)
