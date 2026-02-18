import matplotlib.pyplot as plt

# --------------------
# Data (from your sheet)
# --------------------

# Complete Weighted
n_complete = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
w_complete = [1.1758, 1.1575, 1.1997, 1.1965, 1.1993, 1.1996, 1.2032, 1.21, 1.1998]

# Hypercube
n_hyper = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
w_hyper = [11.7325, 21.2903, 37.498, 65.9899, 122.2502, 218.6783, 397.6677, 741.6098,
           1377.7243, 2578.2893, 4833.7059, 9117.5534]

# Unit Square
n_square = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
w_square = [7.4734, 10.6306, 15.1533, 20.9707, 29.6168, 41.8939, 59.0393, 83.1646, 117.5198]

# Unit Cube
n_cube = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
w_cube = [17.4287, 27.3988, 43.3727, 67.8709, 107.4156, 169.1001, 267.399, 422.5828, 669.2997]

# Unit Hypercube
n_uhyper = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
w_uhyper = [28.5581, 47.1232, 78.398, 130.3202, 216.6957, 360.9654, 603.7268, 1007.3652, 1688.55865]

# --------------------
# Plot helper
# --------------------
def plot_one(x, y, title, xlog=True, ylog=False):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("n")
    plt.ylabel("Average MST weight")
    plt.title(title)
    if xlog:
        plt.xscale("log")
    if ylog:
        plt.yscale("log")

    plt.ylim(min(y) * 0.75, max(y) * 1.3)

    plt.tight_layout()

    plt.show()

# --------------------
# Make plots
# --------------------
plot_one(n_complete, w_complete, "Complete Weighted Graph", xlog=True, ylog=False)
plot_one(n_hyper, w_hyper, "Hypercube Graph", xlog=True, ylog=True)
plot_one(n_square, w_square, "Unit Square Graph", xlog=True, ylog=False)
plot_one(n_cube, w_cube, "Unit Cube Graph", xlog=True, ylog=False)
plot_one(n_uhyper, w_uhyper, "Unit Hypercube Graph", xlog=True, ylog=False)
