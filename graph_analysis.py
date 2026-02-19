import matplotlib.pyplot as plt
import numpy as np

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

# ---------- plot ----------
plt.figure(figsize=(9, 6))

plt.plot(complete_n, complete_t, marker="o", label="Complete Weighted")
plt.plot(hypercube_n, hypercube_t, marker="o", label="Hypercube")
plt.plot(sq2d_n, sq2d_t, marker="o", label="Unit Square (2D)")
plt.plot(cube3d_n, cube3d_t, marker="o", label="Unit Cube (3D)")
plt.plot(hyper4d_n, hyper4d_t, marker="o", label="Unit Hypercube (4D)")

# ---------- n log^2 n reference ----------
x_vals = np.array(hypercube_n)
xlog2x = x_vals * (np.log2(x_vals) ** 2)


# scale so it aligns with hypercube at largest n
x_vals = np.array(hypercube_n)

# Compute x^2 + x log^2 x
x2_plus_xlog2x = x_vals * (np.log2(x_vals) ** 2)

# Scale so it aligns with hypercube at largest n
scale = hypercube_t[-1] / x2_plus_xlog2x[-1]

plt.plot(
    x_vals,
    scale * x2_plus_xlog2x,
    linestyle="--",
    linewidth=2,
    label=r"$n^2 + n \log^2 n$"
)


# log scales
plt.xscale("log", base=2)
plt.yscale("log")

plt.xlabel("n (log2 scale)")
plt.ylabel("Time (sec, log scale)")
plt.title("Timing Results by Graph Type")
plt.grid(True, which="both", linestyle="--", linewidth=0.6)
plt.legend()
plt.tight_layout()
plt.show()
