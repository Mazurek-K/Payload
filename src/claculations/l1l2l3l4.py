import numpy as np
import matplotlib.pyplot as plt

# Constants
AU = 149597870.7       # km
d = 1.5e6 / AU         # Approximate Earth-to-L1/L2 distance in AU

# Coordinates in the orbital plane, in AU
points = {
    "Sun":  (0.0, 0.0),
    "Earth": (1.0, 0.0),
    "L1":   (1.0 - d, 0.0),
    "L2":   (1.0 + d, 0.0),
    "L3":   (-1.0, 0.0),
    "L4":   (0.5, np.sqrt(3) / 2),
    "L5":   (0.5, -np.sqrt(3) / 2),
}

fig, ax = plt.subplots(figsize=(10, 10))

# Plot the Sun and Earth
ax.scatter(*points["Sun"], s=100, color="orange", zorder=5)
ax.scatter(*points["Earth"], s=50, color="blue", zorder=5)

# Plot Lagrange points
for name in ["L1", "L2", "L3", "L4", "L5"]:
    x, y = points[name]
    ax.scatter(x, y, s=40, marker="x", color="red", zorder=6)

# Labels
ax.annotate("Sun", points["Sun"], xytext=(8, 8),
            textcoords="offset points")
ax.annotate("Earth", points["Earth"], xytext=(8, 8),
            textcoords="offset points")

# Offset labels slightly so L1 and L2 can be distinguished
ax.annotate("L1", points["L1"], xytext=(-28, 12),
            textcoords="offset points")
ax.annotate("L2", points["L2"], xytext=(8, 12),
            textcoords="offset points")

ax.annotate("L3", points["L3"], xytext=(-20, 10),
            textcoords="offset points")
ax.annotate("L4", points["L4"], xytext=(8, 8),
            textcoords="offset points")
ax.annotate("L5", points["L5"], xytext=(8, -15),
            textcoords="offset points")

# Sun-Earth line
ax.plot([0, 1], [0, 0], "k--", linewidth=0.8, alpha=0.5)

# True geometric scale: equal distances on x and y axes
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)

ax.set_xlabel("x (AU)")
ax.set_ylabel("y (AU)")
ax.set_title("Sun–Earth System: Lagrange Points (2D, True Scale)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
