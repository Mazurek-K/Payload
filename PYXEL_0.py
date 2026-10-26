import pyxel
import matplotlib.pyplot as plt

with open("my_first_sim.yaml", "r") as file:
    config = pyxel.load("my_first_sim.yaml")

data_tree = pyxel.run_mode(config)


# Inspect the available results
print(data_tree)

# Plot photon counts at the first readout
data_tree["/bucket/photon"].isel(time=0).plot(
    cmap="inferno"
)
plt.title("Photon counts")
plt.show()

# Plot collected charge
data_tree["/bucket/charge"].isel(time=0).plot(
    cmap="inferno"
)
plt.title("Collected charge")
plt.show()

# Plot the final digital image
data_tree["/bucket/image"].isel(time=0).plot(
    cmap="inferno"
)
plt.title("Simulated detector image")
plt.show()