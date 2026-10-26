import pyxel
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


with open('src/pyxel/my_first_sim.yaml', 'r') as file:
    YAML_CONFIG = file.read()
config = pyxel.loads(YAML_CONFIG)
config.pipeline.charge_generation.cosmix.enabled = True

data_tree = pyxel.run_mode(config)
print(data_tree)
data_tree["/bucket/charge"].isel(time=0).plot(robust=True, yincrease=False)
plt.show()


data_tree["/bucket/photon"].plot.hist(
    bins=100,
    color="C0",
    density=False,
    label="Histogram",
)
plt.show()
data_tree["/bucket/pixel"].plot(col="time", col_wrap=4, yincrease=False)
plt.show()