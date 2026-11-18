import requests
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from config import (AU_2_M)

def download_orbit(command_str, filename):
    params = {
        "format": "text",
        "COMMAND": f"'{command_str}'",  # e.g., "-21" for SOHO, "DES=3562756;" for 2011 EO40
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'500@10'",  # Sun-centered (use '500@31' for Sun-Earth L1)
        "START_TIME": "'2019-11-15'",
        "STOP_TIME": "'2019-12-10'",
        "STEP_SIZE": "'1d'",
        "OUT_UNITS": "'KM-S'",
        "VEC_TABLE": "'2'",  # State vector: X, Y, Z, VX, VY, VZ
        "CSV_FORMAT": "'YES'",
    }

    response = requests.get("https://ssd.jpl.nasa.gov/api/horizons.api", params=params)
    text = response.text

    # Safety check to print Horizons' actual error if a query fails
    if "$$SOE" not in text:
        raise ValueError(f"Horizons API error for {command_str}:\n{text}")

    data = text.split("$$SOE")[1].split("$$EOE")[0].strip()

    address = f"../outputs/{filename}.csv"
    with open(address, "w") as f:
        f.write("jd,calendar_date,x_km,y_km,z_km,vx_kms,vy_kms,vz_kms\n")
        for line in data.splitlines():
            f.write(line.rstrip(",") + "\n")

    print(f"Saved {address}")

# download_orbit("DES=2024 MK;", "2024_MK")   # 2011 EO40

#
download_orbit("-21", "soho")
download_orbit("31", "L1")
# download_orbit("DES=2024 VA11;", "2026_VA11")   # 2011 EO40
download_orbit("DES=2019 XF2;", "2019_XF2")


# Read sc
df1 = pd.read_csv("../outputs/soho.csv")
v_sc = np.array([df1["vx_kms"],df1["vy_kms"],df1["vz_kms"]])
r_sc = np.array([df1["x_km"],df1["y_km"],df1["z_km"]])

# Read asteroid
df2 = pd.read_csv("../outputs/2019_XF2.csv")
v_ast = np.array([df2["vx_kms"],df2["vy_kms"],df2["vz_kms"]])
r_ast = np.array([df2["x_km"],df2["y_km"],df2["z_km"]])

# Read time
time = df1["jd"]

# Read L1
df3 = pd.read_csv("../outputs/L1.csv")
v_L1 = np.array([df3["vx_kms"],df3["vy_kms"],df3["vz_kms"]])
r_L1 = np.array([df3["x_km"],df3["y_km"],df3["z_km"]])


# Define coordinate system of the SC, similar to LVLH but not fully
# Always point to Sun but derive other from L1
x_sc = -r_sc / np.linalg.norm(r_sc, axis=0)
h = np.cross(r_L1, v_L1, axis=0)

z_sc = h - np.sum(h * x_sc, axis=0) * x_sc # Gram–Schmidt
z_sc = z_sc / np.linalg.norm(z_sc, axis=0)

y_sc = np.cross(z_sc, x_sc, axis=0)
y_sc = y_sc / np.linalg.norm(y_sc, axis=0)
z_sc = np.cross(x_sc, y_sc, axis=0)


# Sensor rotations relative to SC frame
rot0 = R.from_euler('y', 25, degrees=True)
rot1 = R.from_euler('x', -90, degrees=True)
rot_sensor = rot1 * rot0

# Relative asteroid position and velocity (inertial)
dr = (r_ast - r_sc).T
dv = (v_ast - v_sc).T

# SC -> inertial transformation
M = np.stack([x_sc.T, y_sc.T, z_sc.T], axis=-1)

# Inertial -> SC frame
dr_sc = np.array([M[i].T @ dr[i] for i in range(len(dr))])
dv_sc = np.array([M[i].T @ dv[i] for i in range(len(dv))])

# SC -> sensor frame
dr_sensor = rot_sensor.inv().apply(dr_sc)
dv_sensor = rot_sensor.inv().apply(dv_sc)

# Perspective projection onto sensor XY plane
u = dr_sensor[:, 1] / dr_sensor[:, 0]
v = dr_sensor[:, 2] / dr_sensor[:, 0]

# Apparent image-plane velocity
du = (dv_sensor[:, 1]*dr_sensor[:, 0]
      - dr_sensor[:, 1]*dv_sensor[:, 0]) / dr_sensor[:, 0]**2

dv_img = (dv_sensor[:, 2]*dr_sensor[:, 0]
          - dr_sensor[:, 2]*dv_sensor[:, 0]) / dr_sensor[:, 0]**2

# Exposure streak
exposure = 10.0  # seconds

u_start = u - du * exposure/2
v_start = v - dv_img * exposure/2

u_end = u + du * exposure/2
v_end = v + dv_img * exposure/2

# Visible targets
visible = dr_sensor[:, 0] > 0




plt.figure(figsize=(8, 8))

plt.plot(u[visible], v[visible], '-', label='Asteroid trajectory')

plt.xlabel('Sensor Y / X')
plt.ylabel('Sensor Z / X')
plt.title('Asteroid trajectory in sensor plane')
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))

plt.plot(time, np.linalg.norm(dr * 1000 / AU_2_M, axis = 1), '-', label='DR')

plt.xlabel('Time (JD)')
plt.ylabel('Distance (AU)')
plt.title('Relative distance over time')

plt.margins(x=0.02, y=0.08)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



def plot3d(*dfs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    all_x, all_y, all_z = [], [], []

    for i, df in enumerate(dfs):
        label = df.attrs.get("name", f"Trajectory {i+1}")

        ax.plot(df["x_km"], df["y_km"], df["z_km"], label=label)

        all_x.extend(df["x_km"])
        all_y.extend(df["y_km"])
        all_z.extend(df["z_km"])

    max_range = max(
        max(all_x) - min(all_x),
        max(all_y) - min(all_y),
        max(all_z) - min(all_z)
    ) / 2

    mid_x = (max(all_x) + min(all_x)) / 2
    mid_y = (max(all_y) + min(all_y)) / 2
    mid_z = (max(all_z) + min(all_z)) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend()
    plt.show()

plot3d(df1, df2, df3)