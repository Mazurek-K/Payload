import requests
import matplotlib
matplotlib.use("TkAgg")
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
        "START_TIME": "'2019-11-1'",
        "STOP_TIME": "'2019-12-1'",
        "STEP_SIZE": "'1h'",
        "OUT_UNITS": "'KM-S'",
        "VEC_TABLE": "'2'",  # State vector: X, Y, Z, VX, VY, VZ
        "CSV_FORMAT": "'YES'",
    }
    # "START_TIME": "'2019-11-1'",
    # "STOP_TIME": "'2019-12-15'",

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


download_orbit("-21", "soho")
download_orbit("31", "L1")
download_orbit("DES=2019 XF2;", "2019_XF2")
download_orbit("399", "Earth")

#

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

# Read earth
df4 = pd.read_csv("../outputs/Earth.csv")



# Define coordinate system of the SC, similar to LVLH but not fully
# Always point to Sun but derive other from L1
# x points sun, z points h
x_sc = -r_sc / np.linalg.norm(r_sc, axis=0)
h = np.cross(r_L1, v_L1, axis=0)

z_sc = h - np.sum(h * x_sc, axis=0) * x_sc # Gram–Schmidt
z_sc = z_sc / np.linalg.norm(z_sc, axis=0)

y_sc = np.cross(z_sc, x_sc, axis=0)
y_sc = y_sc / np.linalg.norm(y_sc, axis=0)
z_sc = np.cross(x_sc, y_sc, axis=0)

# SC -> inertial transformation
M = np.stack([x_sc.T, y_sc.T, z_sc.T], axis=-1)

# Sensor rotations relative to SC frame
rot0 = R.from_euler('y', 0, degrees=True)
rot1 = R.from_euler('x', 0, degrees=True)
rot_sensor = rot1 * rot0

x_sensor = rot_sensor.apply([1, 0, 0])
y_sensor = rot_sensor.apply([0, 1, 0])
z_sensor = rot_sensor.apply([0, 0, 1])

x_sensor_inertial = np.array([M[i] @ x_sensor for i in range(len(M))])
y_sensor_inertial = np.array([M[i] @ y_sensor for i in range(len(M))])
z_sensor_inertial = np.array([M[i] @ z_sensor for i in range(len(M))])

# Relative asteroid position (inertial) spacecraft to asteorid
dr = (r_ast - r_sc).T

visibility = np.vecdot(dr, x_sensor_inertial)>0

# Inertial -> SC frame
dr_sc = np.array([M[i].T @ dr[i] for i in range(len(dr))])

# SC -> sensor frame
dr_sensor = rot_sensor.inv().apply(dr_sc)

# Perspective projection onto sensor plane
#
hor_sensor = dr_sensor[:, 1] / dr_sensor[:, 0]
ver_sensor = dr_sensor[:, 2] / dr_sensor[:, 0]

hor_sensor = hor_sensor[visibility]
ver_sensor = ver_sensor[visibility]

fov1 = 45
fov2 = 90
limit_hor = np.tan(np.deg2rad(fov1/2))
limit_ver = np.tan(np.deg2rad(fov2/2))

mask = (np.abs(hor_sensor) <limit_hor )& \
       (np.abs(ver_sensor) < limit_ver)

x_filtered = hor_sensor[mask]
y_filtered = ver_sensor[mask]

focal_len = 1.2
x_filtered*=focal_len
y_filtered*=focal_len

plt.figure(figsize=(6, 8))

# plt.plot(u[visible], v[visible], '-', label='Asteroid trajectory')
plt.plot(x_filtered, y_filtered, '-', label='Asteroid trajectory')
plt.xlim(-limit_hor, limit_hor)
plt.ylim(-limit_ver, limit_ver)
plt.xlabel('Sensor Y / X')
plt.ylabel('Sensor Z / X')
plt.title('Asteroid trajectory in sensor plane')
# plt.axis('equal')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.legend()
plt.show()


# plt.figure(figsize=(10, 5))
#
# plt.plot(time, np.linalg.norm(dr * 1000 / AU_2_M, axis = 1), '-', label='DR')
#
# plt.xlabel('Time (JD)')
# plt.ylabel('Distance (AU)')
# plt.title('Relative distance over time')
#
# plt.margins(x=0.02, y=0.08)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()



def plot3d(*dfs, frame_indices=None):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_x, all_y, all_z = [], [], []

    # Plot trajectories
    for i, df in enumerate(dfs):

        label = df.attrs.get("name", f"Trajectory {i+1}")

        ax.plot(
            df["x_km"],
            df["y_km"],
            df["z_km"],
            label=label
        )

        all_x.extend(df["x_km"])
        all_y.extend(df["y_km"])
        all_z.extend(df["z_km"])

    # Plot spacecraft coordinate systems
    if frame_indices is not None:

        # Scale coordinate axes relative to orbit dimensions
        max_range = max(
            np.ptp(all_x),
            np.ptp(all_y),
            np.ptp(all_z)
        )

        axis_length = max_range * 0.05

        for i, idx in enumerate(frame_indices):

            origin = np.array([
                r_sc[0, idx],
                r_sc[1, idx],
                r_sc[2, idx]
            ])

            # # Spacecraft frame
            # ax.quiver(*origin, *x_sc[:, idx], length=axis_length,
            #           normalize=True, color='red')
            #
            # ax.quiver(*origin, *y_sc[:, idx], length=axis_length,
            #           normalize=True, color='green')
            #
            # ax.quiver(*origin, *z_sc[:, idx], length=axis_length,
            #           normalize=True, color='blue')

            # Sensor frame
            ax.quiver(*origin, *x_sensor_inertial[idx, :],
                      length=axis_length, normalize=True,
                      color='darkred')

            ax.quiver(*origin, *y_sensor_inertial[idx, :],
                      length=axis_length, normalize=True,
                      color='darkgreen')

            ax.quiver(*origin, *z_sensor_inertial[idx, :],
                      length=axis_length, normalize=True,
                      color='darkblue')

            ax.plot(
                [r_sc[0, idx], r_ast[0, idx]],
                [r_sc[1, idx], r_ast[1, idx]],
                [r_sc[2, idx], r_ast[2, idx]],
                color='gray',
                linestyle='--',
                linewidth=0.8,
                alpha=0.5
            )

    # Axis limits
    max_range = max(
        np.ptp(all_x),
        np.ptp(all_y),
        np.ptp(all_z)
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
    plt.tight_layout()
    plt.show()

n_frames = 30

indices = np.linspace(0,len(df1)-1, n_frames,dtype=int)

plot3d(df1, df2, df3,df4, frame_indices=indices)