import requests
import matplotlib
# matplotlib.use("TkAgg")
from config import FOCAL_LENGTH
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from config import (AU_2_M)

params_run ={ "START_TIME": "'2019-11-15 00:00'",
             "STOP_TIME": "'2019-11-15 08:00'",
             "STEP_SIZE": "'28800'"}

params_verify ={"START_TIME": "'2019-11-1'",
                "STOP_TIME": "'2019-12-15'",
                "STEP_SIZE": "1d",}

def download_orbit(command_str, filename, time_params):
    params = {
        "format": "text",
        "COMMAND": f"'{command_str}'",  # e.g., "-21" for SOHO, "DES=3562756;" for 2011 EO40
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'500@10'",  # Sun-centered (use '500@31' for Sun-Earth L1)
        "OUT_UNITS": "'KM-S'",
        "VEC_TABLE": "'2'",  # State vector: X, Y, Z, VX, VY, VZ
        "CSV_FORMAT": "'YES'",
    }
    params.update(time_params)

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


# download_orbit("-21", "soho", params_run)
# download_orbit("31", "L1", params_run)
# download_orbit("DES=2019 XF2;", "2019_XF2",params_run)
# download_orbit("399", "Earth",params_run)

# download_orbit("-21", "soho_ver", params_verify)
# download_orbit("31", "L1_ver", params_verify)
# download_orbit("DES=2019 XF2;", "2019_XF2_ver",params_verify)
# download_orbit("399", "Earth_ver",params_verify)


def observe_asteroid(input, fov1, fov2, single_frame = None, verify = False):

    if not verify:
        df1 = pd.read_csv("../outputs/soho.csv")
        df2 = pd.read_csv("../outputs/2019_XF2.csv")
        df3 = pd.read_csv("../outputs/L1.csv")
    else:
        df1 = pd.read_csv("../outputs/soho_ver.csv")
        df2 = pd.read_csv("../outputs/2019_XF2_ver.csv")
        df3 = pd.read_csv("../outputs/L1_ver.csv")

    r_sc = np.array([df1["x_km"], df1["y_km"], df1["z_km"]])
    r_ast = np.array([df2["x_km"], df2["y_km"], df2["z_km"]])
    r_L1 = np.array([df3["x_km"], df3["y_km"], df3["z_km"]])
    v_L1 = np.array([df3["vx_kms"], df3["vy_kms"], df3["vz_kms"]])

    time_jd = df1["jd"].to_numpy()
    time_seconds = np.round((time_jd - time_jd[0]) * 86400).astype(int)

    if not np.all(np.diff(time_seconds) == 1):
        print("Orbital time steps are not all separated by 1 second.")
        # return

    # Define coordinate system of the SC, similar to LVLH but not fully
    # Always point to Sun but derive other from L1
    # x points sun, z points h
    x_sc = -r_sc / np.linalg.norm(r_sc, axis=0)
    h = np.cross(r_L1, v_L1, axis=0)
    z_sc = h - np.sum(h * x_sc, axis=0) * x_sc
    z_sc /= np.linalg.norm(z_sc, axis=0)
    y_sc = np.cross(z_sc, x_sc, axis=0)
    z_sc = np.cross(x_sc, y_sc, axis=0)

    M = np.stack([x_sc.T, y_sc.T, z_sc.T], axis=-1)

    # Relative asteroid position
    dr = (r_ast - r_sc).T

    limit_hor = np.tan(np.deg2rad(fov1 / 2))
    limit_ver = np.tan(np.deg2rad(fov2 / 2))
    focal_len = FOCAL_LENGTH

    os.makedirs("../outputs/images", exist_ok=True)

    exposure_ends = np.cumsum(input[:, 0])
    exposure_starts = np.insert(exposure_ends[:-1], 0, 0)

    if single_frame is not None:
        exposure_starts = np.asarray(single_frame)
        exposure_ends = exposure_starts + input[:, 0]

    for i, (start, end) in enumerate(zip(exposure_starts, exposure_ends)):
        # print("inter: ", i)

        exposure, rot_y, rot_x = input[i]

        # Find orbital samples belonging to this exposure
        indices = np.where((time_seconds >= start) & (time_seconds < end))[0]
        if len(indices) == 0:
            continue

        rot_sensor = R.from_euler('x', rot_x, degrees=True) * R.from_euler('y', rot_y, degrees=True)

        # Project asteroid at every orbital sample during exposure
        dr_sc = np.array([M[j].T @ dr[j] for j in indices])
        dr_sensor = rot_sensor.inv().apply(dr_sc)

        visible = dr_sensor[:, 0] > 0
        dr_sensor = dr_sensor[visible]

        if len(dr_sensor) == 0:
            continue

        hor = dr_sensor[:, 1] / dr_sensor[:, 0]
        ver = dr_sensor[:, 2] / dr_sensor[:, 0]

        mask = (np.abs(hor) < limit_hor) & (np.abs(ver) < limit_ver)

        if not np.any(mask):
            continue

        x = focal_len * hor[mask]
        y = focal_len * ver[mask]

        fig, ax = plt.subplots(figsize=(6, 8))
        ax.scatter(x, y, marker='o')
        ax.set_xlim(-focal_len * limit_hor, focal_len * limit_hor)
        ax.set_ylim(-focal_len * limit_ver, focal_len * limit_ver)
        ax.set_xlabel("Sensor Y")
        ax.set_ylabel("Sensor Z")
        ax.set_title(f"Exposure {i}: {exposure} seconds\n"
                     f" attitude:{rot_x}, {rot_y} \n"
                     f"t_start = {start}")
        ax.set_aspect('equal')
        ax.grid()

        fig.savefig(f"../outputs/images/image_{i:05d}.png", dpi=150)
        plt.close(fig)

        print("Detected")
        print("x movement: ",np.max(x) - np.min(x))
        print("y: ",np.max(y)-np.min(y))


    return {
        "trajectories": [df1, df2, df3],
        "r_sc": r_sc,
        "r_ast": r_ast,
        "M": M,
        "time_seconds": time_seconds
    }


def plot_orbits(data, attitude=(0, 0), n=7):
    matplotlib.use("TkAgg")

    df1, df2, df3 = data["trajectories"]
    r_sc, r_ast, M = data["r_sc"], data["r_ast"], data["M"]

    rot_y, rot_x = attitude
    rot_sensor = R.from_euler('x', rot_x, degrees=True) * R.from_euler('y', rot_y, degrees=True)

    # Sensor axes in inertial coordinates
    sensor_axes = M @ rot_sensor.apply(np.eye(3)).T

    frame_indices = np.linspace(0, len(r_sc.T) - 1, n, dtype=int)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Orbital trajectories
    for df, name in zip([df1, df2, df3], ["SOHO", "Asteroid", "L1"]):
        ax.plot(df["x_km"], df["y_km"], df["z_km"], label=name)

    # Spacecraft coordinate systems and asteroid connections
    all_positions = np.hstack((r_sc, r_ast, data["trajectories"][2][["x_km", "y_km", "z_km"]].to_numpy().T))
    max_range = np.max(np.ptp(all_positions, axis=1))
    axis_length = max_range * 0.05

    for idx in frame_indices:
        origin = r_sc[:, idx]

        ax.quiver(*origin, *sensor_axes[idx, :, 0], length=axis_length, normalize=True, color="darkred", label="Sensor X" if idx == frame_indices[0] else None)
        ax.quiver(*origin, *sensor_axes[idx, :, 1], length=axis_length, normalize=True, color="darkgreen", label="Sensor Y" if idx == frame_indices[0] else None)
        ax.quiver(*origin, *sensor_axes[idx, :, 2], length=axis_length, normalize=True, color="darkblue", label="Sensor Z" if idx == frame_indices[0] else None)

        ax.plot([r_sc[0, idx], r_ast[0, idx]], [r_sc[1, idx], r_ast[1, idx]], [r_sc[2, idx], r_ast[2, idx]], color="gray", linestyle="--", linewidth=0.8)

    # Equal axis scaling
    midpoint = (np.max(all_positions, axis=1) + np.min(all_positions, axis=1)) / 2
    half_range = max_range / 2

    ax.set_xlim(midpoint[0] - half_range, midpoint[0] + half_range)
    ax.set_ylim(midpoint[1] - half_range, midpoint[1] + half_range)
    ax.set_zlim(midpoint[2] - half_range, midpoint[2] + half_range)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend()
    ax.set_title("Orbital trajectories and sensor coordinate system")
    plt.tight_layout()


    plt.show()

