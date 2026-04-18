import zipfile
import xarray as xr
import pandas as pd
import numpy as np
import os
#/Users/bartkoedijk/Bestanden/Thesis Code/deephedging/era5_timeseries_locations
data_dir = "/Users/bartkoedijk/Bestanden/Thesis Code/deephedging/era5_timeseries_locations/"
master_df = pd.DataFrame()
v_ci, v_r, v_co = 3.0, 12.0, 25.0

files = [f for f in os.listdir(data_dir) if f.startswith("Site_") and f.endswith(".nc")]
files.sort()

print(f"Found {len(files)} sites. Start processing...")

for file_name in files:
    site_id = "_".join(file_name.split("_")[:2])
    file_path = os.path.join(data_dir, file_name)

    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            internal_file_name = z.namelist()[0]
            z.extractall(path=data_dir)
            actual_nc_path = os.path.join(data_dir, internal_file_name)
        
        with xr.open_dataset(actual_nc_path, engine="h5netcdf") as xrds:
            temp_df = xrds[["u100", "v100"]].to_dataframe()

            if master_df.empty:
                master_df.index = temp_df.index

            temp_df["V"] = np.sqrt(temp_df["u100"]**2 + temp_df["v100"]**2)

            conditions = [
            (temp_df["V"] < v_ci),
            (temp_df["V"] >= v_ci) & (temp_df["V"] < v_r),
            (temp_df["V"] >= v_r) & (temp_df["V"] < v_co),
            (temp_df["V"] >= v_co)
            ]

            choices = [
                0.0,
                (temp_df["V"]**3 - v_ci**3) / (v_r**3 - v_ci**3),
                1.0,
                0.0
            ]
            master_df[site_id] = np.select(conditions, choices, default=0.0)
        os.remove(actual_nc_path)
        print(f"{site_id} processed succesfully.")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

print("\nProcessing complete")
print(f"Final matrix shape:{master_df.shape}")
print(master_df.head())

master_df.to_csv("master_capacity_factors_2000_2023.csv")

