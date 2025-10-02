import xarray as xr
import numpy as np
import os
import glob

# Folder paths
u_dir = "wind-u"
v_dir = "wind-v"

# Get all U wind files
u_files = sorted(glob.glob(os.path.join(u_dir, "*.nc")))

for u_file in u_files:
    # Find matching V wind file by replacing 'wind-u' with 'wind-v' in the path
    v_file = u_file.replace("wind-u", "wind-v")
    if not os.path.exists(v_file):
        print(f"Missing V file for {u_file}, skipping.")
        continue

    # Open datasets
    ds_u = xr.open_dataset(u_file)
    ds_v = xr.open_dataset(v_file)

    # Merge datasets
    ds_merged = xr.merge([ds_u, ds_v])

    # Calculate wind speed (replace 'U2M' and 'V2M' with your actual variable names)
    wind_speed = np.sqrt(ds_merged['U2M']**2 + ds_merged['V2M']**2)
    ds_merged['wind_speed'] = wind_speed
    ds_merged['wind_speed'].attrs['long_name'] = '2-meter Wind Speed'
    ds_merged['wind_speed'].attrs['units'] = 'm s-1'


    # Save to new NetCDF file in 'wind-speed' folder
    out_dir = "wind_speed"
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(u_file).replace(".nc", "_windspeed.nc")
    out_file = os.path.join(out_dir, base_name)
    ds_merged.to_netcdf(out_file)
    print(f"Saved wind speed to {out_file}")

print("All wind speed files generated.")