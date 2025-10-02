import xarray as xr
import dask  # Required for open_mfdataset
import os

# Path to precipitation files
precip_dir = "Snowfall"
precip_pattern = os.path.join(precip_dir, "*.nc")

# Open all NetCDF files together (stack along time dimension)
ds = xr.open_mfdataset(precip_pattern, combine="by_coords")

# Group by month and average across years
monthly_climatology = ds.groupby("time.month").mean("time")

# Save to a new NetCDF file
monthly_climatology.to_netcdf("snowfall_monthly_avg_2000_2025.nc")

print("Monthly climatology saved to temp_monthly_avg_2000_2025.nc")

