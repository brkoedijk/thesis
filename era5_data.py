import cdsapi
import os

client = cdsapi.Client()
dataset = "reanalysis-era5-single-levels-timeseries"

# The 15-node grid spanning NW Europe (UK, NL, DE, DK)
# Format: name, latitude, longitude
grid_coordinates = [
    # Row 1: North (Scotland, No. North Sea, Denmark)
    {"name": "Site_01_UK_North", "lat": 56.5, "lon": -3.5},
    {"name": "Site_02_NS_North", "lat": 56.5, "lon": 4.5},
    {"name": "Site_03_DK_North", "lat": 56.5, "lon": 10.0},
    
    # Row 2: Mid-North (Wales/Ireland Sea, Mid North Sea, NL/DE Coast, DE Inland, DE East)
    {"name": "Site_04_UK_West",  "lat": 53.0, "lon": -3.5},
    {"name": "Site_05_NS_Central", "lat": 53.0, "lon": 2.0},
    {"name": "Site_06_NL_DE_Coast", "lat": 53.0, "lon": 6.5},
    {"name": "Site_07_DE_North", "lat": 53.0, "lon": 11.5},
    {"name": "Site_08_DE_East",  "lat": 53.0, "lon": 14.5},
    
    # Row 3: Mid-South (So. UK, English Channel, Benelux/West DE, Central DE)
    {"name": "Site_09_UK_South", "lat": 50.5, "lon": -3.5},
    {"name": "Site_10_Channel",  "lat": 50.5, "lon": 2.0},
    {"name": "Site_11_DE_West",   "lat": 50.5, "lon": 6.5},
    {"name": "Site_12_DE_Central","lat": 50.5, "lon": 11.5},
    
    # Row 4: South (France/Border, SW Germany, SE Germany)
    {"name": "Site_13_FR_North",  "lat": 48.5, "lon": 2.0},
    {"name": "Site_14_DE_SouthW", "lat": 48.5, "lon": 8.0},
    {"name": "Site_15_DE_SouthE", "lat": 48.5, "lon": 13.5}
]

output_dir = "era5_timeseries_locations"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Starting batch download for {len(grid_coordinates)} locations...")

for site in grid_coordinates:
    target_file = f"{output_dir}/{site['name']}_2000_2023.nc"
    
    # Check if we already have it to avoid re-downloading if script restarts
    if os.path.exists(target_file):
        print(f"-> Skipping {site['name']}, file already exists.")
        continue

    request = {
        "variable": ["100m_u_component_of_wind", "100m_v_component_of_wind"],
        "location": {"longitude": site['lon'], "latitude": site['lat']},
        "date": "2000-01-01/2023-12-31",
        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    print(f"-> Requesting {site['name']} (Lat: {site['lat']}, Lon: {site['lon']})...")
    try:
        client.retrieve(dataset, request).download(target_file)
        print(f"✓ {site['name']} download complete.")
    except Exception as e:
        print(f"✗ Failed to download {site['name']}: {e}")

print("\nAll downloads finished")