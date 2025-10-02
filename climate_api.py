from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import xarray as xr
import numpy as np
from datetime import datetime
import os
# Updated Gemini API imports
import google.generativeai as genai
import re
import json

load_dotenv()
app = Flask(__name__)

# Get absolute base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed file paths for datasets
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
GRAPH_FOLDER = os.path.join(BASE_DIR, "static", "graphs")

# Create directories if they don't exist
os.makedirs(GRAPH_FOLDER, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

app.config['GRAPH_FOLDER'] = GRAPH_FOLDER

# Load datasets with absolute paths
try:
    wind_ds = xr.open_dataset(os.path.join(DATASETS_DIR, "wind_speed_monthly_avg_2000_2025.nc"))
    temp_ds = xr.open_dataset(os.path.join(DATASETS_DIR, "temp_monthly_avg_2000_2025.nc"))
    precip_ds = xr.open_dataset(os.path.join(DATASETS_DIR, "precip_monthly_avg_2000_2025.nc"))
    snow_ds = xr.open_dataset(os.path.join(DATASETS_DIR, "snowfall_monthly_avg_2000_2025.nc"))
    aqi_ds = xr.open_dataset(os.path.join(DATASETS_DIR, "aqi_monthly_avg_2000_2025.nc"))
    print("✅ All datasets loaded successfully")
except FileNotFoundError as e:
    print(f"❌ Dataset file not found: {e}")
    print(f"📁 Looking in: {DATASETS_DIR}")
    # Initialize as None to avoid crashes
    wind_ds = temp_ds = precip_ds = snow_ds = aqi_ds = None

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")  # Loaded from .env

# Configure the Gemini API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("⚠️  GOOGLE_API_KEY not found in environment variables")

def get_days_in_month(month, year):
    """Get number of days in a month, accounting for leap years"""
    if month == 2:
        return 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
    elif month in [4, 6, 9, 11]:
        return 30
    else:
        return 31
    
def parse_ai_response(ai_text):
    """Parse AI response to extract JSON from markdown code blocks"""
    try:
        # Remove markdown code blocks if present
        cleaned_text = re.sub(r'```json\s*|\s*```', '', ai_text).strip()
        
        # Parse JSON
        parsed_data = json.loads(cleaned_text)
        return parsed_data
    except json.JSONDecodeError as e:
        # If JSON parsing fails, return the original text
        print(f"JSON parsing error: {e}")
        return {"raw_response": ai_text, "error": "Failed to parse AI response"}

# Add health check endpoint (required for Render)
@app.route('/')
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "Climate API is running",
        "datasets_loaded": all(ds is not None for ds in [wind_ds, temp_ds, precip_ds, snow_ds, aqi_ds])
    })

@app.route('/climate', methods=['GET'])
def get_climate():
    # Check if datasets are loaded
    if any(ds is None for ds in [wind_ds, temp_ds, precip_ds, snow_ds, aqi_ds]):
        return jsonify({"error": "Climate datasets not available"}), 503
    
    date_str = request.args.get('date')  # DD-MM-YYYY
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('lon'))
    rh = request.args.get('rh')
    if rh is not None:
        try:
            rh = float(rh)
        except:
            rh = None
    else:
        rh = None

    # Normalize longitude (Panoply data is often 0–360 instead of -180–180)
    if lon < 0:
        lon = lon + 360

    # Date parsing
    try:
        date_obj = datetime.strptime(date_str, '%d-%m-%Y')
        month = date_obj.month
        year = date_obj.year
    except ValueError:
        return jsonify({"error": "Invalid date format. Use DD-MM-YYYY"}), 400

    # Prepare for graph generation
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    import matplotlib.pyplot as plt
    graph_urls = {}
    variable_info = [
        ("temperature", "Temperature (°C)", lambda x, days: x - 273.15, "temperature", "T2M"),
        ("total_precipitation", "Total Precipitation (mm/month)", lambda x, days: x * days * 24 * 3600, "total_precipitation", "PRECTOT"),
        ("wind_speed", "Wind Speed (km/h)", lambda x, days: x * 3.6, "wind_speed", "wind_speed"),
        ("snowfall", "Snowfall (mm/month)", lambda x, days: x * days * 24 * 3600, "snowfall", "PRECSNO"),
        ("aqi", "Aerosol Optical Thickness (unitless)", lambda x, days: x, "aqi", "TOTEXTTAU"),
    ]
    
    import uuid
    import threading
    import time
    temp_files = []
    
    for var, label, convert, folder, varname in variable_info:
        years = list(range(2000, 2026))
        values = []
        prefixes = ["MERRA2_200", "MERRA2_300", "MERRA2_400", "MERRA2_401"]
        
        for yr in years:
            try:
                days = get_days_in_month(month, yr)
                month_str = f"{month:02d}"
                nc_path = None
                file_found = False
                
                for prefix in prefixes:
                    if var == "aqi":
                        nc_filename = f"{prefix}.tavgM_2d_aer_Nx.{yr}{month_str}.SUB.nc"
                    elif var == "snowfall":
                        if yr >= 2011 and (prefix == "MERRA2_400" or prefix == "MERRA2_401"):
                            nc_filename = f"{prefix}.tavgM_2d_flx_Nx.{yr}{month_str}.SUB.nc"
                        else:
                            nc_filename = f"{prefix}.tavgM_2d_slv_Nx.{yr}{month_str}.SUB.nc"
                    elif var == "total_precipitation":
                        nc_filename = f"{prefix}.tavgM_2d_flx_Nx.{yr}{month_str}.SUB.nc"
                    elif var == "wind_speed":
                        nc_filename = f"{prefix}.tavgM_2d_slv_Nx.{yr}{month_str}.SUB_windspeed.nc"
                    else:
                        nc_filename = f"{prefix}.tavgM_2d_slv_Nx.{yr}{month_str}.SUB.nc"
                    
                    # Fixed file path - look in datasets directory
                    nc_path_try = os.path.join(DATASETS_DIR, folder, nc_filename)
                    if os.path.exists(nc_path_try):
                        nc_path = nc_path_try
                        file_found = True
                        break
                
                if not file_found or nc_path is None:
                    values.append(np.nan)
                    continue
                    
                ds = xr.open_dataset(nc_path)
                lon_val = lon
                if hasattr(ds, 'lon') and np.any(ds.lon.values > 180):
                    if lon_val < 0:
                        lon_val = lon_val + 360
                try:
                    val_arr = ds[varname].interp(lat=lat, lon=lon_val, method="linear").values
                    val = float(val_arr.item() if hasattr(val_arr, 'item') else val_arr[()])
                except Exception:
                    val_arr = ds[varname].interp(lat=lat, lon=lon_val).values
                    val = float(val_arr.item() if hasattr(val_arr, 'item') else val_arr[()])
                val_converted = convert(val, days)
                values.append(val_converted)
                ds.close()
                
            except Exception as e:
                print(f"Error processing {var} for year {yr}: {e}")
                values.append(np.nan)
        
        # Create graph
        plt.figure(figsize=(8, 4))
        plt.plot(years, values, marker='o', color='royalblue')
        plt.title(f"{label} for Month {month:02d} at ({lat}, {lon})")
        plt.xlabel("Year")
        plt.ylabel(label)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        filename = f"{var}_{month:02d}_{lat:.2f}_{lon:.2f}_{uuid.uuid4().hex}.png"
        filepath = os.path.join(GRAPH_FOLDER, filename)
        plt.savefig(filepath, format='png')
        plt.close()
        temp_files.append(filepath)
        url = f"/static/graphs/{filename}"
        graph_urls[var] = url

    result = {}
    values = {}
    days_in_month = get_days_in_month(month, year)
    seconds_in_month = days_in_month * 24 * 3600

    # Wind speed (m/s to km/h)
    try:
        wind = wind_ds["wind_speed"].interp(month=month, lat=lat, lon=lon, method="linear").values.item()
        wind_kmh = wind * 3.6
        values["wind_speed"] = float(f"{wind_kmh:.2f}")
    except Exception as e:
        print(f"Wind error: {e}")
        values["wind_speed"] = None

    # Temperature (Kelvin to Celsius)
    try:
        temp = temp_ds["T2M"].interp(month=month, lat=lat, lon=lon, method="linear").values.item()
        temp_c = temp - 273.15
        values["temperature"] = float(f"{temp_c:.2f}")
    except Exception as e:
        print(f"Temperature error: {e}")
        values["temperature"] = None

    # Total precipitation (kg m⁻² s⁻¹ to mm/month)
    try:
        precip = precip_ds["PRECTOT"].interp(month=month, lat=lat, lon=lon, method="linear").values.item()
        precip_mm = precip * seconds_in_month
        values["total_precipitation"] = float(f"{precip_mm:.2f}")
    except Exception as e:
        print(f"Precipitation error: {e}")
        values["total_precipitation"] = None

    # Snowfall (kg m⁻² s⁻¹ to mm/month)
    try:
        snow = snow_ds["PRECSNO"].interp(month=month, lat=lat, lon=lon, method="linear").values.item()
        snow_mm = snow * seconds_in_month
        values["snowfall"] = float(f"{snow_mm:.7f}")
    except Exception as e:
        print(f"Snowfall error: {e}")
        values["snowfall"] = None

    # AQI (Aerosol Optical Thickness proxy)
    try:
        aqi = aqi_ds["TOTEXTTAU"].interp(month=month, lat=lat, lon=lon, method="linear").values.item()
        values["aqi"] = float(f"{aqi:.2f}")
    except Exception as e:
        print(f"AQI error: {e}")
        values["aqi"] = None

    # Metadata
    result["values"] = values
    result["metadata"] = {
        "units": {
            "temperature": "Celsius",
            "total_precipitation": "mm/month",
            "wind_speed": "km/h",
            "snowfall": "mm/month",
            "aqi": "unitless (AOT)"
        },
        "sources": {
            "temperature": "https://disc.gsfc.nasa.gov/data-access",
            "total_precipitation": "https://disc.gsfc.nasa.gov/data-access",
            "wind_speed": "https://disc.gsfc.gsfc.nasa.gov/data-access",
            "snowfall": "https://disc.gsfc.nasa.gov/data-access",
            "aqi": "https://disc.gsfc.nasa.gov/data-access"
        }
    }
    result["graphs"] = graph_urls

    # --- AI Weather Classification ---
    ai_prompt = f"""
You are an intelligent weather assistant.  
Your task is to classify the weather at a given location and time into one or more of the following categories:

- Very Hot  
- Very Cold  
- Very Windy  
- Very Wet  
- Very Uncomfortable  
- Normal (if none of the above conditions are met)  

### Classification Rules (for AI reference):
- Very Hot → Temperature > 35°C  
- Very Cold → Temperature < 5°C  
- Very Windy → Wind Speed > 60 km/h  
- Very Wet → Total Precipitation > 20 mm/month  
- Very Uncomfortable → Temperature > 30°C OR Temperature > 35°C AND Wind Speed < 2 m/s  
- Air Quality - Clear → AOD < 0.1
- Air Quality - Moderate → 0.1 ≤ AOD ≤ 0.3
- Air Quality - Polluted/Hazy → AOD > 0.3
- Normal → If none of the above thresholds are met  

### Input Data (with units):
- AQI: {values['aqi']} (unitless, Aerosol Optical Thickness)  
- Snowfall: {values['snowfall']} mm/month  
- Temperature: {values['temperature']} °C  
- Total Precipitation: {values['total_precipitation']} mm/month  
- Wind Speed: {values['wind_speed']} km/h  

### Instructions:
1. Analyze the input data and determine which categories apply. Multiple categories can be applied.  
2. Explain **briefly but clearly** why each category applies, citing the values.  
3. Return the output **strictly in JSON format** as follows:

Example output:
{{
    "classification": ["Very Hot", "Very Uncomfortable"],
    "explanation": "Temperature is 36°C which is above 35°C (very hot). Relative Humidity is 82%, making it very uncomfortable."
}}
"""

    ai_response = None
    parsed_ai_response = {}
    try:
        if GOOGLE_API_KEY:
            model = genai.GenerativeModel('gemini-2.5-flash')  # or 'gemini-1.5-pro'
            response = model.generate_content(ai_prompt)
            ai_response = response.text
            # Parse the AI response to extract clean JSON
            parsed_ai_response = parse_ai_response(ai_response)
        else:
            parsed_ai_response = {"error": "Google API key not configured"}
    except Exception as e:
        parsed_ai_response = {"error": f"AI classification failed: {str(e)}"}

    result["ai_classification"] = parsed_ai_response
    
    response = jsonify(result)
    # Schedule deletion of temp files after 1 hour
    def delayed_cleanup(files):
        def delete_files():
            time.sleep(3600)  # 1 hour
            for f in files:
                try:
                    os.remove(f)
                except Exception:
                    pass
        threading.Thread(target=delete_files, daemon=True).start()
    response.call_on_close(lambda: delayed_cleanup(temp_files))
    return response

# Serve static files
@app.route('/static/graphs/<path:filename>')
def serve_graph(filename):
    return send_from_directory(GRAPH_FOLDER, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)