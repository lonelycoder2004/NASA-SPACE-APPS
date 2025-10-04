from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import xarray as xr
import numpy as np
from datetime import datetime
import os
import google.generativeai as genai
import re
import json
import concurrent.futures
import threading
import time
import uuid
import hashlib
from functools import lru_cache

load_dotenv()

# Get absolute base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed file paths for datasets
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
GRAPH_FOLDER = os.path.join(BASE_DIR, "static", "graphs")

# Create directories if they don't exist
os.makedirs(GRAPH_FOLDER, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

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

def generate_single_graph(var_info, month, lat, lon, years):
    """Generate a single graph - to be run in parallel"""
    var, label, convert, folder, varname = var_info
    
    # Import matplotlib inside function to avoid threading issues
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    values = []
    
    for yr in years:
        try:
            days = get_days_in_month(month, yr)
            month_str = f"{month:02d}"
            nc_path = None
            file_found = False
            
            for prefix in ["MERRA2_200", "MERRA2_300", "MERRA2_400", "MERRA2_401"]:
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
    
    return var, f"/static/graphs/{filename}", filepath

def extract_single_value(task, month, lat, lon, days_in_month):
    """Extract single climate value - to be run in parallel"""
    name, ds, conversion_func, varname = task
    try:
        value = ds[varname].interp(month=month, lat=lat, lon=lon, method="linear").values.item()
        converted_value = conversion_func(value, days_in_month)
        return name, float(f"{converted_value:.2f}")
    except Exception as e:
        print(f"{name} error: {e}")
        return name, None

def generate_ai_classification(values):
    """Generate AI classification - runs after data extraction"""
    if not GOOGLE_API_KEY:
        return {"error": "Google API key not configured"}
    
    # Check if we have enough data for AI classification
    if not values or all(v is None for v in values.values()):
        return {"error": "Insufficient data for AI classification"}
    
    ai_prompt = f"""
You are an intelligent weather assistant.  
Your task is to classify the weather at a given location and time into one or more of the following categories:

- Very Hot  
- Very Cold  
- Very Windy  
- Very Wet  
- Very Snowy  
- Very Uncomfortable  
- Normal (if none of the above conditions are met)  

### Classification Rules (for AI reference):
- Very Hot → Temperature > 35°C  
- Very Cold → Temperature < 0°C  
- Very Windy → Wind Speed ≥ 40 km/h  
- Very Wet → Total Precipitation > 200 mm/month  
- Very Snowy → Snowfall > 100 mm/month  
- **Very Uncomfortable** → Triggered when **multiple mild stress factors combine**, such as: 
  - OR (Temperature > 30°C **and** Wind Speed < 5 km/h)  
  - OR (Temperature > 28°C **and** AOD > 0.3)  
  - OR (Temperature between 25-30°C **and** Precipitation > 150 mm/month)  
  - *Reasoning:* High temperature combined with humidity, poor air quality, stagnant air, or sticky wet conditions makes the atmosphere feel heavy, sweaty, and exhausting even if it's not extreme.
- Air Quality - Polluted/Hazy → AOD > 0.3  
- Normal → If none of the above thresholds are met  

### Input Data (with units):
- AQI: {values.get('aqi')} (unitless, Aerosol Optical Thickness)  
- Snowfall: {values.get('snowfall')} mm/month  
- Temperature: {values.get('temperature')} °C  
- Total Precipitation: {values.get('total_precipitation')} mm/month  
- Wind Speed: {values.get('wind_speed')} km/h  

### Instructions:
1. Analyze the input data carefully and decide which categories apply.  
2. You may apply multiple categories if the data matches more than one condition.  
3. When explaining, write **natural, meaningful sentences** that describe why each category applies, not just by listing numbers.  
4. If no specific category applies, classify the weather as "Normal".  
5. Return the output **strictly in JSON format** as follows:

Example output:
{{
    "classification": ["Very Hot", "Very Uncomfortable"],
    "explanation": "The temperature is 36°C, which is above the 35°C threshold, indicating very hot conditions. Additionally, the calm wind speed of 1.2 km/h makes the atmosphere feel stuffy and uncomfortable."
}}

### Notes for Explanation Style:
- Combine reasoning naturally (e.g., "strong winds of 45 km/h make the weather very windy" instead of "Wind speed = 45 km/h → very windy").  
- Mention both value and threshold in a sentence for clarity.  
- Use connecting phrases like "as a result", "which indicates", or "therefore" to make the explanation coherent.  
"""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')  
        response = model.generate_content(ai_prompt)
        ai_response = response.text
        return parse_ai_response(ai_response)
    except Exception as e:
        return {"error": f"AI classification failed: {str(e)}"}

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

    days_in_month = get_days_in_month(month, year)
    years = list(range(2000, 2026))
    
    result = {}
    temp_files = []
    
    # Define tasks for parallel execution
    variable_info = [
        ("temperature", "Temperature (°C)", lambda x, days: x - 273.15, "temperature", "T2M"),
        ("total_precipitation", "Total Precipitation (mm/month)", lambda x, days: x * days * 24 * 3600, "total_precipitation", "PRECTOT"),
        ("wind_speed", "Wind Speed (km/h)", lambda x, days: x * 3.6, "wind_speed", "wind_speed"),
        ("snowfall", "Snowfall (mm/month)", lambda x, days: x * days * 24 * 3600 * 10, "snowfall", "PRECSNO"),
        ("aqi", "Aerosol Optical Thickness (unitless)", lambda x, days: x, "aqi", "TOTEXTTAU"),
    ]
    
    # Data extraction tasks
    data_tasks = [
        ("wind_speed", wind_ds, lambda x, days: x * 3.6, "wind_speed"),
        ("temperature", temp_ds, lambda x, days: x - 273.15, "T2M"),
        ("total_precipitation", precip_ds, lambda x, days: x * days * 24 * 3600, "PRECTOT"),
        ("snowfall", snow_ds, lambda x, days: x * days * 24 * 3600 * 10, "PRECSNO"),
        ("aqi", aqi_ds, lambda x, days: x, "TOTEXTTAU"),
    ]

    # Run operations in optimized parallel stages
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # STAGE 1: Run graph generation and data extraction in parallel (independent tasks)
        graph_futures = {
            executor.submit(generate_single_graph, var_info, month, lat, lon, years): var_info[0]
            for var_info in variable_info
        }
        
        data_futures = {
            executor.submit(extract_single_value, task, month, lat, lon, days_in_month): task[0]
            for task in data_tasks
        }
        
        # Collect graph results
        graph_urls = {}
        for future in concurrent.futures.as_completed(graph_futures):
            var_name = graph_futures[future]
            try:
                var, url, filepath = future.result()
                graph_urls[var] = url
                temp_files.append(filepath)
            except Exception as e:
                print(f"Graph generation error for {var_name}: {e}")
                graph_urls[var_name] = None
        
        # Collect data results
        values = {}
        for future in concurrent.futures.as_completed(data_futures):
            var_name = data_futures[future]
            try:
                name, value = future.result()
                values[name] = value
            except Exception as e:
                print(f"Data extraction error for {var_name}: {e}")
                values[var_name] = None
        
        # STAGE 2: Run AI classification ONLY after data extraction is complete
        ai_future = executor.submit(generate_ai_classification, values)
        
        try:
            ai_result = ai_future.result(timeout=30)  # 30 second timeout for AI
        except concurrent.futures.TimeoutError:
            ai_result = {"error": "AI classification timed out"}
        except Exception as e:
            ai_result = {"error": f"AI classification failed: {str(e)}"}
    
    # Build final result
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
    result["ai_classification"] = ai_result
    
    response = jsonify(result)
    
    # Schedule cleanup of temporary files
    def delayed_cleanup(files):
        def delete_files():
            time.sleep(3600)  # 1 hour
            for f in files:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except Exception as e:
                    print(f"Error cleaning up file {f}: {e}")
        threading.Thread(target=delete_files, daemon=True).start()
    
    response.call_on_close(lambda: delayed_cleanup(temp_files))
    return response

# Serve static files
@app.route('/static/graphs/<path:filename>')
def serve_graph(filename):
    return send_from_directory(GRAPH_FOLDER, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True)