#!/usr/bin/env python3
"""
Test script to verify coordinate handling for negative longitude/latitude values
"""
import requests
import json

# Test cases with various coordinate combinations
test_cases = [
    {"name": "New York (negative lon)", "lat": 40.7128, "lon": -74.0060},
    {"name": "London (positive coords)", "lat": 51.5074, "lon": -0.1278},
    {"name": "Sydney (positive lon)", "lat": -33.8688, "lon": 151.2093},
    {"name": "Buenos Aires (negative lat, negative lon)", "lat": -34.6037, "lon": -58.3816},
    {"name": "Cape Town (negative lat, positive lon)", "lat": -33.9249, "lon": 18.4241},
]

# Test date
test_date = "15-06-2020"

print("🧪 Testing coordinate handling for negative values")
print("=" * 60)

for test_case in test_cases:
    print(f"\n📍 Testing: {test_case['name']}")
    print(f"   Coordinates: ({test_case['lat']}, {test_case['lon']})")
    
    # Test with local server (adjust URL as needed)
    url = "http://localhost:10000/climate"
    params = {
        "date": test_date,
        "lat": test_case['lat'],
        "lon": test_case['lon']
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if coordinate conversion info is in metadata
            if "metadata" in data and "coordinates" in data["metadata"]:
                coord_info = data["metadata"]["coordinates"]
                print(f"   ✅ Request successful")
                print(f"   📥 Requested: ({coord_info['requested']['lat']}, {coord_info['requested']['lon']})")
                print(f"   📤 Dataset format: ({coord_info['dataset_format']['lat']}, {coord_info['dataset_format']['lon']})")
            else:
                print(f"   ✅ Request successful (no coordinate info in response)")
            
            # Show some sample values
            if "values" in data:
                values = data["values"]
                print(f"   🌡️  Temperature: {values.get('temperature', 'N/A')} °C")
                print(f"   💨 Wind Speed: {values.get('wind_speed', 'N/A')} km/h")
            
        else:
            print(f"   ❌ Request failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"   ⚠️  Server not running - start the Flask app first")
        break
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("💡 Key Points:")
print("   • Latitude: -90 to +90 (standard format used by datasets)")
print("   • Longitude: Automatically converted from -180/+180 to 0/360 format")
print("   • NASA MERRA-2 datasets use 0-360 longitude coordinate system")
print("   • Both positive and negative input coordinates should work correctly")