#!/usr/bin/env python3
"""
Oregon Counties Geographic Coordinates
====================================

This module provides geographic coordinates (latitude/longitude) for all 36 Oregon counties
to enable proper mapping and geospatial analysis in Tableau Public and other visualization tools.

Coordinates are based on county seat locations and approximate county centroids.
"""

import pandas as pd
import os
from typing import Dict, Tuple

class OregonCountiesGeographic:
    """Geographic coordinates for Oregon counties"""
    
    def __init__(self):
        self.county_coordinates = self._get_county_coordinates()
        
    def _get_county_coordinates(self) -> Dict[str, Dict[str, float]]:
        """
        Get geographic coordinates for all Oregon counties
        
        Returns:
            Dictionary mapping county FIPS to coordinates
        """
        return {
            "001": {  # Baker County
                "county_name": "Baker County",
                "county_seat": "Baker City",
                "latitude": 44.7749,
                "longitude": -117.8294,
                "centroid_lat": 44.7749,
                "centroid_lng": -117.8294
            },
            "003": {  # Benton County
                "county_name": "Benton County",
                "county_seat": "Corvallis",
                "latitude": 44.5646,
                "longitude": -123.2620,
                "centroid_lat": 44.5646,
                "centroid_lng": -123.2620
            },
            "005": {  # Clackamas County
                "county_name": "Clackamas County",
                "county_seat": "Oregon City",
                "latitude": 45.3573,
                "longitude": -122.6068,
                "centroid_lat": 45.3573,
                "centroid_lng": -122.6068
            },
            "007": {  # Clatsop County
                "county_name": "Clatsop County",
                "county_seat": "Astoria",
                "latitude": 46.1879,
                "longitude": -123.8313,
                "centroid_lat": 46.1879,
                "centroid_lng": -123.8313
            },
            "009": {  # Columbia County
                "county_name": "Columbia County",
                "county_seat": "St. Helens",
                "latitude": 45.8637,
                "longitude": -122.8140,
                "centroid_lat": 45.8637,
                "centroid_lng": -122.8140
            },
            "011": {  # Coos County
                "county_name": "Coos County",
                "county_seat": "Coquille",
                "latitude": 43.1773,
                "longitude": -124.1982,
                "centroid_lat": 43.1773,
                "centroid_lng": -124.1982
            },
            "013": {  # Crook County
                "county_name": "Crook County",
                "county_seat": "Prineville",
                "latitude": 44.3029,
                "longitude": -120.8442,
                "centroid_lat": 44.3029,
                "centroid_lng": -120.8442
            },
            "015": {  # Curry County
                "county_name": "Curry County",
                "county_seat": "Gold Beach",
                "latitude": 42.4072,
                "longitude": -124.4168,
                "centroid_lat": 42.4072,
                "centroid_lng": -124.4168
            },
            "017": {  # Deschutes County
                "county_name": "Deschutes County",
                "county_seat": "Bend",
                "latitude": 44.0582,
                "longitude": -121.3153,
                "centroid_lat": 44.0582,
                "centroid_lng": -121.3153
            },
            "019": {  # Douglas County
                "county_name": "Douglas County",
                "county_seat": "Roseburg",
                "latitude": 43.2165,
                "longitude": -123.3417,
                "centroid_lat": 43.2165,
                "centroid_lng": -123.3417
            },
            "021": {  # Gilliam County
                "county_name": "Gilliam County",
                "county_seat": "Condon",
                "latitude": 45.2343,
                "longitude": -120.1853,
                "centroid_lat": 45.2343,
                "centroid_lng": -120.1853
            },
            "023": {  # Grant County
                "county_name": "Grant County",
                "county_seat": "Canyon City",
                "latitude": 44.3897,
                "longitude": -118.9489,
                "centroid_lat": 44.3897,
                "centroid_lng": -118.9489
            },
            "025": {  # Harney County
                "county_name": "Harney County",
                "county_seat": "Burns",
                "latitude": 43.5865,
                "longitude": -119.0543,
                "centroid_lat": 43.5865,
                "centroid_lng": -119.0543
            },
            "027": {  # Hood River County
                "county_name": "Hood River County",
                "county_seat": "Hood River",
                "latitude": 45.7087,
                "longitude": -121.5115,
                "centroid_lat": 45.7087,
                "centroid_lng": -121.5115
            },
            "029": {  # Jackson County
                "county_name": "Jackson County",
                "county_seat": "Medford",
                "latitude": 42.3265,
                "longitude": -122.8756,
                "centroid_lat": 42.3265,
                "centroid_lng": -122.8756
            },
            "031": {  # Jefferson County
                "county_name": "Jefferson County",
                "county_seat": "Madras",
                "latitude": 44.6337,
                "longitude": -121.1295,
                "centroid_lat": 44.6337,
                "centroid_lng": -121.1295
            },
            "033": {  # Josephine County
                "county_name": "Josephine County",
                "county_seat": "Grants Pass",
                "latitude": 42.4390,
                "longitude": -123.3284,
                "centroid_lat": 42.4390,
                "centroid_lng": -123.3284
            },
            "035": {  # Klamath County
                "county_name": "Klamath County",
                "county_seat": "Klamath Falls",
                "latitude": 42.2249,
                "longitude": -121.7817,
                "centroid_lat": 42.2249,
                "centroid_lng": -121.7817
            },
            "037": {  # Lake County
                "county_name": "Lake County",
                "county_seat": "Lakeview",
                "latitude": 42.1889,
                "longitude": -120.3458,
                "centroid_lat": 42.1889,
                "centroid_lng": -120.3458
            },
            "039": {  # Lane County
                "county_name": "Lane County",
                "county_seat": "Eugene",
                "latitude": 44.0521,
                "longitude": -123.0868,
                "centroid_lat": 44.0521,
                "centroid_lng": -123.0868
            },
            "041": {  # Lincoln County
                "county_name": "Lincoln County",
                "county_seat": "Newport",
                "latitude": 44.6368,
                "longitude": -124.0534,
                "centroid_lat": 44.6368,
                "centroid_lng": -124.0534
            },
            "043": {  # Linn County
                "county_name": "Linn County",
                "county_seat": "Albany",
                "latitude": 44.6365,
                "longitude": -123.1059,
                "centroid_lat": 44.6365,
                "centroid_lng": -123.1059
            },
            "045": {  # Malheur County
                "county_name": "Malheur County",
                "county_seat": "Vale",
                "latitude": 43.9821,
                "longitude": -117.6484,
                "centroid_lat": 43.9821,
                "centroid_lng": -117.6484
            },
            "047": {  # Marion County
                "county_name": "Marion County",
                "county_seat": "Salem",
                "latitude": 44.9429,
                "longitude": -122.9786,
                "centroid_lat": 44.9429,
                "centroid_lng": -122.9786
            },
            "049": {  # Morrow County
                "county_name": "Morrow County",
                "county_seat": "Heppner",
                "latitude": 45.4187,
                "longitude": -119.5843,
                "centroid_lat": 45.4187,
                "centroid_lng": -119.5843
            },
            "051": {  # Multnomah County
                "county_name": "Multnomah County",
                "county_seat": "Portland",
                "latitude": 45.5152,
                "longitude": -122.6784,
                "centroid_lat": 45.5152,
                "centroid_lng": -122.6784
            },
            "053": {  # Polk County
                "county_name": "Polk County",
                "county_seat": "Dallas",
                "latitude": 44.9193,
                "longitude": -123.3170,
                "centroid_lat": 44.9193,
                "centroid_lng": -123.3170
            },
            "055": {  # Sherman County
                "county_name": "Sherman County",
                "county_seat": "Moro",
                "latitude": 45.4152,
                "longitude": -120.6892,
                "centroid_lat": 45.4152,
                "centroid_lng": -120.6892
            },
            "057": {  # Tillamook County
                "county_name": "Tillamook County",
                "county_seat": "Tillamook",
                "latitude": 45.4562,
                "longitude": -123.8429,
                "centroid_lat": 45.4562,
                "centroid_lng": -123.8429
            },
            "059": {  # Umatilla County
                "county_name": "Umatilla County",
                "county_seat": "Pendleton",
                "latitude": 45.6721,
                "longitude": -118.7886,
                "centroid_lat": 45.6721,
                "centroid_lng": -118.7886
            },
            "061": {  # Union County
                "county_name": "Union County",
                "county_seat": "La Grande",
                "latitude": 45.3247,
                "longitude": -118.0876,
                "centroid_lat": 45.3247,
                "centroid_lng": -118.0876
            },
            "063": {  # Wallowa County
                "county_name": "Wallowa County",
                "county_seat": "Enterprise",
                "latitude": 45.4265,
                "longitude": -117.2789,
                "centroid_lat": 45.4265,
                "centroid_lng": -117.2789
            },
            "065": {  # Wasco County
                "county_name": "Wasco County",
                "county_seat": "The Dalles",
                "latitude": 45.5946,
                "longitude": -121.1784,
                "centroid_lat": 45.5946,
                "centroid_lng": -121.1784
            },
            "067": {  # Washington County
                "county_name": "Washington County",
                "county_seat": "Hillsboro",
                "latitude": 45.5469,
                "longitude": -122.9839,
                "centroid_lat": 45.5469,
                "centroid_lng": -122.9839
            },
            "069": {  # Wheeler County
                "county_name": "Wheeler County",
                "county_seat": "Fossil",
                "latitude": 44.9971,
                "longitude": -120.0264,
                "centroid_lat": 44.9971,
                "centroid_lng": -120.0264
            },
            "071": {  # Yamhill County
                "county_name": "Yamhill County",
                "county_seat": "McMinnville",
                "latitude": 45.2037,
                "longitude": -123.1990,
                "centroid_lat": 45.2037,
                "centroid_lng": -123.1990
            }
        }
    
    def get_county_coordinates(self, county_fips: str) -> Dict[str, float]:
        """
        Get coordinates for a specific county
        
        Args:
            county_fips: 3-digit county FIPS code
            
        Returns:
            Dictionary with latitude and longitude
        """
        return self.county_coordinates.get(county_fips, {})
    
    def get_all_coordinates(self) -> pd.DataFrame:
        """
        Get all county coordinates as a DataFrame
        
        Returns:
            DataFrame with county FIPS, names, and coordinates
        """
        data = []
        for fips, coords in self.county_coordinates.items():
            data.append({
                'county_fips': int(fips),  # Convert to integer to match main datasets
                'county_name': coords['county_name'],
                'county_seat': coords['county_seat'],
                'latitude': coords['latitude'],
                'longitude': coords['longitude'],
                'centroid_lat': coords['centroid_lat'],
                'centroid_lng': coords['centroid_lng']
            })
        
        return pd.DataFrame(data)
    
    def export_geographic_data(self, output_dir: str = "Data_Collection_Output/advanced_analytics/tableau_export"):
        """
        Export geographic data for Tableau
        
        Args:
            output_dir: Directory to save the geographic data
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Get all coordinates
            geo_df = self.get_all_coordinates()
            
            # Export to CSV
            output_file = os.path.join(output_dir, "oregon_counties_geographic.csv")
            geo_df.to_csv(output_file, index=False)
            
            print(f"Geographic data exported to: {output_file}")
            print(f"Total counties: {len(geo_df)}")
            
            return output_file
            
        except Exception as e:
            print(f"Error exporting geographic data: {str(e)}")
            return None

if __name__ == "__main__":
    # Test the geographic data
    geo = OregonCountiesGeographic()
    
    # Export for Tableau
    output_file = geo.export_geographic_data()
    
    if output_file:
        print(f"\n✅ Geographic data successfully exported!")
        print(f"📁 File: {output_file}")
        print(f"🗺️  All 36 Oregon counties now have coordinates for Tableau mapping!")
    else:
        print("❌ Failed to export geographic data")
