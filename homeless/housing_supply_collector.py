#!/usr/bin/env python3
"""
Oregon County Housing Supply Data Collector - Professional Edition
================================================================

This script implements a professional-grade housing supply data collection system
with proper naming conventions, data quality assessment, and production-ready
error handling.

Key Improvements:
1. Clear naming: housing supply metrics (not "housed" data)
2. Comprehensive data quality assessment
3. Production-ready error handling and monitoring
4. Proper data lineage tracking
5. Enhanced logging and audit trails
6. Building permits and construction data integration
"""

import requests
import pandas as pd
import time
import logging
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple, Any
import json
import asyncio
import aiohttp
from dataclasses import dataclass
from professional_data_architecture import (
    OregonHousingDataModel, 
    DataQualityFramework, 
    DataSource, 
    DataQualityMetrics
)

@dataclass
class CollectionMetrics:
    """Metrics for data collection performance and quality"""
    total_records: int
    counties_covered: int
    years_covered: int
    data_quality_score: str
    collection_time_seconds: float
    api_calls_made: int
    api_errors: int
    retry_attempts: int

class OregonHousingSupplyCollector:
    """
    Professional housing supply data collector for Oregon counties
    
    This class implements production-ready data collection with:
    - Clear, accurate naming conventions
    - Comprehensive data quality assessment
    - Robust error handling and retry logic
    - Performance monitoring and metrics
    - Data lineage tracking
    - Building permits and construction data
    """
    
    def __init__(self):
        """Initialize the professional housing supply data collector"""
        # Data architecture components
        self.data_model = OregonHousingDataModel()
        self.quality_framework = DataQualityFramework()
        
        # API configuration
        self.base_url = "https://api.census.gov/data"
        self.oregon_fips = "41"  # Oregon state FIPS code
        self.counties = self.data_model.counties
        
        # Directory structure
        self.output_dir = "Data_Collection_Output"
        self.historic_dir = os.path.join(self.output_dir, "historic_data")
        self.log_dir = os.path.join(self.historic_dir, "collection_logs")
        self.metrics_dir = os.path.join(self.historic_dir, "collection_metrics")
        
        # Performance tracking
        self.collection_start_time = None
        self.api_calls_made = 0
        self.api_errors = 0
        self.retry_attempts = 0
        
        # Setup logging and directories
        self.setup_logging()
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for data storage"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.historic_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
    def setup_logging(self):
        """Configure comprehensive logging for production use"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"professional_housing_supply_collection_{timestamp}.log")
        
        # Configure logging with different levels for file vs console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, level=logging.DEBUG),
                logging.StreamHandler(level=logging.INFO)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting Professional Oregon Housing Supply Data Collection - {timestamp}")
        
        # Log system configuration
        self.logger.info(f"Target counties: {len(self.counties)}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Log directory: {self.log_dir}")
        
    def make_api_call_sync(self, url: str, params: Dict, retries: int = 3) -> Optional[Dict]:
        """
        Make synchronous API call with comprehensive error handling
        
        Args:
            url: API endpoint URL
            params: Query parameters
            retries: Number of retry attempts
            
        Returns:
            JSON response data if successful, None if all retries fail
        """
        for attempt in range(retries):
            try:
                start_time = time.time()
                
                # Log API call attempt
                self.logger.debug(f"API Call Attempt {attempt + 1}/{retries}: {url}")
                self.logger.debug(f"Parameters: {params}")
                
                # Make HTTP request
                response = requests.get(url, params=params, timeout=30)
                response_time = time.time() - start_time
                
                # Track API call metrics
                self.api_calls_made += 1
                
                # Log response details
                self.logger.debug(f"Response Status: {response.status_code}")
                self.logger.debug(f"Response Time: {response_time:.2f} seconds")
                
                if response.status_code == 200:
                    data = response.json()
                    self.logger.debug(f"Raw API Response: {json.dumps(data, indent=2)}")
                    return data
                else:
                    self.logger.warning(f"API Error {response.status_code}: {response.text}")
                    self.api_errors += 1
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request Error (Attempt {attempt + 1}): {str(e)}")
                self.api_errors += 1
            except Exception as e:
                self.logger.error(f"Unexpected Error (Attempt {attempt + 1}): {str(e)}")
                self.api_errors += 1
            
            # Retry logic with exponential backoff
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                self.logger.info(f"Waiting {wait_time} seconds before retry...")
                self.retry_attempts += 1
                time.sleep(wait_time)
                
        return None
    
    def get_housing_supply_data(self, year: int) -> List[Dict]:
        """
        Collect comprehensive housing supply data from ACS for a specific year
        
        Args:
            year: ACS year (2010-2023)
            
        Returns:
            List of dictionaries containing housing supply data for each county
        """
        self.logger.info(f"Collecting housing supply data for {year}")
        
        # Comprehensive housing supply variables
        variables = [
            "B25001_001E",  # Total housing units
            "B25003_001E",  # Total occupied housing units
            "B25003_002E",  # Owner-occupied housing units
            "B25003_003E",  # Renter-occupied housing units
            "B25004_001E",  # Total vacant housing units
            "B25004_002E",  # Vacant for rent
            "B25004_003E",  # Vacant for sale only
            "B25004_004E",  # Vacant for seasonal/recreational/occasional use
            "B25004_005E",  # Vacant for migrant workers
            "B25004_006E",  # Other vacant
            "B25064_001E",  # Median gross rent
            "B25077_001E",  # Median home value
            "B25035_001E",  # Year structure built (median)
            "B25034_001E",  # Year structure built (total)
            "B25034_002E",  # Built 2014 or later
            "B25034_003E",  # Built 2010-2013
            "B25034_004E",  # Built 2000-2009
            "B25034_005E",  # Built 1990-1999
            "B25034_006E",  # Built 1980-1989
            "B25034_007E",  # Built 1970-1979
            "B25034_008E",  # Built 1960-1969
            "B25034_009E",  # Built 1950-1959
            "B25034_010E",  # Built 1940-1949
            "B25034_011E"   # Built 1939 or earlier
        ]
        
        url = f"{self.base_url}/{year}/acs/acs5"
        params = {
            "get": ",".join(variables),
            "for": "county:*",
            "in": f"state:{self.oregon_fips}",
            "key": ""
        }
        
        # Make API call
        data = self.make_api_call_sync(url, params)
        
        if data:
            self.logger.info(f"Successfully collected {year} housing supply data for {len(data)-1} counties")
            return self._process_housing_supply_data(data, year)
        else:
            self.logger.error(f"Failed to collect {year} housing supply data")
            return []
    
    def _process_housing_supply_data(self, data: List, year: int) -> List[Dict]:
        """
        Process raw ACS housing supply data into structured format
        
        Args:
            data: Raw API response from ACS
            year: ACS year being processed
            
        Returns:
            List of dictionaries with structured housing supply data for each county
        """
        processed_data = []
        collection_date = datetime.now()
        
        for row in data[1:]:  # Skip header row
            try:
                # Extract all housing supply variables
                total_housing_units = int(row[0]) if row[0] else None
                total_occupied_units = int(row[1]) if row[1] else None
                owner_occupied_units = int(row[2]) if row[2] else None
                renter_occupied_units = int(row[3]) if row[3] else None
                total_vacant_units = int(row[4]) if row[4] else None
                vacant_for_rent = int(row[5]) if row[5] else None
                vacant_for_sale = int(row[6]) if row[6] else None
                vacant_seasonal = int(row[7]) if row[7] else None
                vacant_migrant = int(row[8]) if row[8] else None
                vacant_other = int(row[9]) if row[9] else None
                median_gross_rent = int(row[10]) if row[10] else None
                median_home_value = int(row[11]) if row[11] else None
                median_year_built = int(row[12]) if row[12] else None
                total_structures = int(row[13]) if row[13] else None
                built_2014_later = int(row[14]) if row[14] else None
                built_2010_2013 = int(row[15]) if row[15] else None
                built_2000_2009 = int(row[16]) if row[16] else None
                built_1990_1999 = int(row[17]) if row[17] else None
                built_1980_1989 = int(row[18]) if row[18] else None
                built_1970_1979 = int(row[19]) if row[19] else None
                built_1960_1969 = int(row[20]) if row[20] else None
                built_1950_1959 = int(row[21]) if row[21] else None
                built_1940_1949 = int(row[22]) if row[22] else None
                built_1939_earlier = int(row[23]) if row[23] else None
                
                # Extract county information
                county_fips = row[25]  # Position after all variables
                county_name = self.counties.get(county_fips, f"Unknown County {county_fips}")
                
                # Calculate derived metrics
                vacancy_rate = (total_vacant_units / total_housing_units * 100) if total_vacant_units and total_housing_units else None
                homeownership_rate = (owner_occupied_units / total_occupied_units * 100) if owner_occupied_units and total_occupied_units else None
                rental_rate = (renter_occupied_units / total_occupied_units * 100) if renter_occupied_units and total_occupied_units else None
                
                # Calculate new construction metrics
                new_construction_units = built_2014_later + built_2010_2013 if built_2014_later and built_2010_2013 else None
                recent_construction_units = built_2000_2009 + built_2010_2013 + built_2014_later if all([built_2000_2009, built_2010_2013, built_2014_later]) else None
                
                # Assess data quality for this record
                quality_metrics = self.quality_framework.assess_dataset_quality(
                    pd.DataFrame([{
                        'total_housing_units': total_housing_units,
                        'total_occupied_units': total_occupied_units,
                        'total_vacant_units': total_vacant_units
                    }]),
                    DataSource.CENSUS_ACS.value,
                    collection_date,
                    year
                )
                
                # Create comprehensive housing supply record
                processed_data.append({
                    # Basic identifiers
                    "year": year,
                    "county_fips": county_fips,
                    "county_name": county_name,
                    
                    # Housing capacity
                    "total_housing_units": total_housing_units,
                    "total_occupied_units": total_occupied_units,
                    "total_vacant_units": total_vacant_units,
                    
                    # Occupancy breakdown
                    "owner_occupied_units": owner_occupied_units,
                    "renter_occupied_units": renter_occupied_units,
                    
                    # Vacancy breakdown
                    "vacant_for_rent": vacant_for_rent,
                    "vacant_for_sale": vacant_for_sale,
                    "vacant_seasonal": vacant_seasonal,
                    "vacant_migrant": vacant_migrant,
                    "vacant_other": vacant_other,
                    
                    # Affordability metrics
                    "median_gross_rent": median_gross_rent,
                    "median_home_value": median_home_value,
                    
                    # Construction age metrics
                    "median_year_built": median_year_built,
                    "total_structures": total_structures,
                    "built_2014_later": built_2014_later,
                    "built_2010_2013": built_2010_2013,
                    "built_2000_2009": built_2000_2009,
                    "built_1990_1999": built_1990_1999,
                    "built_1980_1989": built_1980_1989,
                    "built_1970_1979": built_1970_1979,
                    "built_1960_1969": built_1960_1969,
                    "built_1950_1959": built_1950_1959,
                    "built_1940_1949": built_1940_1949,
                    "built_1939_earlier": built_1939_earlier,
                    
                    # Derived metrics
                    "vacancy_rate_percent": vacancy_rate,
                    "homeownership_rate_percent": homeownership_rate,
                    "rental_rate_percent": rental_rate,
                    "new_construction_units": new_construction_units,
                    "recent_construction_units": recent_construction_units,
                    
                    # Metadata
                    "data_source": DataSource.CENSUS_ACS.value,
                    "data_quality_score": quality_metrics.overall_score.value,
                    "collection_date": collection_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Error processing {year} housing supply data row {row}: {str(e)}")
                
        return processed_data
    
    def get_building_permits_data(self, year: int) -> List[Dict]:
        """
        Collect building permits data for housing supply analysis
        
        Args:
            year: Year for building permits data
            
        Returns:
            List of dictionaries containing building permits data for each county
        """
        self.logger.info(f"Collecting building permits data for {year}")
        
        # Note: Building permits data would typically come from a different API
        # For now, we'll create placeholder data structure
        # In production, this would integrate with HUD or local building departments
        
        building_permits_data = []
        
        for county_fips, county_name in self.counties.items():
            # Placeholder data - in production this would be real API calls
            building_permits_data.append({
                "year": year,
                "county_fips": county_fips,
                "county_name": county_name,
                "building_permits_issued": None,  # Would be real data
                "new_construction_units": None,   # Would be real data
                "renovation_permits": None,       # Would be real data
                "data_source": DataSource.BUILDING_PERMITS.value,
                "data_quality_score": "unknown",  # Unknown quality for placeholder
                "collection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        self.logger.info(f"Created building permits placeholder data for {len(building_permits_data)} counties")
        return building_permits_data
    
    async def collect_all_data_async(self) -> pd.DataFrame:
        """
        Collect all available housing supply data asynchronously
        
        Returns:
            Pandas DataFrame with all housing supply data organized by county and year
        """
        all_data = []
        
        # Collect ACS housing supply data for 2010-2023
        acs_years = list(range(2010, 2024))  # 2010-2023
        
        for year in acs_years:
            self.logger.info(f"Processing housing supply year: {year}")
            
            # Get housing supply data for this specific year
            year_data = self.get_housing_supply_data(year)
            all_data.extend(year_data)
            
            # Get building permits data (placeholder for now)
            permits_data = self.get_building_permits_data(year)
            # Note: In production, this would be integrated with the main data
            
            await asyncio.sleep(1)  # Rate limiting
        
        # Convert to DataFrame and organize
        df = pd.DataFrame(all_data)
        
        if not df.empty:
            # Sort by county and year
            df = df.sort_values(['county_fips', 'year']).reset_index(drop=True)
            
            # Validate data quality
            self._validate_collected_data(df)
        
        return df
    
    def _validate_collected_data(self, df: pd.DataFrame):
        """
        Validate the collected data for quality and completeness
        
        Args:
            df: DataFrame to validate
        """
        if df.empty:
            self.logger.warning("No data collected - validation skipped")
            return
        
        # Check for missing counties
        expected_counties = 36
        actual_counties = df['county_fips'].nunique()
        if actual_counties < expected_counties:
            missing_counties = expected_counties - actual_counties
            self.logger.warning(f"Missing data for {missing_counties} counties")
        
        # Check for missing years
        expected_years = 14  # 2010-2023
        actual_years = df['year'].nunique()
        if actual_years < expected_years:
            missing_years = expected_years - actual_years
            self.logger.warning(f"Missing data for {missing_years} years")
        
        # Check for logical consistency
        if 'total_housing_units' in df.columns and 'total_occupied_units' in df.columns:
            if 'total_vacant_units' in df.columns:
                # Check: total = occupied + vacant
                logical_check = df['total_housing_units'] >= (df['total_occupied_units'] + df['total_vacant_units'])
                inconsistent_records = (~logical_check).sum()
                if inconsistent_records > 0:
                    self.logger.warning(f"Found {inconsistent_records} records with inconsistent housing unit counts")
        
        # Check for data quality issues
        quality_issues = df[df['data_quality_score'] == 'poor']
        if not quality_issues.empty:
            self.logger.warning(f"Found {len(quality_issues)} records with poor data quality")
    
    def save_data(self, df: pd.DataFrame) -> str:
        """
        Save collected data to CSV files with proper organization
        
        Args:
            df: DataFrame containing collected housing supply data
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save timestamped version (preserved for history)
        filename = f"oregon_county_housing_supply_2010_2023_acs_{timestamp}.csv"
        filepath = os.path.join(self.historic_dir, filename)
        
        df.to_csv(filepath, index=False)
        self.logger.info(f"Timestamped data saved to: {filepath}")
        
        # Save standard version (overwritten each time for easy access)
        standard_filename = "oregon_county_housing_supply_2010_2023_acs.csv"
        standard_filepath = os.path.join(self.output_dir, standard_filename)
        
        df.to_csv(standard_filepath, index=False)
        self.logger.info(f"Standard data saved to: {standard_filepath}")
        
        return filepath
    
    def save_collection_metrics(self, metrics: CollectionMetrics):
        """
        Save collection performance and quality metrics
        
        Args:
            metrics: CollectionMetrics object containing performance data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(self.metrics_dir, f"housing_supply_collection_metrics_{timestamp}.json")
        
        metrics_data = {
            "collection_timestamp": timestamp,
            "total_records": metrics.total_records,
            "counties_covered": metrics.counties_covered,
            "years_covered": metrics.years_covered,
            "data_quality_score": metrics.data_quality_score,
            "collection_time_seconds": metrics.collection_time_seconds,
            "api_calls_made": metrics.api_calls_made,
            "api_errors": metrics.api_errors,
            "retry_attempts": metrics.retry_attempts,
            "performance_metrics": {
                "records_per_second": metrics.total_records / metrics.collection_time_seconds if metrics.collection_time_seconds > 0 else 0,
                "api_success_rate": (metrics.api_calls_made - metrics.api_errors) / metrics.api_calls_made if metrics.api_calls_made > 0 else 0,
                "average_api_calls_per_record": metrics.api_calls_made / metrics.total_records if metrics.total_records > 0 else 0
            }
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        self.logger.info(f"Collection metrics saved to: {metrics_file}")
    
    async def run_collection(self) -> Tuple[str, CollectionMetrics]:
        """
        Main method to run the complete data collection process
        
        Returns:
            Tuple of (filepath, collection_metrics)
        """
        try:
            self.collection_start_time = time.time()
            
            self.logger.info("Starting Professional Oregon County Housing Supply Data Collection")
            self.logger.info(f"Target counties: {len(self.counties)}")
            self.logger.info("Collection period: 2010-2023")
            self.logger.info("Data sources: ACS Estimates + Building Permits (placeholder)")
            
            # Collect data
            df = await self.collect_all_data_async()
            
            # Calculate collection metrics
            collection_time = time.time() - self.collection_start_time
            
            # Assess overall data quality
            overall_quality = self.quality_framework.assess_dataset_quality(
                df,
                DataSource.CENSUS_ACS.value,
                datetime.now(),
                2023
            )
            
            # Create collection metrics
            metrics = CollectionMetrics(
                total_records=len(df),
                counties_covered=df['county_fips'].nunique() if not df.empty else 0,
                years_covered=df['year'].nunique() if not df.empty else 0,
                data_quality_score=overall_quality.overall_score.value,
                collection_time_seconds=collection_time,
                api_calls_made=self.api_calls_made,
                api_errors=self.api_errors,
                retry_attempts=self.retry_attempts
            )
            
            # Log summary statistics
            self.logger.info(f"Collection complete. Total records: {metrics.total_records}")
            self.logger.info(f"Years covered: {metrics.years_covered}")
            self.logger.info(f"Counties covered: {metrics.counties_covered}")
            self.logger.info(f"Data quality score: {metrics.data_quality_score}")
            self.logger.info(f"Collection time: {metrics.collection_time_seconds:.2f} seconds")
            self.logger.info(f"API calls made: {metrics.api_calls_made}")
            self.logger.info(f"API errors: {metrics.api_errors}")
            
            # Save data and metrics
            filepath = self.save_data(df)
            self.save_collection_metrics(metrics)
            
            self.logger.info("Professional housing supply data collection completed successfully!")
            return filepath, metrics
            
        except Exception as e:
            self.logger.error(f"Professional housing supply data collection failed: {str(e)}")
            raise

async def main():
    """Main execution function"""
    print("🏗️ Professional Oregon Housing Supply Data Collection")
    print("=" * 58)
    
    collector = OregonHousingSupplyCollector()
    
    try:
        filepath, metrics = await collector.run_collection()
        
        print(f"\n✅ Professional housing supply data collection completed successfully!")
        print(f"📁 Output file: {filepath}")
        print(f"📊 Collection metrics:")
        print(f"   - Total records: {metrics.total_records:,}")
        print(f"   - Counties covered: {metrics.counties_covered}")
        print(f"   - Years covered: {metrics.years_covered}")
        print(f"   - Data quality: {metrics.data_quality_score.title()}")
        print(f"   - Collection time: {metrics.collection_time_seconds:.2f} seconds")
        print(f"   - API success rate: {((metrics.api_calls_made - metrics.api_errors) / metrics.api_calls_made * 100):.1f}%")
        print(f"📋 Check the logs in: {collector.log_dir}")
        print(f"📈 Check the metrics in: {collector.metrics_dir}")
        
    except Exception as e:
        print(f"❌ Professional housing supply data collection failed: {str(e)}")
        print(f"📋 Check the logs in: {collector.log_dir}")

if __name__ == "__main__":
    asyncio.run(main())
