#!/usr/bin/env python3
"""
Oregon County Population Data Collector - Professional Edition
============================================================

This script implements a professional-grade population data collection system
with proper naming conventions, data quality assessment, and production-ready
error handling.

Key Improvements:
1. Clear naming: total_population (not "homeless_total_population")
2. Comprehensive data quality assessment
3. Production-ready error handling and monitoring
4. Proper data lineage tracking
5. Enhanced logging and audit trails
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

class OregonPopulationDataCollector:
    """
    Professional population data collector for Oregon counties
    
    This class implements production-ready data collection with:
    - Clear, accurate naming conventions
    - Comprehensive data quality assessment
    - Robust error handling and retry logic
    - Performance monitoring and metrics
    - Data lineage tracking
    """
    
    def __init__(self):
        """Initialize the professional population data collector"""
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
        log_file = os.path.join(self.log_dir, f"professional_population_collection_{timestamp}.log")
        
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
        self.logger.info(f"Starting Professional Oregon Population Data Collection - {timestamp}")
        
        # Log system configuration
        self.logger.info(f"Target counties: {len(self.counties)}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Log directory: {self.log_dir}")
        
    async def make_api_call_async(self, session: aiohttp.ClientSession, url: str, 
                                params: Dict, retries: int = 3) -> Optional[Dict]:
        """
        Make asynchronous API call with comprehensive error handling
        
        Args:
            session: aiohttp session for async requests
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
                
                # Make async HTTP request
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    response_time = time.time() - start_time
                    
                    # Track API call metrics
                    self.api_calls_made += 1
                    
                    # Log response details
                    self.logger.debug(f"Response Status: {response.status}")
                    self.logger.debug(f"Response Time: {response_time:.2f} seconds")
                    
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug(f"Raw API Response: {json.dumps(data, indent=2)}")
                        return data
                    else:
                        error_text = await response.text()
                        self.logger.warning(f"API Error {response.status}: {error_text}")
                        self.api_errors += 1
                        
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout Error (Attempt {attempt + 1}): Request timed out")
                self.api_errors += 1
            except aiohttp.ClientError as e:
                self.logger.error(f"Client Error (Attempt {attempt + 1}): {str(e)}")
                self.api_errors += 1
            except Exception as e:
                self.logger.error(f"Unexpected Error (Attempt {attempt + 1}): {str(e)}")
                self.api_errors += 1
            
            # Retry logic with exponential backoff
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                self.logger.info(f"Waiting {wait_time} seconds before retry...")
                self.retry_attempts += 1
                await asyncio.sleep(wait_time)
                
        return None
    
    def make_api_call_sync(self, url: str, params: Dict, retries: int = 3) -> Optional[Dict]:
        """
        Make synchronous API call (fallback for non-async operations)
        
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
    
    def get_decennial_data(self, year: int) -> List[Dict]:
        """
        Collect population data from Decennial Census for a specific year
        
        Args:
            year: Census year (1990, 2000, 2010, or 2020)
            
        Returns:
            List of dictionaries containing population data for each county
        """
        self.logger.info(f"Collecting Decennial Census data for {year}")
        
        # Different years use different API structures and variable names
        if year == 2020:
            variables = "P1_001N"  # Total population variable for 2020
            dataset = "pl"         # New dataset name
        elif year in [1990, 2000, 2010]:
            variables = "P0010001"  # Total population variable for 1990-2010
            dataset = "sf1"        # Dataset name for 1990-2010
        else:
            self.logger.error(f"Unsupported decennial year: {year}")
            return []
        
        # Construct API URL
        url = f"{self.base_url}/{year}/dec/{dataset}"
        params = {
            "get": variables,
            "for": "county:*",
            "in": f"state:{self.oregon_fips}",
            "key": ""  # Will be added when API key is available
        }
        
        # Make API call
        data = self.make_api_call_sync(url, params)
        
        if data:
            self.logger.info(f"Successfully collected {year} decennial data for {len(data)-1} counties")
            return self._process_decennial_data(data, year)
        else:
            self.logger.error(f"Failed to collect {year} decennial data")
            return []
    
    def _process_decennial_data(self, data: List, year: int) -> List[Dict]:
        """
        Process raw decennial census data into structured format
        
        Args:
            data: Raw API response from Census Bureau
            year: Census year being processed
            
        Returns:
            List of dictionaries with structured county data
        """
        processed_data = []
        collection_date = datetime.now()
        
        for row in data[1:]:  # Skip header row
            try:
                # Extract population count (clear naming)
                total_population = int(row[0]) if row[0] else None
                county_fips = row[2]
                county_name = self.counties.get(county_fips, f"Unknown County {county_fips}")
                
                # Assess data quality for this record
                quality_metrics = self.quality_framework.assess_dataset_quality(
                    pd.DataFrame([{'total_population': total_population}]),
                    DataSource.CENSUS_DECENNIAL.value,
                    collection_date,
                    year
                )
                
                # Create structured record with proper naming
                processed_data.append({
                    "year": year,                                    # Census year
                    "county_fips": county_fips,                      # 3-digit county code
                    "county_name": county_name,                      # Human-readable name
                    "total_population": total_population,            # Total population (clear naming)
                    "data_source": DataSource.CENSUS_DECENNIAL.value, # Data source identifier
                    "margin_of_error": None,                         # Decennial has no margin of error
                    "data_quality_score": quality_metrics.overall_score.value, # Quality assessment
                    "collection_date": collection_date.strftime("%Y-%m-%d %H:%M:%S"), # When collected
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Last update
                })
                
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Error processing {year} data row {row}: {str(e)}")
                
        return processed_data
    
    def get_acs_data(self, year: int) -> List[Dict]:
        """
        Collect population data from ACS for a specific year
        
        Args:
            year: ACS year (2010-2023)
            
        Returns:
            List of dictionaries containing population data for each county
        """
        self.logger.info(f"Collecting ACS data for {year}")
        
        # ACS variables for total population
        variables = "B01003_001E,B01003_001M"  # Population estimate + margin of error
        
        url = f"{self.base_url}/{year}/acs/acs5"
        params = {
            "get": variables,
            "for": "county:*",
            "in": f"state:{self.oregon_fips}",
            "key": ""
        }
        
        # Make API call
        data = self.make_api_call_sync(url, params)
        
        if data:
            self.logger.info(f"Successfully collected {year} ACS data for {len(data)-1} counties")
            return self._process_acs_data(data, year)
        else:
            self.logger.error(f"Failed to collect {year} ACS data")
            return []
    
    def _process_acs_data(self, data: List, year: int) -> List[Dict]:
        """
        Process raw ACS data into structured format
        
        Args:
            data: Raw API response from ACS
            year: ACS year being processed
            
        Returns:
            List of dictionaries with structured county data
        """
        processed_data = []
        collection_date = datetime.now()
        
        for row in data[1:]:  # Skip header row
            try:
                # Extract population estimate and margin of error
                total_population = int(row[0]) if row[0] else None
                margin_of_error = int(row[1]) if row[1] and row[1] != "-555555555" else None
                county_fips = row[3]
                county_name = self.counties.get(county_fips, f"Unknown County {county_fips}")
                
                # Assess data quality for this record
                quality_metrics = self.quality_framework.assess_dataset_quality(
                    pd.DataFrame([{'total_population': total_population}]),
                    DataSource.CENSUS_ACS.value,
                    collection_date,
                    year
                )
                
                # Create structured record with proper naming
                processed_data.append({
                    "year": year,                                    # ACS year
                    "county_fips": county_fips,                      # 3-digit county code
                    "county_name": county_name,                      # Human-readable name
                    "total_population": total_population,            # Total population (clear naming)
                    "data_source": DataSource.CENSUS_ACS.value,      # Data source identifier
                    "margin_of_error": margin_of_error,              # Uncertainty in estimate
                    "data_quality_score": quality_metrics.overall_score.value, # Quality assessment
                    "collection_date": collection_date.strftime("%Y-%m-%d %H:%M:%S"), # When collected
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Last update
                })
                
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Error processing {year} ACS data row {row}: {str(e)}")
                
        return processed_data
    
    async def collect_all_data_async(self) -> pd.DataFrame:
        """
        Collect all available population data asynchronously
        
        Returns:
            Pandas DataFrame with all population data organized by county and year
        """
        all_data = []
        
        # Collect decennial census data (synchronous for now)
        decennial_years = [1990, 2000, 2010, 2020]
        for year in decennial_years:
            self.logger.info(f"Processing decennial year: {year}")
            year_data = self.get_decennial_data(year)
            all_data.extend(year_data)
            await asyncio.sleep(1)  # Rate limiting
        
        # Collect ACS data
        acs_years = list(range(2010, 2024))  # 2010-2023
        for year in acs_years:
            self.logger.info(f"Processing ACS year: {year}")
            year_data = self.get_acs_data(year)
            all_data.extend(year_data)
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
        expected_years = len(set([1990, 2000, 2010, 2020] + list(range(2010, 2024))))
        actual_years = df['year'].nunique()
        if actual_years < expected_years:
            missing_years = expected_years - actual_years
            self.logger.warning(f"Missing data for {missing_years} years")
        
        # Check for data quality issues
        quality_issues = df[df['data_quality_score'] == 'poor']
        if not quality_issues.empty:
            self.logger.warning(f"Found {len(quality_issues)} records with poor data quality")
    
    def save_data(self, df: pd.DataFrame) -> str:
        """
        Save collected data to CSV files with proper organization
        
        Args:
            df: DataFrame containing collected population data
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save timestamped version (preserved for history)
        filename = f"oregon_county_population_1990_2023_census_acs_{timestamp}.csv"
        filepath = os.path.join(self.historic_dir, filename)
        
        df.to_csv(filepath, index=False)
        self.logger.info(f"Timestamped data saved to: {filepath}")
        
        # Save standard version (overwritten each time for easy access)
        standard_filename = "oregon_county_population_1990_2023_census_acs.csv"
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
        metrics_file = os.path.join(self.metrics_dir, f"collection_metrics_{timestamp}.json")
        
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
            
            self.logger.info("Starting Professional Oregon County Population Data Collection")
            self.logger.info(f"Target counties: {len(self.counties)}")
            self.logger.info("Collection period: 1990-2023")
            self.logger.info("Data sources: Decennial Census + ACS Estimates")
            
            # Collect data
            df = await self.collect_all_data_async()
            
            # Calculate collection metrics
            collection_time = time.time() - self.collection_start_time
            
            # Assess overall data quality
            overall_quality = self.quality_framework.assess_dataset_quality(
                df,
                DataSource.CENSUS_ACS.value,  # Use ACS as baseline for overall assessment
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
            
            self.logger.info("Professional population data collection completed successfully!")
            return filepath, metrics
            
        except Exception as e:
            self.logger.error(f"Professional population data collection failed: {str(e)}")
            raise

async def main():
    """Main execution function"""
    print("🏗️ Professional Oregon Population Data Collection")
    print("=" * 55)
    
    collector = OregonPopulationDataCollector()
    
    try:
        filepath, metrics = await collector.run_collection()
        
        print(f"\n✅ Professional population data collection completed successfully!")
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
        print(f"❌ Professional population data collection failed: {str(e)}")
        print(f"📋 Check the logs in: {collector.log_dir}")

if __name__ == "__main__":
    asyncio.run(main())
