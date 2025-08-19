#!/usr/bin/env python3
"""
Oregon County Income Data Collector
==================================

This script implements a production-ready income data collection system
with clear naming conventions, data quality assessment, and comprehensive
error handling.

Key Features:
1. Clear naming: income and affordability metrics
2. Comprehensive data quality assessment
3. Production-ready error handling and monitoring
4. Proper data lineage tracking
5. Enhanced logging and audit trails
6. Income distribution and poverty rate analysis
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
from data_architecture import (
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

class OregonIncomeDataCollector:
    """
    Production-ready income data collector for Oregon counties
    
    This class implements comprehensive income data collection with:
    - Clear, accurate naming conventions
    - Comprehensive data quality assessment
    - Robust error handling and retry logic
    - Performance monitoring and metrics
    - Data lineage tracking
    - Income distribution and affordability analysis
    """
    
    def __init__(self):
        """Initialize the income data collector"""
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
        log_file = os.path.join(self.log_dir, f"income_collection_{timestamp}.log")
        
        # Create handlers with proper level configuration
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Configure logging with different levels for file vs console
        logging.basicConfig(
            level=logging.DEBUG,  # Set root level to lowest to capture all
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                file_handler, console_handler
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting Oregon Income Data Collection - {timestamp}")
        
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
    
    def get_income_data(self, year: int) -> List[Dict]:
        """
        Collect comprehensive income data from ACS for a specific year
        
        Args:
            year: ACS year (2009-2023)
            
        Returns:
            List of dictionaries containing income data for each county
        """
        self.logger.info(f"Collecting income data for {year}")
        
        # Core income variables (available in all years)
        core_variables = [
            "B19013_001E",  # Median household income
            "B19001_001E",  # Total households
            "B19001_002E",  # Less than $10,000
            "B19001_003E",  # $10,000 to $14,999
            "B19001_004E",  # $15,000 to $19,999
            "B19001_005E",  # $20,000 to $24,999
            "B19001_006E",  # $25,000 to $29,999
            "B19001_007E",  # $30,000 to $34,999
            "B19001_008E",  # $35,000 to $39,999
            "B19001_009E",  # $40,000 to $44,999
            "B19001_010E",  # $45,000 to $49,999
            "B19001_011E",  # $50,000 to $59,999
            "B19001_012E",  # $60,000 to $74,999
            "B19001_013E",  # $75,000 to $99,999
            "B19001_014E",  # $100,000 to $124,999
            "B19001_015E",  # $125,000 to $149,999
            "B19001_016E",  # $150,000 to $199,999
            "B19001_017E",  # $200,000 or more
        ]
        
        # Poverty and affordability variables (availability varies by year)
        if year >= 2015:
            # Full set available from 2015 onwards
            poverty_variables = [
                "B17001_001E",  # Total population for poverty determination
                "B17001_002E",  # Income below poverty level
                "B17001_003E",  # Income below poverty level - Under 18 years
                "B17001_004E",  # Income below poverty level - 18 to 64 years
                "B17001_005E",  # Income below poverty level - 65 years and over
                "B17001_006E",  # Income at or above poverty level
                "B17001_007E",  # Income at or above poverty level - Under 18 years
                "B17001_008E",  # Income at or above poverty level - 18 to 64 years
                "B17001_009E",  # Income at or above poverty level - 65 years and over
            ]
        elif year >= 2010:
            # Limited set for 2010-2014
            poverty_variables = [
                "B17001_001E",  # Total population for poverty determination
                "B17001_002E",  # Income below poverty level
                "B17001_003E",  # Income below poverty level - Under 18 years
                "B17001_004E",  # Income below poverty level - 18 to 64 years
                "B17001_005E",  # Income below poverty level - 65 years and over
                "B17001_006E",  # Income at or above poverty level
                "B17001_007E",  # Income at or above poverty level - Under 18 years
                "B17001_008E",  # Income at or above poverty level - 18 to 64 years
                "B17001_009E",  # Income at or above poverty level - 65 years and over
            ]
        else:
            # Minimal set for 2009
            poverty_variables = [
                "B17001_001E",  # Total population for poverty determination
                "B17001_002E",  # Income below poverty level
                "B17001_003E",  # Income below poverty level - Under 18 years
                "B17001_004E",  # Income below poverty level - 18 to 64 years
                "B17001_005E",  # Income below poverty level - 65 years and over
                "B17001_006E",  # Income at or above poverty level
                "B17001_007E",  # Income at or above poverty level - Under 18 years
                "B17001_008E",  # Income at or above poverty level - 18 to 64 years
                "B17001_009E",  # Income at or above poverty level - 65 years and over
            ]
        
        # Housing cost burden variables (availability varies by year)
        if year >= 2015:
            # Full set available from 2015 onwards
            cost_burden_variables = [
                "B25070_001E",  # Total renter-occupied housing units
                "B25070_002E",  # Less than 10.0 percent
                "B25070_003E",  # 10.0 to 14.9 percent
                "B25070_004E",  # 15.0 to 19.9 percent
                "B25070_005E",  # 20.0 to 24.9 percent
                "B25070_006E",  # 25.0 to 29.9 percent
                "B25070_007E",  # 30.0 to 34.9 percent
                "B25070_008E",  # 35.0 to 39.9 percent
                "B25070_009E",  # 40.0 to 49.9 percent
                "B25070_010E",  # 50.0 percent or more
                "B25070_011E",  # Not computed
            ]
        else:
            # Limited set for 2009-2014
            cost_burden_variables = [
                "B25070_001E",  # Total renter-occupied housing units
                "B25070_002E",  # Less than 10.0 percent
                "B25070_003E",  # 10.0 to 14.9 percent
                "B25070_004E",  # 15.0 to 19.9 percent
                "B25070_005E",  # 20.0 to 24.9 percent
                "B25070_006E",  # 25.0 to 29.9 percent
                "B25070_007E",  # 30.0 to 34.9 percent
                "B25070_008E",  # 35.0 to 39.9 percent
                "B25070_009E",  # 40.0 to 49.9 percent
                "B25070_010E",  # 50.0 percent or more
                "B25070_011E",  # Not computed
            ]
        
        # Combine all variables
        variables = core_variables + poverty_variables + cost_burden_variables
        
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
            self.logger.info(f"Successfully collected {year} income data for {len(data)-1} counties")
            return self._process_income_data(data, year)
        else:
            self.logger.error(f"Failed to collect {year} income data")
            return []
    
    def _process_income_data(self, data: List, year: int) -> List[Dict]:
        """
        Process raw ACS income data into structured format
        
        Args:
            data: Raw API response from ACS
            year: ACS year being processed
            
        Returns:
            List of dictionaries with structured income data for each county
        """
        processed_data = []
        collection_date = datetime.now()
        
        for row in data[1:]:  # Skip header row
            try:
                # Extract core income variables (fixed positions)
                median_household_income = int(row[0]) if row[0] else None
                total_households = int(row[1]) if row[1] else None
                
                # Extract income distribution variables
                income_less_10k = int(row[2]) if row[2] else None
                income_10k_15k = int(row[3]) if row[3] else None
                income_15k_20k = int(row[4]) if row[4] else None
                income_20k_25k = int(row[5]) if row[5] else None
                income_25k_30k = int(row[6]) if row[6] else None
                income_30k_35k = int(row[7]) if row[7] else None
                income_35k_40k = int(row[8]) if row[8] else None
                income_40k_45k = int(row[9]) if row[9] else None
                income_45k_50k = int(row[10]) if row[10] else None
                income_50k_60k = int(row[11]) if row[11] else None
                income_60k_75k = int(row[12]) if row[12] else None
                income_75k_100k = int(row[13]) if row[13] else None
                income_100k_125k = int(row[14]) if row[14] else None
                income_125k_150k = int(row[15]) if row[15] else None
                income_150k_200k = int(row[16]) if row[16] else None
                income_200k_plus = int(row[17]) if row[17] else None
                
                # Extract poverty variables (positions vary by year)
                base_index = 18
                total_poverty_population = int(row[base_index]) if row[base_index] else None
                poverty_under_18 = int(row[base_index + 1]) if row[base_index + 1] else None
                poverty_18_64 = int(row[base_index + 2]) if row[base_index + 2] else None
                poverty_65_plus = int(row[base_index + 3]) if row[base_index + 3] else None
                above_poverty_total = int(row[base_index + 4]) if row[base_index + 4] else None
                above_poverty_under_18 = int(row[base_index + 5]) if row[base_index + 5] else None
                above_poverty_18_64 = int(row[base_index + 6]) if row[base_index + 6] else None
                above_poverty_65_plus = int(row[base_index + 7]) if row[base_index + 7] else None
                
                # Extract cost burden variables (positions vary by year)
                cost_burden_base = base_index + 8
                total_renter_units = int(row[cost_burden_base]) if row[cost_burden_base] else None
                cost_burden_less_10 = int(row[cost_burden_base + 1]) if row[cost_burden_base + 1] else None
                cost_burden_10_15 = int(row[cost_burden_base + 2]) if row[cost_burden_base + 2] else None
                cost_burden_15_20 = int(row[cost_burden_base + 3]) if row[cost_burden_base + 3] else None
                cost_burden_20_25 = int(row[cost_burden_base + 4]) if row[cost_burden_base + 4] else None
                cost_burden_25_30 = int(row[cost_burden_base + 5]) if row[cost_burden_base + 5] else None
                cost_burden_30_35 = int(row[cost_burden_base + 6]) if row[cost_burden_base + 6] else None
                cost_burden_35_40 = int(row[cost_burden_base + 7]) if row[cost_burden_base + 7] else None
                cost_burden_40_50 = int(row[cost_burden_base + 8]) if row[cost_burden_base + 8] else None
                cost_burden_50_plus = int(row[cost_burden_base + 9]) if row[cost_burden_base + 9] else None
                cost_burden_not_computed = int(row[cost_burden_base + 10]) if row[cost_burden_base + 10] else None
                
                # Extract county information
                county_fips = row[-1]  # Last position is county FIPS
                county_name = self.counties.get(county_fips, f"Unknown County {county_fips}")
                
                # Calculate derived metrics
                poverty_rate = (total_poverty_population / (total_poverty_population + above_poverty_total) * 100) if total_poverty_population and above_poverty_total else None
                
                # Calculate income distribution percentages
                low_income_households = (income_less_10k + income_10k_15k + income_15k_20k + income_20k_25k) if all([income_less_10k, income_10k_15k, income_15k_20k, income_20k_25k]) else None
                middle_income_households = (income_25k_30k + income_30k_35k + income_35k_40k + income_40k_45k + income_45k_50k + income_50k_60k) if all([income_25k_30k, income_30k_35k, income_35k_40k, income_40k_45k, income_45k_50k, income_50k_60k]) else None
                high_income_households = (income_60k_75k + income_75k_100k + income_100k_125k + income_125k_150k + income_150k_200k + income_200k_plus) if all([income_60k_75k, income_75k_100k, income_100k_125k, income_125k_150k, income_150k_200k, income_200k_plus]) else None
                
                # Calculate cost burden metrics
                high_cost_burden_households = (cost_burden_30_35 + cost_burden_35_40 + cost_burden_40_50 + cost_burden_50_plus) if all([cost_burden_30_35, cost_burden_35_40, cost_burden_40_50, cost_burden_50_plus]) else None
                cost_burden_rate = (high_cost_burden_households / total_renter_units * 100) if high_cost_burden_households and total_renter_units else None
                
                # Assess data quality for this record
                quality_metrics = self.quality_framework.assess_dataset_quality(
                    pd.DataFrame([{
                        'median_household_income': median_household_income,
                        'total_households': total_households,
                        'total_poverty_population': total_poverty_population,
                        'county_fips': county_fips,
                        'year': year
                    }]),
                    DataSource.CENSUS_ACS.value,
                    collection_date,
                    year
                )
                
                # Create comprehensive income record
                processed_data.append({
                    # Basic identifiers
                    "year": year,
                    "county_fips": county_fips,
                    "county_name": county_name,
                    
                    # Core income metrics
                    "median_household_income": median_household_income,
                    "total_households": total_households,
                    
                    # Income distribution (raw counts)
                    "income_less_10k": income_less_10k,
                    "income_10k_15k": income_10k_15k,
                    "income_15k_20k": income_15k_20k,
                    "income_20k_25k": income_20k_25k,
                    "income_25k_30k": income_25k_30k,
                    "income_30k_35k": income_30k_35k,
                    "income_35k_40k": income_35k_40k,
                    "income_40k_45k": income_40k_45k,
                    "income_45k_50k": income_45k_50k,
                    "income_50k_60k": income_50k_60k,
                    "income_60k_75k": income_60k_75k,
                    "income_75k_100k": income_75k_100k,
                    "income_100k_125k": income_100k_125k,
                    "income_125k_150k": income_125k_150k,
                    "income_150k_200k": income_150k_200k,
                    "income_200k_plus": income_200k_plus,
                    
                    # Income distribution (aggregated)
                    "low_income_households": low_income_households,
                    "middle_income_households": middle_income_households,
                    "high_income_households": high_income_households,
                    
                    # Poverty metrics
                    "total_poverty_population": total_poverty_population,
                    "poverty_under_18": poverty_under_18,
                    "poverty_18_64": poverty_18_64,
                    "poverty_65_plus": poverty_65_plus,
                    "above_poverty_total": above_poverty_total,
                    "above_poverty_under_18": above_poverty_under_18,
                    "above_poverty_18_64": above_poverty_18_64,
                    "above_poverty_65_plus": above_poverty_65_plus,
                    "poverty_rate_percent": poverty_rate,
                    
                    # Housing cost burden metrics
                    "total_renter_units": total_renter_units,
                    "cost_burden_less_10": cost_burden_less_10,
                    "cost_burden_10_15": cost_burden_10_15,
                    "cost_burden_15_20": cost_burden_15_20,
                    "cost_burden_20_25": cost_burden_20_25,
                    "cost_burden_25_30": cost_burden_25_30,
                    "cost_burden_30_35": cost_burden_30_35,
                    "cost_burden_35_40": cost_burden_35_40,
                    "cost_burden_40_50": cost_burden_40_50,
                    "cost_burden_50_plus": cost_burden_50_plus,
                    "cost_burden_not_computed": cost_burden_not_computed,
                    "high_cost_burden_households": high_cost_burden_households,
                    "cost_burden_rate_percent": cost_burden_rate,
                    
                    # Metadata
                    "data_source": DataSource.CENSUS_ACS.value,
                    "data_quality_score": quality_metrics.overall_score.value,
                    "collection_date": collection_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Error processing {year} income data row {row}: {str(e)}")
                
        return processed_data
    
    async def collect_all_data_async(self) -> pd.DataFrame:
        """
        Collect all available income data asynchronously
        
        Returns:
            Pandas DataFrame with all income data organized by county and year
        """
        all_data = []
        
        # Collect ACS income data for 2009-2023
        acs_years = list(range(2009, 2024))  # 2009-2023
        
        for year in acs_years:
            self.logger.info(f"Processing income year: {year}")
            
            # Get income data for this specific year
            year_data = self.get_income_data(year)
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
        expected_years = 15  # 2009-2023
        actual_years = df['year'].nunique()
        if actual_years < expected_years:
            missing_years = expected_years - actual_years
            self.logger.warning(f"Missing data for {missing_years} years")
        
        # Check for logical consistency
        if 'total_households' in df.columns and 'low_income_households' in df.columns:
            if 'middle_income_households' in df.columns and 'high_income_households' in df.columns:
                # Check: total households >= sum of income brackets
                logical_check = df['total_households'] >= (df['low_income_households'] + df['middle_income_households'] + df['high_income_households'])
                inconsistent_records = (~logical_check).sum()
                if inconsistent_records > 0:
                    self.logger.warning(f"Found {inconsistent_records} records with inconsistent household counts")
        
        # Check for data quality issues
        quality_issues = df[df['data_quality_score'] == 'poor']
        if not quality_issues.empty:
            self.logger.warning(f"Found {len(quality_issues)} records with poor data quality")
    
    def save_data(self, df: pd.DataFrame) -> str:
        """
        Save collected data to CSV files with proper organization
        
        Args:
            df: DataFrame containing collected income data
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save timestamped version (preserved for history)
        filename = f"oregon_county_income_2009_2023_acs_{timestamp}.csv"
        filepath = os.path.join(self.historic_dir, filename)
        
        df.to_csv(filepath, index=False)
        self.logger.info(f"Timestamped data saved to: {filepath}")
        
        # Save standard version (overwritten each time for easy access)
        standard_filename = "oregon_county_income_2009_2023_acs.csv"
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
        metrics_file = os.path.join(self.metrics_dir, f"income_collection_metrics_{timestamp}.json")
        
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
            
            self.logger.info("Starting Oregon County Income Data Collection")
            self.logger.info(f"Target counties: {len(self.counties)}")
            self.logger.info("Collection period: 2009-2023")
            self.logger.info("Data sources: ACS Estimates")
            
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
            
            self.logger.info("Income data collection completed successfully!")
            return filepath, metrics
            
        except Exception as e:
            self.logger.error(f"Income data collection failed: {str(e)}")
            raise

async def main():
    """Main execution function"""
    print("💰 Oregon Income Data Collection")
    print("=" * 58)
    
    collector = OregonIncomeDataCollector()
    
    try:
        filepath, metrics = await collector.run_collection()
        
        print(f"\n✅ Income data collection completed successfully!")
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
        print(f"❌ Income data collection failed: {str(e)}")
        print(f"📋 Check the logs in: {collector.log_dir}")

if __name__ == "__main__":
    asyncio.run(main())
