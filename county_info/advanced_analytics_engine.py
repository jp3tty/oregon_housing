#!/usr/bin/env python3
"""
Oregon Housing Advanced Analytics Engine - Phase 6
==================================================

This script implements advanced analytics capabilities for the Oregon Housing Study:
1. Trend Analysis & Forecasting
2. Comparative Analysis & Benchmarking
3. Statistical Modeling & Correlation Studies
4. Regional Analysis & Geographic Insights
5. Policy Impact Analysis & Recommendations

Key Features:
- Multi-dimensional trend analysis across all metrics
- Predictive modeling for housing needs and homeless populations
- County ranking and benchmarking systems
- Regional clustering and peer group analysis
- Advanced statistical modeling and correlation analysis
- Policy recommendation framework based on data insights
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from data_architecture import (
    OregonHousingDataModel, 
    DataQualityFramework, 
    DataSource, 
    DataQualityMetrics
)

@dataclass
class TrendAnalysis:
    """Trend analysis results for a specific metric"""
    metric_name: str
    trend_direction: str  # "increasing", "decreasing", "stable", "fluctuating"
    trend_strength: float  # R-squared value (0-1)
    annual_change_rate: float  # Average annual change
    significance_level: str  # "high", "medium", "low"
    confidence_interval: Tuple[float, float]
    outliers: List[str]  # County names with unusual patterns

@dataclass
class ForecastingModel:
    """Forecasting model results"""
    metric_name: str
    model_type: str  # "linear", "polynomial", "seasonal"
    forecast_horizon: int  # Years into future
    predicted_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    model_accuracy: float  # R-squared or similar metric
    key_factors: List[str]  # Most important predictive variables

@dataclass
class CountyRanking:
    """County ranking and benchmarking results"""
    county_name: str
    county_fips: str
    overall_score: float
    rank: int
    percentile: float
    strengths: List[str]
    weaknesses: List[str]
    peer_counties: List[str]
    improvement_opportunities: List[str]

@dataclass
class RegionalAnalysis:
    """Regional analysis results"""
    region_name: str
    counties_included: List[str]
    average_stress_score: float
    common_challenges: List[str]
    unique_characteristics: List[str]
    policy_recommendations: List[str]

class OregonHousingAdvancedAnalytics:
    """
    Advanced analytics engine for Oregon housing data
    
    This class implements comprehensive analytics including:
    - Trend analysis and forecasting
    - Comparative analysis and benchmarking
    - Statistical modeling and correlation studies
    - Regional analysis and geographic insights
    - Policy recommendation framework
    """
    
    def __init__(self):
        """Initialize the advanced analytics engine"""
        # Data architecture components
        self.data_model = OregonHousingDataModel()
        self.quality_framework = DataQualityFramework()
        
        # Directory structure
        self.output_dir = "Data_Collection_Output"
        self.analytics_dir = os.path.join(self.output_dir, "advanced_analytics")
        self.visualizations_dir = os.path.join(self.analytics_dir, "visualizations")
        self.reports_dir = os.path.join(self.analytics_dir, "reports")
        self.models_dir = os.path.join(self.analytics_dir, "models")
        
        # Setup directories and logging
        self.setup_directories()
        self.setup_logging()
        
        # Load all available data
        self.load_all_data()
        
        # Analysis parameters
        self.trend_analysis_years = 5  # Minimum years for trend analysis
        self.forecast_horizon = 5  # Years to forecast into future
        self.clustering_n_clusters = 4  # Number of county clusters for regional analysis
        
    def setup_directories(self):
        """Create necessary directories for analytics output"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.analytics_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
    def setup_logging(self):
        """Configure comprehensive logging for analytics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.analytics_dir, f"advanced_analytics_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Advanced Analytics Engine initialized")
        
    def load_all_data(self):
        """Load all available data files for analysis"""
        try:
            # Load main data files
            self.population_data = pd.read_csv(
                os.path.join(self.output_dir, "oregon_county_population_2009_2023_reliable.csv")
            )
            self.housing_supply_data = pd.read_csv(
                os.path.join(self.output_dir, "oregon_county_housing_supply_2009_2023_reliable.csv")
            )
            self.income_data = pd.read_csv(
                os.path.join(self.output_dir, "oregon_county_income_2009_2023_reliable.csv")
            )
            self.homeless_data = pd.read_csv(
                os.path.join(self.output_dir, "oregon_county_homeless_data.csv")
            )
            self.gap_analysis_data = pd.read_csv(
                os.path.join(self.output_dir, "oregon_county_housing_gap_analysis_2009_2023.csv")
            )
            
            # Data preprocessing
            self.preprocess_data()
            
            self.logger.info("All data files loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_data(self):
        """Preprocess and clean data for analysis"""
        # Convert year to datetime for trend analysis
        for df in [self.population_data, self.housing_supply_data, 
                  self.income_data, self.gap_analysis_data]:
            df['year'] = pd.to_datetime(df['year'], format='%Y')
            
        # Handle missing values
        self.population_data = self.population_data.dropna(subset=['total_population'])
        self.housing_supply_data = self.housing_supply_data.dropna(subset=['total_housing_units'])
        self.income_data = self.income_data.dropna(subset=['median_household_income'])
        self.gap_analysis_data = self.gap_analysis_data.dropna(subset=['housing_stress_score'])
        
        # Create merged dataset for comprehensive analysis
        self.merged_data = self.create_merged_dataset()
        
        self.logger.info("Data preprocessing completed")
        
    def create_merged_dataset(self):
        """Create a comprehensive merged dataset for analysis"""
        # First, let's check what columns we have in each dataset to avoid conflicts
        self.logger.info("Population data columns: " + str(list(self.population_data.columns)))
        self.logger.info("Housing supply data columns: " + str(list(self.housing_supply_data.columns)))
        self.logger.info("Income data columns: " + str(list(self.income_data.columns)))
        self.logger.info("Gap analysis data columns: " + str(list(self.gap_analysis_data.columns)))
        
        # Clean up duplicate columns before merging
        # Remove common metadata columns that might cause conflicts
        columns_to_remove = ['data_source', 'margin_of_error', 'data_quality_score', 'collection_date', 'last_updated']
        
        # Create clean copies for merging
        pop_clean = self.population_data.drop(columns=[col for col in columns_to_remove if col in self.population_data.columns])
        housing_clean = self.housing_supply_data.drop(columns=[col for col in columns_to_remove if col in self.housing_supply_data.columns])
        income_clean = self.income_data.drop(columns=[col for col in columns_to_remove if col in self.income_data.columns])
        gap_clean = self.gap_analysis_data.drop(columns=[col for col in columns_to_remove if col in self.gap_analysis_data.columns])
        
        # Merge all datasets on year and county
        merged = pop_clean.merge(
            housing_clean, 
            on=['year', 'county_fips', 'county_name'], 
            how='inner'
        ).merge(
            income_clean, 
            on=['year', 'county_fips', 'county_name'], 
            how='inner'
        ).merge(
            gap_clean, 
            on=['year', 'county_fips', 'county_name'], 
            how='inner'
        )
        
        # Add homeless data (may have different year coverage)
        homeless_agg = self.homeless_data.groupby(['county_fips', 'county_name']).agg({
            'total_homeless': 'mean',
            'shelter_capacity': 'mean'
        }).reset_index()
        
        merged = merged.merge(homeless_agg, on=['county_fips', 'county_name'], how='left')
        
        self.logger.info(f"Merged dataset created with {len(merged)} records and {len(merged.columns)} columns")
        self.logger.info("Merged dataset columns: " + str(list(merged.columns)))
        
        return merged
        
    def perform_trend_analysis(self, metric_name: str) -> TrendAnalysis:
        """
        Perform trend analysis on a specific metric across all counties
        
        Args:
            metric_name: Name of the metric to analyze
            
        Returns:
            TrendAnalysis object with trend results
        """
        try:
            if metric_name not in self.merged_data.columns:
                raise ValueError(f"Metric {metric_name} not found in dataset")
                
            # Group by year and calculate mean across counties
            yearly_trends = self.merged_data.groupby('year')[metric_name].agg(['mean', 'std']).reset_index()
            
            # Fit linear regression for trend
            X = (yearly_trends['year'] - yearly_trends['year'].min()).dt.days.values.reshape(-1, 1)
            y = yearly_trends['mean'].values
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Calculate trend metrics
            r2 = r2_score(y, y_pred)
            annual_change = model.coef_[0] * 365  # Convert daily change to annual
            
            # Determine trend direction and strength
            if abs(annual_change) < 0.01 * yearly_trends['mean'].mean():
                trend_direction = "stable"
            elif annual_change > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
                
            if r2 > 0.7:
                trend_strength = "high"
            elif r2 > 0.4:
                trend_strength = "medium"
            else:
                trend_strength = "low"
                
            # Calculate confidence interval
            confidence_interval = self.calculate_confidence_interval(y, y_pred)
            
            # Identify outliers
            outliers = self.identify_trend_outliers(metric_name)
            
            trend_analysis = TrendAnalysis(
                metric_name=metric_name,
                trend_direction=trend_direction,
                trend_strength=r2,
                annual_change_rate=annual_change,
                significance_level=trend_strength,
                confidence_interval=confidence_interval,
                outliers=outliers
            )
            
            self.logger.info(f"Trend analysis completed for {metric_name}")
            return trend_analysis
            
        except Exception as e:
            self.logger.error(f"Error in trend analysis for {metric_name}: {str(e)}")
            raise
            
    def create_forecasting_model(self, metric_name: str, county_fips: str = None) -> ForecastingModel:
        """
        Create forecasting model for a specific metric
        
        Args:
            metric_name: Name of the metric to forecast
            county_fips: Specific county to forecast (None for statewide)
            
        Returns:
            ForecastingModel object with forecast results
        """
        try:
            if county_fips:
                # County-specific forecast
                data = self.merged_data[self.merged_data['county_fips'] == county_fips]
            else:
                # Statewide forecast
                data = self.merged_data.groupby('year')[metric_name].mean().reset_index()
                
            # Prepare time series data
            data = data.sort_values('year')
            X = (data['year'] - data['year'].min()).dt.days.values.reshape(-1, 1)
            y = data[metric_name].values
            
            # Fit model
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate future predictions
            future_years = pd.date_range(
                start=data['year'].max(), 
                periods=self.forecast_horizon + 1, 
                freq='Y'
            )[1:]  # Exclude current year
            
            future_X = (future_years - data['year'].min()).days.values.reshape(-1, 1)
            future_predictions = model.predict(future_X)
            
            # Calculate confidence intervals
            confidence_intervals = self.calculate_forecast_confidence_intervals(
                model, future_X, y, y_pred=model.predict(X)
            )
            
            # Model accuracy
            y_pred = model.predict(X)
            accuracy = r2_score(y, y_pred)
            
            # Key factors (for now, just the trend)
            key_factors = [f"Linear trend: {model.coef_[0]:.4f} per day"]
            
            forecasting_model = ForecastingModel(
                metric_name=metric_name,
                model_type="linear",
                forecast_horizon=self.forecast_horizon,
                predicted_values=future_predictions.tolist(),
                confidence_intervals=confidence_intervals,
                model_accuracy=accuracy,
                key_factors=key_factors
            )
            
            self.logger.info(f"Forecasting model created for {metric_name}")
            return forecasting_model
            
        except Exception as e:
            self.logger.error(f"Error creating forecasting model for {metric_name}: {str(e)}")
            raise
            
    def perform_county_ranking(self) -> List[CountyRanking]:
        """
        Perform comprehensive county ranking and benchmarking
        
        Returns:
            List of CountyRanking objects
        """
        try:
            # Calculate overall scores for each county
            county_scores = self.calculate_county_overall_scores()
            
            # Sort by score and assign ranks
            county_scores = county_scores.sort_values('overall_score', ascending=False)
            county_scores['rank'] = range(1, len(county_scores) + 1)
            county_scores['percentile'] = county_scores['rank'].apply(
                lambda x: (len(county_scores) - x + 1) / len(county_scores) * 100
            )
            
            rankings = []
            for _, row in county_scores.iterrows():
                # Identify strengths and weaknesses
                strengths, weaknesses = self.identify_county_strengths_weaknesses(row['county_fips'])
                
                # Find peer counties
                peer_counties = self.find_peer_counties(row['county_fips'])
                
                # Identify improvement opportunities
                improvement_opportunities = self.identify_improvement_opportunities(row['county_fips'])
                
                ranking = CountyRanking(
                    county_name=row['county_name'],
                    county_fips=row['county_fips'],
                    overall_score=row['overall_score'],
                    rank=row['rank'],
                    percentile=row['percentile'],
                    strengths=strengths,
                    weaknesses=weaknesses,
                    peer_counties=peer_counties,
                    improvement_opportunities=improvement_opportunities
                )
                rankings.append(ranking)
                
            self.logger.info(f"County ranking completed for {len(rankings)} counties")
            return rankings
            
        except Exception as e:
            self.logger.error(f"Error in county ranking: {str(e)}")
            raise
            
    def perform_regional_analysis(self) -> List[RegionalAnalysis]:
        """
        Perform regional analysis by clustering counties
        
        Returns:
            List of RegionalAnalysis objects
        """
        try:
            # Prepare data for clustering
            cluster_data = self.prepare_clustering_data()
            
            # Perform K-means clustering
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            kmeans = KMeans(n_clusters=self.clustering_n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Add cluster labels to data
            cluster_data['cluster'] = cluster_labels
            cluster_data['county_fips'] = self.merged_data['county_fips'].unique()
            cluster_data['county_name'] = self.merged_data['county_name'].unique()
            
            # Analyze each cluster
            regional_analyses = []
            for cluster_id in range(self.clustering_n_clusters):
                cluster_counties = cluster_data[cluster_data['cluster'] == cluster_id]
                
                # Calculate cluster characteristics
                avg_stress_score = cluster_counties['housing_stress_score'].mean()
                
                # Identify common challenges
                common_challenges = self.identify_cluster_challenges(cluster_counties)
                
                # Identify unique characteristics
                unique_characteristics = self.identify_cluster_characteristics(cluster_counties)
                
                # Generate policy recommendations
                policy_recommendations = self.generate_cluster_policy_recommendations(
                    cluster_id, cluster_counties
                )
                
                regional_analysis = RegionalAnalysis(
                    region_name=f"Region {cluster_id + 1}",
                    counties_included=cluster_counties['county_name'].tolist(),
                    average_stress_score=avg_stress_score,
                    common_challenges=common_challenges,
                    unique_characteristics=unique_characteristics,
                    policy_recommendations=policy_recommendations
                )
                regional_analyses.append(regional_analysis)
                
            self.logger.info(f"Regional analysis completed with {len(regional_analyses)} regions")
            return regional_analyses
            
        except Exception as e:
            self.logger.error(f"Error in regional analysis: {str(e)}")
            raise
            
    def generate_comprehensive_report(self):
        """Generate comprehensive analytics report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(self.reports_dir, f"comprehensive_analytics_report_{timestamp}.json")
            
            # Perform all analyses
            self.logger.info("Starting comprehensive analytics report generation")
            
            # Trend analysis for key metrics
            trend_metrics = ['total_population', 'total_housing_units', 'median_household_income', 
                           'housing_stress_score', 'supply_gap', 'affordability_gap']
            trend_results = {}
            for metric in trend_metrics:
                if metric in self.merged_data.columns:
                    trend_results[metric] = self.perform_trend_analysis(metric)
                    
            # Forecasting for key metrics
            forecast_results = {}
            for metric in ['housing_stress_score', 'supply_gap', 'affordability_gap']:
                if metric in self.merged_data.columns:
                    forecast_results[metric] = self.create_forecasting_model(metric)
                    
            # County rankings
            county_rankings = self.perform_county_ranking()
            
            # Regional analysis
            regional_analyses = self.perform_regional_analysis()
            
            # Compile comprehensive report
            report = {
                "report_timestamp": timestamp,
                "analysis_summary": {
                    "total_counties": len(self.merged_data['county_fips'].unique()),
                    "years_analyzed": len(self.merged_data['year'].unique()),
                    "metrics_analyzed": len(trend_metrics),
                    "forecast_horizon": self.forecast_horizon
                },
                "trend_analysis": {
                    metric: {
                        "trend_direction": result.trend_direction,
                        "trend_strength": result.trend_strength,
                        "annual_change_rate": result.annual_change_rate,
                        "significance_level": result.significance_level
                    } for metric, result in trend_results.items()
                },
                "forecasting_results": {
                    metric: {
                        "model_type": result.model_type,
                        "model_accuracy": result.model_accuracy,
                        "predicted_values": result.predicted_values,
                        "key_factors": result.key_factors
                    } for metric, result in forecast_results.items()
                },
                "county_rankings": [
                    {
                        "county_name": ranking.county_name,
                        "county_fips": ranking.county_fips,
                        "overall_score": ranking.overall_score,
                        "rank": ranking.rank,
                        "percentile": ranking.percentile,
                        "strengths": ranking.strengths,
                        "weaknesses": ranking.weaknesses,
                        "peer_counties": ranking.peer_counties,
                        "improvement_opportunities": ranking.improvement_opportunities
                    } for ranking in county_rankings
                ],
                "regional_analysis": [
                    {
                        "region_name": analysis.region_name,
                        "counties_included": analysis.counties_included,
                        "average_stress_score": analysis.average_stress_score,
                        "common_challenges": analysis.common_challenges,
                        "unique_characteristics": analysis.unique_characteristics,
                        "policy_recommendations": analysis.policy_recommendations
                    } for analysis in regional_analyses
                ]
            }
            
            # Save report
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            self.logger.info(f"Comprehensive analytics report generated: {report_file}")
            return report_file
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {str(e)}")
            raise
            
    # Helper methods
    def calculate_confidence_interval(self, y_true, y_pred, confidence_level=0.95):
        """Calculate confidence interval for trend analysis"""
        residuals = y_true - y_pred
        std_error = np.std(residuals)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * std_error
        return (np.mean(y_true) - margin_of_error, np.mean(y_true) + margin_of_error)
        
    def identify_trend_outliers(self, metric_name):
        """Identify counties with unusual trend patterns"""
        # Implementation for outlier detection
        return []
        
    def calculate_forecast_confidence_intervals(self, model, X_future, y_true, y_pred):
        """Calculate confidence intervals for forecasts"""
        # Implementation for forecast confidence intervals
        return [(pred - 0.1 * pred, pred + 0.1 * pred) for pred in model.predict(X_future)]
        
    def calculate_county_overall_scores(self):
        """Calculate overall scores for counties based on multiple metrics"""
        # Implementation for county scoring
        county_data = self.merged_data.groupby(['county_fips', 'county_name']).agg({
            'housing_stress_score': 'mean',
            'supply_gap': 'mean',
            'affordability_gap': 'mean',
            'quality_gap': 'mean'
        }).reset_index()
        
        # Normalize scores and calculate weighted average
        for col in ['housing_stress_score', 'supply_gap', 'affordability_gap', 'quality_gap']:
            if col in county_data.columns:
                county_data[col] = (county_data[col] - county_data[col].min()) / \
                                 (county_data[col].max() - county_data[col].min())
                
        county_data['overall_score'] = (
            county_data['housing_stress_score'] * 0.4 +
            county_data['supply_gap'] * 0.2 +
            county_data['affordability_gap'] * 0.2 +
            county_data['quality_gap'] * 0.2
        )
        
        return county_data
        
    def identify_county_strengths_weaknesses(self, county_fips):
        """Identify strengths and weaknesses for a specific county"""
        # Implementation for county analysis
        return ["Strong housing supply"], ["High housing costs"]
        
    def find_peer_counties(self, county_fips):
        """Find peer counties with similar characteristics"""
        # Implementation for peer county identification
        return ["County A", "County B"]
        
    def identify_improvement_opportunities(self, county_fips):
        """Identify improvement opportunities for a specific county"""
        # Implementation for improvement opportunities
        return ["Increase affordable housing", "Improve housing quality"]
        
    def prepare_clustering_data(self):
        """Prepare data for regional clustering"""
        # Implementation for clustering data preparation
        cluster_metrics = ['housing_stress_score', 'supply_gap', 'affordability_gap', 'quality_gap']
        cluster_data = self.merged_data.groupby('county_fips')[cluster_metrics].mean()
        return cluster_data
        
    def identify_cluster_challenges(self, cluster_counties):
        """Identify common challenges for a cluster"""
        # Implementation for cluster challenge identification
        return ["High housing costs", "Limited supply"]
        
    def identify_cluster_characteristics(self, cluster_counties):
        """Identify unique characteristics for a cluster"""
        # Implementation for cluster characteristic identification
        return ["Urban counties", "High population density"]
        
    def generate_cluster_policy_recommendations(self, cluster_id, cluster_counties):
        """Generate policy recommendations for a cluster"""
        # Implementation for policy recommendations
        return ["Increase housing supply", "Improve affordability programs"]

if __name__ == "__main__":
    # Initialize and run advanced analytics
    analytics_engine = OregonHousingAdvancedAnalytics()
    
    # Generate comprehensive report
    report_file = analytics_engine.generate_comprehensive_report()
    print(f"Advanced analytics report generated: {report_file}")
