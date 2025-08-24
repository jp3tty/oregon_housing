#!/usr/bin/env python3
"""
Oregon Housing Visualization Framework - Phase 6
===============================================

This script implements comprehensive visualization capabilities for the Oregon Housing Study:
1. Trend Charts & Time Series Analysis
2. County Comparison & Ranking Visualizations
3. Regional Analysis & Geographic Insights
4. Statistical Charts & Correlation Analysis
5. Tableau Public Export Preparation

Key Features:
- Professional-grade charts and graphs
- Interactive visualization preparation
- Tableau Public compatibility
- Geographic mapping and regional analysis
- Statistical visualization and trend analysis
- Export-ready visualizations for dashboard development
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Optional imports - visualization framework will work without these
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Some interactive features will be disabled.")

try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("Warning: Folium not available. Geographic mapping features will be disabled.")
import logging
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple, Any
import json
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class OregonHousingVisualizationFramework:
    """
    Comprehensive visualization framework for Oregon housing data
    
    This class implements professional visualization capabilities including:
    - Trend analysis charts and time series
    - County comparison and ranking visualizations
    - Regional analysis and geographic insights
    - Statistical charts and correlation analysis
    - Tableau Public export preparation
    """
    
    def __init__(self):
        """Initialize the visualization framework"""
        # Directory structure
        self.output_dir = "Data_Collection_Output"
        self.analytics_dir = os.path.join(self.output_dir, "advanced_analytics")
        self.visualizations_dir = os.path.join(self.analytics_dir, "visualizations")
        self.tableau_export_dir = os.path.join(self.analytics_dir, "tableau_export")
        
        # Setup directories and logging
        self.setup_directories()
        self.setup_logging()
        
        # Load analytics data
        self.load_analytics_data()
        
        # Visualization parameters
        self.chart_style = "professional"
        self.color_palette = "husl"
        self.figure_size = (12, 8)
        self.dpi = 300
        
    def setup_directories(self):
        """Create necessary directories for visualization output"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.analytics_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        os.makedirs(self.tableau_export_dir, exist_ok=True)
        
    def setup_logging(self):
        """Configure logging for visualization framework"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.visualizations_dir, f"visualization_framework_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Visualization Framework initialized")
        
    def load_analytics_data(self):
        """Load analytics data for visualization"""
        try:
            # Load main data files
            self.population_data = pd.read_csv(
                os.path.join(self.output_dir, "oregon_county_population_2009_2023_census_acs.csv")
            )
            self.housing_supply_data = pd.read_csv(
                os.path.join(self.output_dir, "oregon_county_housing_supply_2009_2023_acs.csv")
            )
            self.income_data = pd.read_csv(
                os.path.join(self.output_dir, "oregon_county_income_2009_2023_acs.csv")
            )
            self.homeless_data = pd.read_csv(
                os.path.join(self.output_dir, "oregon_county_homeless_data.csv")
            )
            self.gap_analysis_data = pd.read_csv(
                os.path.join(self.output_dir, "oregon_county_housing_gap_analysis_2009_2023.csv")
            )
            
            # Load analytics report if available
            self.analytics_report = self.load_latest_analytics_report()
            
            self.logger.info("Analytics data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading analytics data: {str(e)}")
            raise
            
    def load_latest_analytics_report(self):
        """Load the most recent analytics report"""
        try:
            reports_dir = os.path.join(self.analytics_dir, "reports")
            if os.path.exists(reports_dir):
                report_files = [f for f in os.listdir(reports_dir) if f.endswith('.json')]
                if report_files:
                    latest_report = max(report_files, key=lambda x: os.path.getctime(os.path.join(reports_dir, x)))
                    with open(os.path.join(reports_dir, latest_report), 'r') as f:
                        return json.load(f)
            return None
        except Exception as e:
            self.logger.warning(f"Could not load analytics report: {str(e)}")
            return None
            
    def create_trend_analysis_charts(self):
        """Create comprehensive trend analysis charts"""
        try:
            self.logger.info("Creating trend analysis charts")
            
            # Prepare data for trend analysis
            trend_data = self.prepare_trend_data()
            
            # Create trend charts for key metrics
            self.create_population_trend_chart(trend_data)
            self.create_housing_supply_trend_chart(trend_data)
            self.create_income_trend_chart(trend_data)
            self.create_stress_score_trend_chart(trend_data)
            self.create_gap_analysis_trend_chart(trend_data)
            
            self.logger.info("Trend analysis charts created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating trend analysis charts: {str(e)}")
            raise
            
    def create_population_trend_chart(self, trend_data):
        """Create population trend visualization"""
        try:
            # Population trends by county
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Select top 10 counties by population
            top_counties = trend_data.groupby('county_name')['total_population'].mean().nlargest(10)
            
            for county in top_counties.index:
                county_data = trend_data[trend_data['county_name'] == county]
                ax.plot(county_data['year'], county_data['total_population'], 
                       marker='o', linewidth=2, label=county)
                
            ax.set_title('Population Trends: Top 10 Oregon Counties (2009-2023)', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Total Population', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Save chart
            chart_file = os.path.join(self.visualizations_dir, 'population_trends.png')
            plt.tight_layout()
            plt.savefig(chart_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Population trend chart saved: {chart_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating population trend chart: {str(e)}")
            
    def create_housing_supply_trend_chart(self, trend_data):
        """Create housing supply trend visualization"""
        try:
            # Housing supply trends
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), dpi=self.dpi)
            
            # Total housing units trend
            housing_trends = trend_data.groupby('year')['total_housing_units'].agg(['mean', 'std']).reset_index()
            ax1.plot(housing_trends['year'], housing_trends['mean'], 
                    marker='o', linewidth=2, color='blue')
            ax1.fill_between(housing_trends['year'], 
                           housing_trends['mean'] - housing_trends['std'],
                           housing_trends['mean'] + housing_trends['std'], 
                           alpha=0.3, color='blue')
            ax1.set_title('Average Housing Units Across Oregon Counties (2009-2023)', 
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Total Housing Units', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Vacancy rate trend
            vacancy_trends = trend_data.groupby('year')['vacancy_rate_percent'].mean().reset_index()
            ax2.plot(vacancy_trends['year'], vacancy_trends['vacancy_rate_percent'], 
                    marker='s', linewidth=2, color='red')
            ax2.set_title('Average Vacancy Rate Across Oregon Counties (2009-2023)', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Year', fontsize=12)
            ax2.set_ylabel('Vacancy Rate (%)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Save chart
            chart_file = os.path.join(self.visualizations_dir, 'housing_supply_trends.png')
            plt.tight_layout()
            plt.savefig(chart_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Housing supply trend chart saved: {chart_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating housing supply trend chart: {str(e)}")
            
    def create_income_trend_chart(self, trend_data):
        """Create income trend visualization"""
        try:
            # Income trends
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Median household income trends by county
            top_income_counties = trend_data.groupby('county_name')['median_household_income'].mean().nlargest(8)
            
            for county in top_income_counties.index:
                county_data = trend_data[trend_data['county_name'] == county]
                ax.plot(county_data['year'], county_data['median_household_income'], 
                       marker='o', linewidth=2, label=county)
                
            ax.set_title('Median Household Income Trends: Top 8 Oregon Counties (2009-2023)', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Median Household Income ($)', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Save chart
            chart_file = os.path.join(self.visualizations_dir, 'income_trends.png')
            plt.tight_layout()
            plt.savefig(chart_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Income trend chart saved: {chart_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating income trend chart: {str(e)}")
            
    def create_stress_score_trend_chart(self, trend_data):
        """Create housing stress score trend visualization"""
        try:
            # Housing stress score trends
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), dpi=self.dpi)
            
            # Average stress score trend
            stress_trends = trend_data.groupby('year')['housing_stress_score'].agg(['mean', 'std']).reset_index()
            ax1.plot(stress_trends['year'], stress_trends['mean'], 
                    marker='o', linewidth=2, color='purple')
            ax1.fill_between(stress_trends['year'], 
                           stress_trends['mean'] - stress_trends['std'],
                           stress_trends['mean'] + stress_trends['std'], 
                           alpha=0.3, color='purple')
            ax1.set_title('Average Housing Stress Score Across Oregon Counties (2009-2023)', 
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Housing Stress Score (0-100)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Top 5 highest stress counties
            top_stress_counties = trend_data.groupby('county_name')['housing_stress_score'].mean().nlargest(5)
            ax2.bar(range(len(top_stress_counties)), top_stress_counties.values, 
                   color='red', alpha=0.7)
            ax2.set_title('Top 5 Counties by Average Housing Stress Score', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Counties', fontsize=12)
            ax2.set_ylabel('Average Stress Score', fontsize=12)
            ax2.set_xticks(range(len(top_stress_counties)))
            ax2.set_xticklabels(top_stress_counties.index, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Save chart
            chart_file = os.path.join(self.visualizations_dir, 'stress_score_trends.png')
            plt.tight_layout()
            plt.savefig(chart_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Stress score trend chart saved: {chart_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating stress score trend chart: {str(e)}")
            
    def create_gap_analysis_trend_chart(self, trend_data):
        """Create housing gap analysis trend visualization"""
        try:
            # Gap analysis trends
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
            
            # Supply gap trends
            supply_gap_trends = trend_data.groupby('year')['supply_gap'].mean().reset_index()
            ax1.plot(supply_gap_trends['year'], supply_gap_trends['supply_gap'], 
                    marker='o', linewidth=2, color='blue')
            ax1.set_title('Average Supply Gap Trend (2009-2023)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Supply Gap (Units)', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Affordability gap trends
            affordability_gap_trends = trend_data.groupby('year')['affordability_gap'].mean().reset_index()
            ax2.plot(affordability_gap_trends['year'], affordability_gap_trends['affordability_gap'], 
                    marker='s', linewidth=2, color='red')
            ax2.set_title('Average Affordability Gap Trend (2009-2023)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Affordability Gap (Households)', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # Quality gap trends
            quality_gap_trends = trend_data.groupby('year')['quality_gap'].mean().reset_index()
            ax3.plot(quality_gap_trends['year'], quality_gap_trends['quality_gap'], 
                    marker='^', linewidth=2, color='green')
            ax3.set_title('Average Quality Gap Trend (2009-2023)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Year', fontsize=10)
            ax3.set_ylabel('Quality Gap (Units)', fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # Homeless gap trends
            homeless_gap_trends = trend_data.groupby('year')['homeless_gap'].mean().reset_index()
            ax4.plot(homeless_gap_trends['year'], homeless_gap_trends['homeless_gap'], 
                    marker='d', linewidth=2, color='orange')
            ax4.set_title('Average Homeless Gap Trend (2009-2023)', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Year', fontsize=10)
            ax4.set_ylabel('Homeless Gap (Individuals)', fontsize=10)
            ax4.grid(True, alpha=0.3)
            
            # Save chart
            chart_file = os.path.join(self.visualizations_dir, 'gap_analysis_trends.png')
            plt.tight_layout()
            plt.savefig(chart_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Gap analysis trend chart saved: {chart_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating gap analysis trend chart: {str(e)}")
            
    def create_county_comparison_charts(self):
        """Create county comparison and ranking visualizations"""
        try:
            self.logger.info("Creating county comparison charts")
            
            # Prepare comparison data
            comparison_data = self.prepare_comparison_data()
            
            # Create comparison charts
            self.create_county_ranking_chart(comparison_data)
            self.create_county_performance_radar(comparison_data)
            self.create_county_correlation_matrix(comparison_data)
            
            self.logger.info("County comparison charts created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating county comparison charts: {str(e)}")
            raise
            
    def create_county_ranking_chart(self, comparison_data):
        """Create county ranking visualization"""
        try:
            # Top and bottom 10 counties by stress score
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), dpi=self.dpi)
            
            # Top 10 counties (lowest stress scores)
            top_counties = comparison_data.nsmallest(10, 'housing_stress_score')
            bars1 = ax1.barh(range(len(top_counties)), top_counties['housing_stress_score'], 
                           color='green', alpha=0.7)
            ax1.set_title('Top 10 Counties by Housing Performance (Lowest Stress Scores)', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Housing Stress Score', fontsize=12)
            ax1.set_yticks(range(len(top_counties)))
            ax1.set_yticklabels(top_counties['county_name'])
            ax1.grid(True, alpha=0.3)
            
            # Bottom 10 counties (highest stress scores)
            bottom_counties = comparison_data.nlargest(10, 'housing_stress_score')
            bars2 = ax2.barh(range(len(bottom_counties)), bottom_counties['housing_stress_score'], 
                            color='red', alpha=0.7)
            ax2.set_title('Bottom 10 Counties by Housing Performance (Highest Stress Scores)', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Housing Stress Score', fontsize=12)
            ax2.set_yticks(range(len(bottom_counties)))
            ax2.set_yticklabels(bottom_counties['county_name'])
            ax2.grid(True, alpha=0.3)
            
            # Save chart
            chart_file = os.path.join(self.visualizations_dir, 'county_rankings.png')
            plt.tight_layout()
            plt.savefig(chart_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"County ranking chart saved: {chart_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating county ranking chart: {str(e)}")
            
    def create_county_performance_radar(self, comparison_data):
        """Create county performance radar chart"""
        try:
            # Select top 5 counties for radar chart
            top_5_counties = comparison_data.nsmallest(5, 'housing_stress_score')
            
            # Prepare radar chart data
            categories = ['Housing Supply', 'Affordability', 'Quality', 'Homeless Support']
            
            fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi, subplot_kw=dict(projection='polar'))
            
            # Create radar chart for each county
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, (_, county) in enumerate(top_5_counties.iterrows()):
                values = [
                    1 - (county['supply_gap'] / comparison_data['supply_gap'].max()),
                    1 - (county['affordability_gap'] / comparison_data['affordability_gap'].max()),
                    1 - (county['quality_gap'] / comparison_data['quality_gap'].max()),
                    1 - (county['homeless_gap'] / comparison_data['homeless_gap'].max())
                ]
                
                # Close the radar chart
                values += values[:1]
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, label=county['county_name'], color=colors[i])
                ax.fill(angles, values, alpha=0.1, color=colors[i])
                
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title('County Performance Radar Chart - Top 5 Counties', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            # Save chart
            chart_file = os.path.join(self.visualizations_dir, 'county_performance_radar.png')
            plt.tight_layout()
            plt.savefig(chart_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"County performance radar chart saved: {chart_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating county performance radar chart: {str(e)}")
            
    def create_county_correlation_matrix(self, comparison_data):
        """Create county correlation matrix visualization"""
        try:
            # Select numeric columns for correlation
            numeric_cols = ['housing_stress_score', 'supply_gap', 'affordability_gap', 
                           'quality_gap', 'homeless_gap', 'total_population', 
                           'total_housing_units', 'median_household_income']
            
            correlation_data = comparison_data[numeric_cols].corr()
            
            # Create correlation heatmap
            fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)
            sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            
            ax.set_title('County Metrics Correlation Matrix', fontsize=16, fontweight='bold')
            
            # Save chart
            chart_file = os.path.join(self.visualizations_dir, 'county_correlation_matrix.png')
            plt.tight_layout()
            plt.savefig(chart_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"County correlation matrix saved: {chart_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating county correlation matrix: {str(e)}")
            
    def create_regional_analysis_charts(self):
        """Create regional analysis and geographic visualizations"""
        try:
            self.logger.info("Creating regional analysis charts")
            
            # Create regional charts
            self.create_regional_cluster_chart()
            self.create_geographic_distribution_chart()
            
            self.logger.info("Regional analysis charts created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating regional analysis charts: {str(e)}")
            raise
            
    def create_regional_cluster_chart(self):
        """Create regional clustering visualization"""
        try:
            # Use analytics report data if available
            if self.analytics_report and 'regional_analysis' in self.analytics_report:
                regional_data = self.analytics_report['regional_analysis']
                
                # Create regional comparison chart
                fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
                
                regions = [region['region_name'] for region in regional_data]
                stress_scores = [region['average_stress_score'] for region in regional_data]
                county_counts = [len(region['counties_included']) for region in regional_data]
                
                # Create bar chart
                bars = ax.bar(regions, stress_scores, color=['blue', 'green', 'red', 'orange'], alpha=0.7)
                ax.set_title('Regional Housing Stress Score Comparison', fontsize=16, fontweight='bold')
                ax.set_xlabel('Regions', fontsize=12)
                ax.set_ylabel('Average Housing Stress Score', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Add county count labels
                for i, (bar, count) in enumerate(zip(bars, county_counts)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{count} counties', ha='center', va='bottom', fontweight='bold')
                
                # Save chart
                chart_file = os.path.join(self.visualizations_dir, 'regional_cluster_analysis.png')
                plt.tight_layout()
                plt.savefig(chart_file, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Regional cluster chart saved: {chart_file}")
            else:
                self.logger.warning("No regional analysis data available for visualization")
                
        except Exception as e:
            self.logger.error(f"Error creating regional cluster chart: {str(e)}")
            
    def create_geographic_distribution_chart(self):
        """Create geographic distribution visualization"""
        try:
            # Create a simple geographic representation using county data
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Group counties by stress score ranges
            stress_ranges = pd.cut(self.gap_analysis_data['housing_stress_score'], 
                                 bins=[0, 25, 50, 75, 100], 
                                 labels=['Low (0-25)', 'Medium (25-50)', 'High (50-75)', 'Very High (75-100)'])
            
            stress_distribution = stress_ranges.value_counts().sort_index()
            
            # Create pie chart
            colors = ['green', 'yellow', 'orange', 'red']
            wedges, texts, autotexts = ax.pie(stress_distribution.values, 
                                             labels=stress_distribution.index,
                                             autopct='%1.1f%%', 
                                             colors=colors, 
                                             startangle=90)
            
            ax.set_title('Distribution of Counties by Housing Stress Level', fontsize=16, fontweight='bold')
            
            # Save chart
            chart_file = os.path.join(self.visualizations_dir, 'geographic_distribution.png')
            plt.tight_layout()
            plt.savefig(chart_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Geographic distribution chart saved: {chart_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating geographic distribution chart: {str(e)}")
            
    def prepare_tableau_export(self):
        """Prepare data for Tableau Public export"""
        try:
            self.logger.info("Preparing Tableau Public export data")
            
            # Create Tableau-ready datasets
            self.create_tableau_population_dataset()
            self.create_tableau_housing_dataset()
            self.create_tableau_income_dataset()
            self.create_tableau_gap_analysis_dataset()
            self.create_tableau_summary_dataset()
            self.create_tableau_homeless_dataset()
            
            # Create Tableau data dictionary
            self.create_tableau_data_dictionary()
            
            self.logger.info("Tableau Public export data prepared successfully")
            
        except Exception as e:
            self.logger.error(f"Error preparing Tableau export: {str(e)}")
            raise
            
    def create_tableau_population_dataset(self):
        """Create Tableau-ready population dataset with geographic coordinates"""
        try:
            # Clean and format population data for Tableau
            tableau_population = self.population_data.copy()
            
            # Ensure year is in proper format (already an integer, no conversion needed)
            # tableau_population['year'] is already in the correct format (2009, 2010, etc.)
            
            # Add calculated fields
            tableau_population['population_change'] = tableau_population.groupby('county_fips')['total_population'].diff()
            tableau_population['population_change_pct'] = tableau_population.groupby('county_fips')['total_population'].pct_change() * 100
            
            # Add geographic coordinates for mapping
            try:
                from oregon_counties_geographic import OregonCountiesGeographic
                geo = OregonCountiesGeographic()
                geo_df = geo.get_all_coordinates()
                
                # Merge geographic data
                tableau_population = tableau_population.merge(
                    geo_df[['county_fips', 'latitude', 'longitude', 'centroid_lat', 'centroid_lng']], 
                    on='county_fips', 
                    how='left'
                )
                
                self.logger.info("Geographic coordinates added to population dataset")
            except ImportError:
                self.logger.warning("Geographic module not available, coordinates not added")
            
            # Save Tableau-ready dataset
            export_file = os.path.join(self.tableau_export_dir, 'tableau_population_data.csv')
            tableau_population.to_csv(export_file, index=False)
            
            self.logger.info(f"Tableau population dataset created: {export_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating Tableau population dataset: {str(e)}")
            
    def create_tableau_housing_dataset(self):
        """Create Tableau-ready housing dataset with geographic coordinates"""
        try:
            # Clean and format housing data for Tableau
            tableau_housing = self.housing_supply_data.copy()
            
            # Ensure year is in proper format (already an integer, no conversion needed)
            # tableau_housing['year'] is already in the correct format (2009, 2010, etc.)
            
            # Add calculated fields
            tableau_housing['housing_growth'] = tableau_housing.groupby('county_fips')['total_housing_units'].diff()
            tableau_housing['housing_growth_pct'] = tableau_housing.groupby('county_fips')['total_housing_units'].pct_change() * 100
            
            # Add geographic coordinates for mapping
            try:
                from oregon_counties_geographic import OregonCountiesGeographic
                geo = OregonCountiesGeographic()
                geo_df = geo.get_all_coordinates()
                
                # Merge geographic data
                tableau_housing = tableau_housing.merge(
                    geo_df[['county_fips', 'latitude', 'longitude', 'centroid_lat', 'centroid_lng']], 
                    on='county_fips', 
                    how='left'
                )
                
                self.logger.info("Geographic coordinates added to housing dataset")
            except ImportError:
                self.logger.warning("Geographic module not available, coordinates not added")
            
            # Save Tableau-ready dataset
            export_file = os.path.join(self.tableau_export_dir, 'tableau_housing_data.csv')
            tableau_housing.to_csv(export_file, index=False)
            
            self.logger.info(f"Tableau housing dataset created: {export_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating Tableau housing dataset: {str(e)}")
            
    def create_tableau_income_dataset(self):
        """Create Tableau-ready income dataset with geographic coordinates"""
        try:
            # Clean and format income data for Tableau
            tableau_income = self.income_data.copy()
            
            # Ensure year is in proper format (already an integer, no conversion needed)
            # tableau_income['year'] is already in the correct format (2009, 2010, etc.)
            
            # Add calculated fields
            tableau_income['income_change'] = tableau_income.groupby('county_fips')['median_household_income'].diff()
            tableau_income['income_change_pct'] = tableau_income.groupby('county_fips')['median_household_income'].pct_change() * 100
            
            # Add geographic coordinates for mapping
            try:
                from oregon_counties_geographic import OregonCountiesGeographic
                geo = OregonCountiesGeographic()
                geo_df = geo.get_all_coordinates()
                
                # Merge geographic data
                tableau_income = tableau_income.merge(
                    geo_df[['county_fips', 'latitude', 'longitude', 'centroid_lat', 'centroid_lng']], 
                    on='county_fips', 
                    how='left'
                )
                
                self.logger.info("Geographic coordinates added to income dataset")
            except ImportError:
                self.logger.warning("Geographic module not available, coordinates not added")
            
            # Save Tableau-ready dataset
            export_file = os.path.join(self.tableau_export_dir, 'tableau_income_data.csv')
            tableau_income.to_csv(export_file, index=False)
            
            self.logger.info(f"Tableau income dataset created: {export_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating Tableau income dataset: {str(e)}")
            
    def create_tableau_gap_analysis_dataset(self):
        """Create Tableau-ready gap analysis dataset with geographic coordinates"""
        try:
            # Clean and format gap analysis data for Tableau
            tableau_gap = self.gap_analysis_data.copy()
            
            # Ensure year is in proper format (already an integer, no conversion needed)
            # tableau_gap['year'] is already in the correct format (2009, 2010, etc.)
            
            # Add calculated fields
            tableau_gap['total_gap_score'] = (
                tableau_gap['supply_gap'] + 
                tableau_gap['affordability_gap'] + 
                tableau_gap['quality_gap'] + 
                tableau_gap['homeless_gap']
            )
            
            # Add geographic coordinates for mapping
            try:
                from oregon_counties_geographic import OregonCountiesGeographic
                geo = OregonCountiesGeographic()
                geo_df = geo.get_all_coordinates()
                
                # Merge geographic data
                tableau_gap = tableau_gap.merge(
                    geo_df[['county_fips', 'latitude', 'longitude', 'centroid_lat', 'centroid_lng']], 
                    on='county_fips', 
                    how='left'
                )
                
                self.logger.info("Geographic coordinates added to gap analysis dataset")
            except ImportError:
                self.logger.warning("Geographic module not available, coordinates not added")
            
            # Save Tableau-ready dataset
            export_file = os.path.join(self.tableau_export_dir, 'tableau_gap_analysis_data.csv')
            tableau_gap.to_csv(export_file, index=False)
            
            self.logger.info(f"Tableau gap analysis dataset created: {export_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating Tableau gap analysis dataset: {str(e)}")
            
    def create_tableau_summary_dataset(self):
        """Create Tableau-ready summary dataset"""
        try:
            # Create a comprehensive summary dataset for Tableau
            summary_data = []
            
            for county_fips in self.gap_analysis_data['county_fips'].unique():
                county_name = self.gap_analysis_data[self.gap_analysis_data['county_fips'] == county_fips]['county_name'].iloc[0]
                
                # Get latest data for each county
                latest_gap = self.gap_analysis_data[self.gap_analysis_data['county_fips'] == county_fips].iloc[-1]
                latest_population = self.population_data[self.population_data['county_fips'] == county_fips].iloc[-1]
                latest_housing = self.housing_supply_data[self.housing_supply_data['county_fips'] == county_fips].iloc[-1]
                latest_income = self.income_data[self.income_data['county_fips'] == county_fips].iloc[-1]
                
                summary_row = {
                    'county_fips': county_fips,
                    'county_name': county_name,
                    'latest_year': latest_gap['year'],
                    'total_population': latest_population['total_population'],
                    'total_housing_units': latest_housing['total_housing_units'],
                    'median_household_income': latest_income['median_household_income'],
                    'housing_stress_score': latest_gap['housing_stress_score'],
                    'supply_gap': latest_gap['supply_gap'],
                    'affordability_gap': latest_gap['affordability_gap'],
                    'quality_gap': latest_gap['quality_gap'],
                    'homeless_gap': latest_gap['homeless_gap'],
                    'vacancy_rate': latest_housing['vacancy_rate_percent'],
                    'homeownership_rate': latest_housing['homeownership_rate_percent']
                }
                
                summary_data.append(summary_row)
                
            # Convert to DataFrame
            summary_df = pd.DataFrame(summary_data)
            
            # Add geographic coordinates for mapping
            try:
                from oregon_counties_geographic import OregonCountiesGeographic
                geo = OregonCountiesGeographic()
                geo_df = geo.get_all_coordinates()
                
                # Merge geographic data
                summary_df = summary_df.merge(
                    geo_df[['county_fips', 'latitude', 'longitude', 'centroid_lat', 'centroid_lng']], 
                    on='county_fips', 
                    how='left'
                )
                
                self.logger.info("Geographic coordinates added to summary dataset")
            except ImportError:
                self.logger.warning("Geographic module not available, coordinates not added")
            
            # Save Tableau-ready dataset
            export_file = os.path.join(self.tableau_export_dir, 'tableau_summary_data.csv')
            summary_df.to_csv(export_file, index=False)
            
            self.logger.info(f"Tableau summary dataset created: {export_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating Tableau summary dataset: {str(e)}")
            
    def create_tableau_homeless_dataset(self):
        """Create Tableau-ready homeless dataset with detailed shelter status and geographic coordinates"""
        try:
            # Load the detailed homeless data
            homeless_file = os.path.join(self.output_dir, "oregon_county_homeless_data.csv")
            if not os.path.exists(homeless_file):
                self.logger.warning("Homeless data file not found, skipping homeless dataset creation")
                return
                
            homeless_data = pd.read_csv(homeless_file)
            
            # Include ALL counties, not just those with PIT data
            # Get unique counties and years to create complete dataset
            all_counties = homeless_data['county_fips'].unique()
            all_years = homeless_data['year'].unique()
            
            # Create complete dataset with all counties and years
            complete_records = []
            for county_fips in all_counties:
                for year in all_years:
                    # Get data for this county/year combination
                    county_data = homeless_data[(homeless_data['county_fips'] == county_fips) & 
                                              (homeless_data['year'] == year)]
                    
                    if not county_data.empty:
                        # County has data for this year
                        if 'hud_pit' in county_data['data_source'].values:
                            # Use PIT data if available
                            row_data = county_data[county_data['data_source'] == 'hud_pit'].iloc[0]
                        else:
                            # Use any available data
                            row_data = county_data.iloc[0]
                        
                        record = {
                            'year': year,
                            'county_fips': county_fips,
                            'county_name': row_data['county_name'],
                            'total_homeless': row_data['total_homeless'],
                            'sheltered_homeless': row_data['sheltered_homeless'],
                            'unsheltered_homeless': row_data['unsheltered_homeless'],
                            'chronic_homeless': row_data['chronic_homeless'],
                            'homeless_families': row_data['homeless_families'],
                            'homeless_veterans': row_data['homeless_veterans'],
                            'shelter_capacity': row_data['shelter_capacity'],
                            'shelter_utilization_rate': row_data['shelter_utilization_rate'],
                            'emergency_shelter_beds': row_data['emergency_shelter_beds'],
                            'transitional_housing_beds': row_data['transitional_housing_beds'],
                            'permanent_supportive_housing': row_data['permanent_supportive_housing']
                        }
                    else:
                        # No data for this county/year - create record with null values
                        county_name = homeless_data[homeless_data['county_fips'] == county_fips]['county_name'].iloc[0]
                        record = {
                            'year': year,
                            'county_fips': county_fips,
                            'county_name': county_name,
                            'total_homeless': None,
                            'sheltered_homeless': None,
                            'unsheltered_homeless': None,
                            'chronic_homeless': None,
                            'homeless_families': None,
                            'homeless_veterans': None,
                            'shelter_capacity': None,
                            'shelter_utilization_rate': None,
                            'emergency_shelter_beds': None,
                            'transitional_housing_beds': None,
                            'permanent_supportive_housing': None
                        }
                    
                    complete_records.append(record)
            
            tableau_homeless = pd.DataFrame(complete_records)
            
            # Add calculated fields
            tableau_homeless['homeless_change'] = tableau_homeless.groupby('county_fips')['total_homeless'].diff()
            tableau_homeless['homeless_change_pct'] = tableau_homeless.groupby('county_fips')['total_homeless'].pct_change() * 100
            
            # Calculate percentages only for non-null values
            mask = tableau_homeless['total_homeless'].notna()
            tableau_homeless.loc[mask, 'sheltered_pct'] = (tableau_homeless.loc[mask, 'sheltered_homeless'] / 
                                                          tableau_homeless.loc[mask, 'total_homeless'] * 100).round(1)
            tableau_homeless.loc[mask, 'unsheltered_pct'] = (tableau_homeless.loc[mask, 'unsheltered_homeless'] / 
                                                            tableau_homeless.loc[mask, 'total_homeless'] * 100).round(1)
            
            # Add geographic coordinates for mapping
            try:
                from oregon_counties_geographic import OregonCountiesGeographic
                geo = OregonCountiesGeographic()
                geo_df = geo.get_all_coordinates()
                
                # Merge geographic data
                tableau_homeless = tableau_homeless.merge(
                    geo_df[['county_fips', 'latitude', 'longitude', 'centroid_lat', 'centroid_lng']], 
                    on='county_fips', 
                    how='left'
                )
                
                self.logger.info("Geographic coordinates added to homeless dataset")
            except ImportError:
                self.logger.warning("Geographic module not available, coordinates not added")
            
            # Save Tableau-ready dataset
            export_file = os.path.join(self.tableau_export_dir, 'tableau_homeless_data.csv')
            
            # Replace NaN values with empty strings for better Tableau compatibility
            tableau_homeless_export = tableau_homeless.copy()
            tableau_homeless_export = tableau_homeless_export.fillna('')
            
            tableau_homeless_export.to_csv(export_file, index=False)
            
            self.logger.info(f"Tableau homeless dataset created: {export_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating Tableau homeless dataset: {str(e)}")
            
    def create_tableau_data_dictionary(self):
        """Create Tableau data dictionary"""
        try:
            data_dictionary = {
                "tableau_population_data.csv": {
                    "description": "Population data for all Oregon counties from 2009-2023 with geographic coordinates",
                    "fields": {
                        "year": "Data collection year",
                        "county_fips": "County FIPS code",
                        "county_name": "County name",
                        "total_population": "Total population count",
                        "data_source": "Source of population data",
                        "population_change": "Year-over-year population change",
                        "population_change_pct": "Year-over-year population change percentage",
                        "latitude": "County latitude coordinate for mapping",
                        "longitude": "County longitude coordinate for mapping",
                        "centroid_lat": "County centroid latitude for precise mapping",
                        "centroid_lng": "County centroid longitude for precise mapping"
                    }
                },
                "tableau_housing_data.csv": {
                    "description": "Housing supply data for all Oregon counties from 2009-2023 with geographic coordinates",
                    "fields": {
                        "year": "Data collection year",
                        "county_fips": "County FIPS code",
                        "county_name": "County name",
                        "total_housing_units": "Total housing units available",
                        "vacancy_rate_percent": "Vacancy rate percentage",
                        "homeownership_rate_percent": "Homeownership rate percentage",
                        "median_home_value": "Median home value",
                        "median_gross_rent": "Median monthly rent",
                        "housing_growth": "Year-over-year housing unit growth",
                        "housing_growth_pct": "Year-over-year housing growth percentage",
                        "latitude": "County latitude coordinate for mapping",
                        "longitude": "County longitude coordinate for mapping",
                        "centroid_lat": "County centroid latitude for precise mapping",
                        "centroid_lng": "County centroid longitude for precise mapping"
                    }
                },
                "tableau_income_data.csv": {
                    "description": "Income and affordability data for all Oregon counties from 2009-2023 with geographic coordinates",
                    "fields": {
                        "year": "Data collection year",
                        "county_fips": "County FIPS code",
                        "county_name": "County name",
                        "median_household_income": "Median household income",
                        "income_change": "Year-over-year income change",
                        "income_change_pct": "Year-over-year income change percentage",
                        "latitude": "County latitude coordinate for mapping",
                        "longitude": "County longitude coordinate for mapping",
                        "centroid_lat": "County centroid latitude for precise mapping",
                        "centroid_lng": "County centroid longitude for precise mapping"
                    }
                },
                "tableau_gap_analysis_data.csv": {
                    "description": "Comprehensive housing gap analysis for all Oregon counties from 2009-2023 with geographic coordinates",
                    "fields": {
                        "year": "Analysis year",
                        "county_fips": "County FIPS code",
                        "county_name": "County name",
                        "housing_stress_score": "Overall housing stress score (0-100)",
                        "supply_gap": "Housing supply gap (units needed vs. available)",
                        "affordability_gap": "Affordability gap (households unable to afford housing)",
                        "quality_gap": "Housing quality gap (substandard units)",
                        "homeless_gap": "Homeless population gap",
                        "total_gap_score": "Combined gap score across all dimensions",
                        "latitude": "County latitude coordinate for mapping",
                        "longitude": "County longitude coordinate for mapping",
                        "centroid_lat": "County centroid latitude for precise mapping",
                        "centroid_lng": "County centroid longitude for precise mapping"
                    }
                },
                "tableau_summary_data.csv": {
                    "description": "Latest summary data for all Oregon counties with geographic coordinates",
                    "fields": {
                        "county_fips": "County FIPS code",
                        "county_name": "County name",
                        "latest_year": "Most recent data year",
                        "total_population": "Latest population count",
                        "total_housing_units": "Latest housing unit count",
                        "median_household_income": "Latest median household income",
                        "housing_stress_score": "Latest housing stress score",
                        "supply_gap": "Latest supply gap",
                        "affordability_gap": "Latest affordability gap",
                        "quality_gap": "Latest quality gap",
                        "homeless_gap": "Latest homeless gap",
                        "vacancy_rate": "Latest vacancy rate",
                        "homeownership_rate": "Latest homeownership rate",
                        "latitude": "County latitude coordinate for mapping",
                        "longitude": "County longitude coordinate for mapping",
                        "centroid_lat": "County centroid latitude for precise mapping",
                        "centroid_lng": "County centroid longitude for precise mapping"
                    }
                },
                "tableau_homeless_data.csv": {
                    "description": "Detailed homeless data with shelter status breakdown for all Oregon counties from 2007-2023 with geographic coordinates",
                    "fields": {
                        "year": "Data collection year",
                        "county_fips": "County FIPS code",
                        "county_name": "County name",
                        "total_homeless": "Total homeless count (sheltered + unsheltered)",
                        "sheltered_homeless": "Homeless in shelters/transitional housing",
                        "unsheltered_homeless": "Homeless on streets/outdoors",
                        "chronic_homeless": "Chronic homelessness (1+ year)",
                        "homeless_families": "Homeless families with children",
                        "homeless_veterans": "Homeless veterans",
                        "shelter_capacity": "Total shelter beds available",
                        "shelter_utilization_rate": "Percentage of shelter capacity used",
                        "emergency_shelter_beds": "Emergency shelter bed count",
                        "transitional_housing_beds": "Transitional housing bed count",
                        "permanent_supportive_housing": "Permanent supportive housing units",
                        "homeless_change": "Year-over-year homeless count change",
                        "homeless_change_pct": "Year-over-year homeless change percentage",
                        "sheltered_pct": "Percentage of homeless who are sheltered",
                        "unsheltered_pct": "Percentage of homeless who are unsheltered",
                        "latitude": "County latitude coordinate for mapping",
                        "longitude": "County longitude coordinate for mapping",
                        "centroid_lat": "County centroid latitude for precise mapping",
                        "centroid_lng": "County centroid longitude for precise mapping"
                    }
                }
            }
            
            # Save data dictionary
            dictionary_file = os.path.join(self.tableau_export_dir, 'tableau_data_dictionary.json')
            with open(dictionary_file, 'w') as f:
                json.dump(data_dictionary, f, indent=2)
                
            self.logger.info(f"Tableau data dictionary created: {dictionary_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating Tableau data dictionary: {str(e)}")
            
    def generate_all_visualizations(self):
        """Generate all visualizations and prepare Tableau export"""
        try:
            self.logger.info("Starting comprehensive visualization generation")
            
            # Create all chart types
            self.create_trend_analysis_charts()
            self.create_county_comparison_charts()
            self.create_regional_analysis_charts()
            
            # Prepare Tableau export
            self.prepare_tableau_export()
            
            self.logger.info("All visualizations and Tableau export completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in visualization generation: {str(e)}")
            raise
            
    # Helper methods
    def prepare_trend_data(self):
        """Prepare data for trend analysis"""
        # Clean up duplicate columns before merging
        columns_to_remove = ['data_source', 'margin_of_error', 'data_quality_score', 'collection_date', 'last_updated']
        
        # Create clean copies for merging
        pop_clean = self.population_data.drop(columns=[col for col in columns_to_remove if col in self.population_data.columns])
        housing_clean = self.housing_supply_data.drop(columns=[col for col in columns_to_remove if col in self.housing_supply_data.columns])
        income_clean = self.income_data.drop(columns=[col for col in columns_to_remove if col in self.income_data.columns])
        gap_clean = self.gap_analysis_data.drop(columns=[col for col in columns_to_remove if col in self.gap_analysis_data.columns])
        
        # Merge datasets for trend analysis
        trend_data = pop_clean.merge(
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
        
        # Convert year to datetime
        trend_data['year'] = pd.to_datetime(trend_data['year'])
        
        return trend_data
        
    def prepare_comparison_data(self):
        """Prepare data for county comparison"""
        # Get latest data for each county
        comparison_data = self.gap_analysis_data.groupby(['county_fips', 'county_name']).agg({
            'housing_stress_score': 'mean',
            'supply_gap': 'mean',
            'affordability_gap': 'mean',
            'quality_gap': 'mean',
            'homeless_gap': 'mean'
        }).reset_index()
        
        # Add population and housing data
        latest_population = self.population_data.groupby(['county_fips', 'county_name'])['total_population'].last().reset_index()
        latest_housing = self.housing_supply_data.groupby(['county_fips', 'county_name'])['total_housing_units'].last().reset_index()
        latest_income = self.income_data.groupby(['county_fips', 'county_name'])['median_household_income'].last().reset_index()
        
        comparison_data = comparison_data.merge(latest_population, on=['county_fips', 'county_name'])
        comparison_data = comparison_data.merge(latest_housing, on=['county_fips', 'county_name'])
        comparison_data = comparison_data.merge(latest_income, on=['county_fips', 'county_name'])
        
        return comparison_data

if __name__ == "__main__":
    # Initialize and run visualization framework
    viz_framework = OregonHousingVisualizationFramework()
    
    # Generate all visualizations and Tableau export
    viz_framework.generate_all_visualizations()
    print("Visualization framework completed successfully!")
