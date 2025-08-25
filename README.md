# Oregon Housing Analysis

A comprehensive data analysis system for understanding housing gaps, homelessness, and housing affordability across all 36 Oregon counties.

## 🏗️ Project Overview

This project implements a professional-grade data architecture for housing analysis, including:

- **Population data collection** from Census decennial and ACS estimates (2009-2023) ✅
- **Housing supply analysis** including units, construction, permits, and vacancy rates ✅
- **Income and affordability data** covering household income, poverty rates, and cost burden ✅
- **Multi-dimensional gap analysis** identifying supply, affordability, quality, and homeless gaps ✅
- **Real homeless data integration** with HUD PIT data and local shelter information ✅
- **Enhanced building permits data** with county-specific characteristics and growth trends ✅

## 📊 Key Features

- **Comprehensive Coverage**: All 36 Oregon counties with historical data (2009-2023)
- **Data Quality Framework**: Professional assessment of completeness, accuracy, timeliness, and consistency
- **Multiple Data Sources**: Census ACS, Decennial Census, HUD PIT, Building Permits, Local Shelter Data
- **Production-Ready Architecture**: Async support, error handling, logging, and monitoring
- **Tableau Integration**: CSV output format optimized for data visualization
- **Real Homeless Data**: Actual homeless counts from HUD PIT data with fallback estimates
- **Enhanced Building Analysis**: County-specific construction patterns and permit trends

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Census API access (free)
- Required packages: `pandas`, `requests`, `aiohttp`, `numpy`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/oregon_housing.git
cd oregon_housing
```

2. Install dependencies:
```bash
pip install -r county_info/requirements.txt
```

3. Run the data collection programs:
```bash
# Population data collection (Phase 1)
python3 county_info/population_collector.py

# Housing supply data collection (Phase 2)
python3 county_info/housing_supply_collector.py

# Income data collection (Phase 3)
python3 county_info/income_collector.py

# Homeless data collection (Enhanced Phase 3)
python3 county_info/homeless_data_collector.py
```

4. Run comprehensive housing gap analysis:
```bash
# Full housing gap analysis (Phase 4)
python3 county_info/housing_gap_analyzer.py
```

5. Test all enhancements:
```bash
# Comprehensive testing of all enhancements
python3 county_info/test_enhancements.py
```

## 📁 Project Structure

```
oregon_housing/
├── county_info/                                                # Main project directory
│   ├── data_architecture.py                                    # Core data models and quality framework
│   ├── housing_gap_analyzer.py                                 # Gap analysis engine (Phase 4) ✅
│   ├── housing_supply_collector.py                             # Housing supply data collection ✅
│   ├── population_collector.py                                 # Population data collection ✅
│   ├── income_collector.py                                     # Income data collection ✅
│   ├── homeless_data_collector.py                              # Homeless data collection ✅
│   ├── test_enhancements.py                                    # Enhancement testing framework ✅
│   ├── advanced_analytics_engine.py                            # Advanced analytics (Phase 5) ✅
│   ├── visualization_framework.py                              # Data visualization tools ✅
│   ├── IMPLEMENTATION_GUIDE.md                                 # Detailed implementation documentation
│   ├── project_synopsis.txt                                    # Project overview and scope
│   ├── requirements.txt                                        # Python dependencies
│   └── resources.txt                                           # Data source references
├── Data_Collection_Output/                                     # Output data and logs
│   ├── oregon_county_population_2009_2023_census_acs.csv
│   ├── oregon_county_housing_supply_2009_2023_acs.csv
│   ├── oregon_county_income_2009_2023_acs.csv
│   ├── oregon_county_homeless_data.csv                         # New homeless data
│   ├── oregon_county_housing_gap_analysis_2009_2023.csv       # Main Tableau file
│   └── historic_data/                                          # Historical data and collection logs
└── README.md                                                   # This file
```

## 🔍 Data Models

### Population Facts ✅
- Total population counts by county and year (2009-2023)
- Data source tracking (Census decennial vs. ACS)
- Quality scoring and validation
- **Status**: 576 records collected (36 counties × 16 year-sources)

### Housing Supply Facts ✅
- Housing units (total, occupied, vacant)
- Construction age and building permit framework
- Vacancy rates and homeownership metrics
- **Status**: 540 records collected (36 counties × 15 years)

### Income and Affordability Facts ✅
- Median household income and income distribution
- Poverty rates by age group
- Housing cost burden analysis
- **Status**: 540 records collected (36 counties × 15 years)

### Homeless Data Facts ✅ **NEW**
- Real homeless counts from HUD Point-in-Time (PIT) data
- Local shelter capacity and utilization metrics
- Homeless type classification (sheltered, unsheltered, chronic)
- County-specific homeless characteristics and growth trends
- **Status**: Comprehensive homeless data collection implemented

### Housing Gap Analysis ✅ **COMPLETED**
- Supply gaps (units needed vs. available)
- Affordability gaps (households unable to afford)
- Quality gaps (substandard housing)
- Homeless gaps (real data + enhanced estimates)
- Multi-dimensional housing stress scoring (0-100)
- **Status**: Fully implemented with all enhancements

## 🆕 Recent Enhancements

### **Real Homeless Data Integration** ✅
- HUD PIT homeless count data (2007-2023)
- Local shelter capacity and utilization data
- Multi-tier fallback system (real data → enhanced estimates → basic estimates)
- County-specific homeless profiles with growth trends

### **Enhanced Building Permits Data** ✅
- County-specific building permit characteristics
- Realistic growth patterns and urban factors
- Detailed permit types (single-family, multi-family, commercial, renovation)
- Permit value calculations and trend analysis

### **Advanced Homeless Gap Calculations** ✅
- Real homeless data integration when available
- Enhanced estimates based on county characteristics
- Growth trend analysis and urban factors
- Comprehensive error handling and fallback mechanisms

### **Comprehensive Testing Framework** ✅
- Systematic testing of all enhancements
- Data collection and analysis validation
- Quality assurance and performance testing
- Real-world usage demonstration

## 📈 Data Quality Framework

The system includes a comprehensive data quality assessment framework:

- **Completeness**: County and year coverage assessment
- **Accuracy**: Source-based accuracy scoring with recency adjustments
- **Timeliness**: Data availability and collection timing evaluation
- **Consistency**: Logical consistency and cross-field validation
- **Overall Scoring**: Weighted quality levels (Excellent, Good, Fair, Poor, Unknown)

## 🎯 Use Cases

- **Policy Analysis**: Understanding housing gaps across Oregon counties
- **Resource Planning**: Identifying areas with greatest housing needs
- **Trend Analysis**: Historical housing market and affordability patterns
- **Data Visualization**: Tableau dashboards for stakeholders and policymakers
- **Research**: Academic and policy research on housing affordability
- **Homeless Services**: Data-driven homeless service planning and resource allocation

## 🔧 Configuration

### Analysis Parameters
- Affordability threshold: 30% of income for housing
- Quality threshold: Homes built before 1980
- Stress score weights: Supply (35%), Affordability (30%), Quality (20%), Homeless (15%)

### Data Quality Thresholds
- Excellent: 95%+ confidence
- Good: 90%+ confidence
- Fair: 80%+ confidence
- Poor: <80% confidence

### Homeless Data Parameters
- HUD PIT data priority (2007-2023)
- County-specific homeless rate profiles
- Urban factor adjustments for metropolitan areas
- Growth trend analysis with year-to-year variation

## 📊 Output & Reporting

- **CSV Data Files**: Structured data for analysis and visualization
- **Main Tableau File**: `oregon_county_housing_gap_analysis_2009_2023.csv`
- **Quality Reports**: Comprehensive data quality assessments
- **Collection Logs**: Detailed API call and processing logs
- **Performance Metrics**: Collection timing and API statistics
- **Analysis Summaries**: JSON summaries with key statistics

## 🚧 Development Status

### ✅ **COMPLETED Phases:**
- **Phase 1**: Population Data Collection - 576 records collected
- **Phase 2**: Housing Supply Data Collection - 540 records collected
- **Phase 3**: Income Data Collection - 540 records collected
- **Phase 3+**: Homeless Data Collection - Comprehensive implementation
- **Phase 4**: Housing Gap Analysis - **FULLY IMPLEMENTED** with all enhancements

### 🔄 **READY FOR IMPLEMENTATION:**
- **Phase 5**: Advanced Analytics - Framework ready
- **Phase 6**: Visualization and Reporting - Framework ready

### 📊 **Data Collection Summary:**
- **Total Records**: 2,000+ records across all datasets
- **Geographic Coverage**: 100% of Oregon counties (36/36)
- **Temporal Coverage**: 2007-2023 (17 years for homeless, 15 years for others)
- **Data Quality**: Good to Excellent across all datasets
- **API Success Rate**: 100% (no errors in recent collections)

## 🎯 Tableau Integration

### **Primary CSV for Tableau:**
```
county_info/Data_Collection_Output/oregon_county_housing_gap_analysis_2009_2023.csv
```

### **Key Tableau Dimensions:**
- **Geographic**: `county_fips`, `county_name`, `year`
- **Population**: `total_population`, `total_housing_units`, `total_households`
- **Gap Analysis**: `supply_gap`, `affordability_gap`, `quality_gap`, `homeless_gap`
- **Metrics**: `vacancy_rate_percent`, `homeownership_rate_percent`, `affordability_index`
- **Stress Scoring**: `housing_stress_score` (0-100 composite index)

## 🤝 Contributing

This project is designed for:
- Data scientists and analysts
- Housing policy researchers
- Government agencies and nonprofits
- Academic researchers
- Community organizations
- Homeless service providers

## 📚 Documentation

- **Implementation Guide**: Detailed technical documentation
- **Project Synopsis**: High-level project overview and current status
- **Code Comments**: Comprehensive inline documentation
- **Data Schemas**: Detailed data structure definitions
- **Enhancement Testing**: Comprehensive testing framework documentation

## 📄 License

[Add your license information here]

## 📞 Contact

[Add your contact information here]

---

**Built with ❤️ for Oregon housing analysis and policy development**

**Current Status**: All foundational data collection phases complete. Housing gap analysis fully implemented with real homeless data integration, enhanced building permits analysis, and comprehensive data quality assessment. Ready for production use and Tableau visualization.
