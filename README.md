# Oregon Housing Analysis

A comprehensive data analysis system for understanding housing gaps, homelessness, and housing affordability across all 36 Oregon counties.

## 🏗️ Project Overview

This project implements a professional-grade data architecture for housing analysis, including:

- **Population data collection** from Census decennial and ACS estimates (2009-2023) ✅
- **Housing supply analysis** including units, construction, permits, and vacancy rates ✅
- **Income and affordability data** covering household income, poverty rates, and cost burden ✅
- **Multi-dimensional gap analysis** identifying supply, affordability, quality, and homeless gaps 🔄

## 📊 Key Features

- **Comprehensive Coverage**: All 36 Oregon counties with historical data (2009-2023)
- **Data Quality Framework**: Professional assessment of completeness, accuracy, timeliness, and consistency
- **Multiple Data Sources**: Census ACS, Decennial Census, with framework for HUD PIT and building permits
- **Production-Ready Architecture**: Async support, error handling, logging, and monitoring
- **Tableau Integration**: CSV output format optimized for data visualization

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Census API access (free)
- Required packages: `pandas`, `requests`, `aiohttp`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/oregon_housing.git
cd oregon_housing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the data collection programs:
```bash
# Population data collection (Phase 1)
python3 county_info/population_collector.py

# Housing supply data collection (Phase 2)
python3 county_info/housing_supply_collector.py

# Income data collection (Phase 3)
python3 county_info/income_collector.py
```

## 📁 Project Structure

```
oregon_housing/
├── county_info/                                                # Main project directory
│   ├── data_architecture.py                                    # Core data models and quality framework
│   ├── housing_gap_analyzer.py                                 # Gap analysis engine (Phase 4)
│   ├── housing_supply_collector.py                             # Housing supply data collection ✅
│   ├── population_collector.py                                 # Population data collection ✅
│   ├── income_collector.py                                     # Income data collection ✅
│   ├── IMPLEMENTATION_GUIDE.md                                 # Detailed implementation documentation
│   ├── project_synopsis.txt                                    # Project overview and scope
│   ├── requirements.txt                                        # Python dependencies
│   └── resources.txt                                           # Data source references
├── Data_Collection_Output/                                     # Output data and logs
│   ├── oregon_county_population_2009_2023_census_acs.csv
│   ├── oregon_county_housing_supply_2009_2023_acs.csv
│   ├── oregon_county_income_2009_2023_acs.csv
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

### Housing Gap Analysis 🔄
- Supply gaps (units needed vs. available)
- Affordability gaps (households unable to afford)
- Quality gaps (substandard housing)
- Homeless gaps (actual homeless counts)
- **Status**: Framework ready for implementation

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

## 📊 Output & Reporting

- **CSV Data Files**: Structured data for analysis and visualization
- **Quality Reports**: Comprehensive data quality assessments
- **Collection Logs**: Detailed API call and processing logs
- **Performance Metrics**: Collection timing and API statistics

## 🚧 Development Status

### ✅ **COMPLETED Phases:**
- **Phase 1**: Population Data Collection - 576 records collected
- **Phase 2**: Housing Supply Data Collection - 540 records collected
- **Phase 3**: Income Data Collection - 540 records collected

### 🔄 **READY FOR IMPLEMENTATION:**
- **Phase 4**: Housing Gap Analysis - All data foundation complete
- **Phase 5**: Advanced Analytics - Framework ready

### 📊 **Data Collection Summary:**
- **Total Records**: 1,656 records across all three datasets
- **Geographic Coverage**: 100% of Oregon counties (36/36)
- **Temporal Coverage**: 2009-2023 (15 years)
- **Data Quality**: Good to Excellent across all datasets
- **API Success Rate**: 100% (no errors in recent collections)

## 🤝 Contributing

This project is designed for:
- Data scientists and analysts
- Housing policy researchers
- Government agencies and nonprofits
- Academic researchers
- Community organizations

## 📚 Documentation

- **Implementation Guide**: Detailed technical documentation
- **Project Synopsis**: High-level project overview and current status
- **Code Comments**: Comprehensive inline documentation
- **Data Schemas**: Detailed data structure definitions

## 📄 License

[Add your license information here]

## 📞 Contact

[Add your contact information here]

---

**Built with ❤️ for Oregon housing analysis and policy development**

**Current Status**: All foundational data collection phases complete. Ready for comprehensive housing gap analysis across Oregon's 36 counties.
