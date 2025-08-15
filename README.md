# Oregon Housing Analysis

A comprehensive data analysis system for understanding housing gaps, homelessness, and housing affordability across all 36 Oregon counties.

## 🏗️ Project Overview

This project implements a professional-grade data architecture for housing analysis, including:

- **Population data collection** from Census decennial and ACS estimates (1990-2023)
- **Housing supply analysis** including units, construction, permits, and vacancy rates
- **Housing demand assessment** covering households, income, and affordability metrics
- **Homeless data integration** with shelter capacity and utilization tracking
- **Multi-dimensional gap analysis** identifying supply, affordability, quality, and homeless gaps

## 📊 Key Features

- **Comprehensive Coverage**: All 36 Oregon counties with historical data (1990-2023)
- **Data Quality Framework**: Professional assessment of completeness, accuracy, timeliness, and consistency
- **Multiple Data Sources**: Census, HUD PIT, building permits, income data, and local surveys
- **Production-Ready Architecture**: Async support, error handling, logging, and monitoring
- **Tableau Integration**: CSV output format optimized for data visualization

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Census API access (free)
- Required packages: `pandas`, `requests`, `census`, `matplotlib`

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

3. Run the data architecture demo:
```bash
python homeless/data_architecture.py
```

## 📁 Project Structure

```
oregon_housing/
├── homeless/                          # Main project directory
│   ├── data_architecture.py          # Core data models and quality framework
│   ├── housing_gap_analyzer.py       # Gap analysis engine
│   ├── housing_supply_collector.py   # Housing supply data collection
│   ├── population_collector.py       # Population data collection
│   ├── IMPLEMENTATION_GUIDE.md       # Detailed implementation documentation
│   ├── project_synopsis.txt          # Project overview and scope
│   ├── requirements.txt              # Python dependencies
│   └── resources.txt                 # Data source references
├── Data_Collection/                  # Data collection scripts
├── Data_Collection_Output/           # Output data and logs
│   └── historic_data/               # Historical data and collection logs
└── README.md                         # This file
```

## 🔍 Data Models

### Population Facts
- Total population counts by county and year
- Data source tracking (Census decennial vs. ACS)
- Quality scoring and validation

### Housing Supply Facts
- Housing units (total, occupied, vacant)
- Construction and building permit data
- Vacancy rates and homeownership metrics

### Housing Demand Facts
- Household counts and tenure
- Income levels and affordability metrics
- Cost burden analysis

### Homeless Facts
- Actual homeless counts and types
- Shelter capacity and utilization
- Point-in-time survey data

### Housing Gap Analysis
- Supply gaps (units needed vs. available)
- Affordability gaps (households unable to afford)
- Quality gaps (substandard housing)
- Homeless gaps (actual homeless counts)

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
- **Trend Analysis**: Historical housing market and homelessness patterns
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
- **Analysis Summaries**: Gap statistics and housing stress indicators
- **Collection Logs**: Detailed API call and processing logs
- **Performance Metrics**: Collection timing and API statistics

## 🚧 Development Status

- ✅ **Data Architecture**: Complete and production-ready
- ✅ **Implementation Guide**: Comprehensive documentation
- 🔄 **Data Collection**: Framework ready for implementation
- 🔄 **Gap Analysis**: Engine ready for implementation
- 📋 **Future Enhancements**: HUD PIT integration, building permits, income data

## 🤝 Contributing

This project is designed for:
- Data scientists and analysts
- Housing policy researchers
- Government agencies and nonprofits
- Academic researchers
- Community organizations

## 📚 Documentation

- **Implementation Guide**: Detailed technical documentation
- **Project Synopsis**: High-level project overview
- **Code Comments**: Comprehensive inline documentation
- **Data Schemas**: Detailed data structure definitions

## 📄 License

[Add your license information here]

## 📞 Contact

[Add your contact information here]

---

**Built with ❤️ for Oregon housing analysis and policy development**
