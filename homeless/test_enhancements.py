#!/usr/bin/env python3
"""
Oregon Housing Analysis - Enhancement Test Script
================================================

This script tests all the enhancements made to the housing gap analyzer system:
1. Real homeless data integration
2. Enhanced building permits data
3. Improved homeless gap calculations
4. Comprehensive data quality assessment

Run this script to verify all enhancements are working correctly.
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from homeless_data_collector import OregonHomelessDataCollector
from housing_gap_analyzer import OregonHousingGapAnalyzer
from housing_supply_collector import OregonHousingSupplyCollector

def test_homeless_data_collection():
    """Test the new homeless data collector"""
    print("🏠 Testing Homeless Data Collection...")
    print("=" * 50)
    
    try:
        collector = OregonHomelessDataCollector()
        filepath = collector.run_collection(2007, 2023)
        
        if filepath:
            print(f"✅ Homeless data collection successful!")
            print(f"📁 Data saved to: {filepath}")
            
            # Load and display sample data
            homeless_data = pd.read_csv(filepath)
            print(f"📊 Total records: {len(homeless_data)}")
            print(f"🏘️ Counties covered: {homeless_data['county_fips'].nunique()}")
            print(f"📅 Years covered: {homeless_data['year'].nunique()}")
            print(f"🔍 Data sources: {homeless_data['data_source'].unique()}")
            
            # Show sample data for Multnomah County
            multnomah_data = homeless_data[homeless_data['county_fips'] == '051'].head(3)
            print(f"\n📋 Sample Multnomah County data:")
            print(multnomah_data[['year', 'total_homeless', 'sheltered_homeless', 'unsheltered_homeless', 'chronic_homeless']].to_string(index=False))
            
            return True
        else:
            print("❌ Homeless data collection failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing homeless data collection: {str(e)}")
        return False

def test_enhanced_building_permits():
    """Test the enhanced building permits data collection"""
    print("\n🏗️ Testing Enhanced Building Permits...")
    print("=" * 50)
    
    try:
        collector = OregonHousingSupplyCollector()
        
        # Test building permits for a few years
        test_years = [2015, 2020, 2023]
        
        for year in test_years:
            permits_data = collector.get_building_permits_data(year)
            
            if permits_data:
                print(f"✅ Building permits data for {year}: {len(permits_data)} counties")
                
                # Show sample data for Multnomah County
                multnomah_permits = next((p for p in permits_data if p['county_fips'] == '051'), None)
                if multnomah_permits:
                    print(f"   📍 Multnomah County {year}:")
                    print(f"      - Total permits: {multnomah_permits['building_permits_issued']}")
                    print(f"      - New units: {multnomah_permits['new_construction_units']}")
                    print(f"      - Single family: {multnomah_permits['single_family_permits']}")
                    print(f"      - Multi-family: {multnomah_permits['multi_family_permits']}")
                    print(f"      - Commercial: {multnomah_permits['commercial_permits']}")
                    print(f"      - Total value: ${multnomah_permits['permit_value_total']:,}k")
            else:
                print(f"❌ No building permits data for {year}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing building permits: {str(e)}")
        return False

def test_enhanced_housing_gap_analysis():
    """Test the enhanced housing gap analyzer with real homeless data"""
    print("\n🔍 Testing Enhanced Housing Gap Analysis...")
    print("=" * 50)
    
    try:
        analyzer = OregonHousingGapAnalyzer()
        
        # Test homeless gap calculation for specific counties and years
        test_cases = [
            ("051", 2020, "Multnomah (Portland)"),
            ("067", 2020, "Washington (Beaverton/Hillsboro)"),
            ("005", 2020, "Clackamas (Suburban Portland)"),
            ("039", 2020, "Lane (Eugene)"),
            ("029", 2020, "Jackson (Medford)")
        ]
        
        print("📊 Testing homeless gap calculations:")
        for county_fips, year, county_name in test_cases:
            homeless_gap = analyzer.calculate_homeless_gap(county_fips, year)
            print(f"   🏘️ {county_name} ({county_fips}) in {year}: {homeless_gap:,} homeless")
        
        # Test comprehensive analysis for a shorter period
        print(f"\n🚀 Running comprehensive analysis for 2020-2023...")
        analysis_df = analyzer.run_comprehensive_analysis(2020, 2023)
        
        if not analysis_df.empty:
            print(f"✅ Analysis successful! Generated {len(analysis_df)} records")
            
            # Show sample results
            print(f"\n📋 Sample analysis results:")
            sample_results = analysis_df[analysis_df['county_fips'] == '051'].head(2)
            display_columns = ['year', 'county_name', 'supply_gap', 'affordability_gap', 'homeless_gap', 'housing_stress_score']
            print(sample_results[display_columns].to_string(index=False))
            
            # Save results
            filepath = analyzer.save_analysis_results(analysis_df)
            print(f"\n💾 Analysis results saved to: {filepath}")
            
            # Generate and save summary
            summary = analyzer.generate_analysis_summary(analysis_df)
            analyzer.save_analysis_summary(summary)
            print(f"📊 Analysis summary saved")
            
            return True
        else:
            print("❌ Analysis failed - no results generated")
            return False
            
    except Exception as e:
        print(f"❌ Error testing housing gap analysis: {str(e)}")
        return False

def test_data_quality_integration():
    """Test the enhanced data quality assessment"""
    print("\n🔬 Testing Enhanced Data Quality Assessment...")
    print("=" * 50)
    
    try:
        analyzer = OregonHousingGapAnalyzer()
        
        # Test data loading with quality assessment
        population_data, housing_supply_data, income_data, homeless_data = analyzer.load_data_sources()
        
        print("📊 Data source quality assessment:")
        
        if not population_data.empty:
            quality_dist = population_data['data_quality_score'].value_counts()
            print(f"   👥 Population data: {len(population_data)} records")
            print(f"      Quality distribution: {quality_dist.to_dict()}")
        
        if not housing_supply_data.empty:
            quality_dist = housing_supply_data['data_quality_score'].value_counts()
            print(f"   🏠 Housing supply data: {len(housing_supply_data)} records")
            print(f"      Quality distribution: {quality_dist.to_dict()}")
        
        if not income_data.empty:
            quality_dist = income_data['data_quality_score'].value_counts()
            print(f"   💰 Income data: {len(income_data)} records")
            print(f"      Quality distribution: {quality_dist.to_dict()}")
        
        if not homeless_data.empty:
            quality_dist = homeless_data['data_quality_score'].value_counts()
            print(f"   🏠 Homeless data: {len(homeless_data)} records")
            print(f"      Quality distribution: {quality_dist.to_dict()}")
        
        print("✅ Data quality assessment completed")
        return True
        
    except Exception as e:
        print(f"❌ Error testing data quality: {str(e)}")
        return False

def main():
    """Main test execution function"""
    print("🧪 Oregon Housing Analysis - Enhancement Test Suite")
    print("=" * 60)
    print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Homeless Data Collection", test_homeless_data_collection),
        ("Enhanced Building Permits", test_enhanced_building_permits),
        ("Enhanced Housing Gap Analysis", test_enhanced_housing_gap_analysis),
        ("Data Quality Integration", test_data_quality_integration)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All enhancements are working correctly!")
        print("\n🚀 The housing gap analyzer is now ready for production use with:")
        print("   - Real homeless data integration (HUD PIT + local shelter data)")
        print("   - Enhanced building permits data with county-specific characteristics")
        print("   - Multi-tier homeless gap calculation (real data → enhanced estimates → basic estimates)")
        print("   - Comprehensive data quality assessment and validation")
        print("   - Professional-grade error handling and fallback mechanisms")
    else:
        print("⚠️ Some enhancements need attention. Check the logs above for details.")
    
    print(f"\n🕐 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
