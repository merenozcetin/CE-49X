"""
Tests for the visualization module.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization import LCAVisualizer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'product_id': ['P001', 'P001', 'P001', 'P002', 'P002', 'P002'],
        'product_name': ['Product1', 'Product1', 'Product1', 'Product2', 'Product2', 'Product2'],
        'life_cycle_stage': ['Manufacturing', 'Transportation', 'End-of-Life'] * 2,
        'material_type': ['steel', 'steel', 'steel', 'aluminum', 'aluminum', 'aluminum'],
        'quantity_kg': [100, 100, 100, 50, 50, 50],
        'energy_consumption_kwh': [120, 20, 50, 180, 25, 20],
        'transport_distance_km': [50, 100, 30, 180, 140, 35],
        'transport_mode': ['Truck'] * 6,
        'waste_generated_kg': [5, 0, 100, 1, 0, 20],
        'recycling_rate': [0.9, 0, 0.9, 0.85, 0, 0.85],
        'landfill_rate': [0.05, 0, 0.05, 0.1, 0, 0.1],
        'incineration_rate': [0.05, 0, 0.05, 0.05, 0, 0.05],
        'carbon_footprint_kg_co2e': [180, 50, 10, 125, 30, 5],
        'water_usage_liters': [150, 30, 10, 100, 0, 6],
        'carbon_impact (kg CO2e)': [180, 50, 10, 125, 30, 5],
        'energy_impact (MJ)': [120, 20, 50, 180, 25, 20],
        'water_impact (L)': [150, 30, 10, 100, 0, 6],
        'waste_generated (kg)': [5, 0, 100, 1, 0, 20]
    })

def test_plot_impact_breakdown(sample_data):
    """Test impact breakdown plot."""
    visualizer = LCAVisualizer()
    
    # Test by material type
    fig = visualizer.plot_impact_breakdown(sample_data, 'carbon_impact (kg CO2e)', 'material_type')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test by life cycle stage
    fig = visualizer.plot_impact_breakdown(sample_data, 'carbon_impact (kg CO2e)', 'life_cycle_stage')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_life_cycle_impacts(sample_data):
    """Test life cycle impacts plot."""
    visualizer = LCAVisualizer()
    fig = visualizer.plot_life_cycle_impacts(sample_data, 'P001')
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4  # 2x2 subplot layout
    plt.close(fig)

def test_plot_product_comparison(sample_data):
    """Test product comparison plot."""
    visualizer = LCAVisualizer()
    fig = visualizer.plot_product_comparison(sample_data, ['P001', 'P002'])
    
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_end_of_life_breakdown(sample_data):
    """Test end-of-life breakdown plot."""
    visualizer = LCAVisualizer()
    
    # Test with valid end-of-life data
    fig = visualizer.plot_end_of_life_breakdown(sample_data, 'P001')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Test with invalid product ID (should raise ValueError)
    with pytest.raises(ValueError):
        visualizer.plot_end_of_life_breakdown(sample_data, 'INVALID_ID')

def test_plot_impact_correlation(sample_data):
    """Test impact correlation plot."""
    visualizer = LCAVisualizer()
    fig = visualizer.plot_impact_correlation(sample_data)
    
    assert isinstance(fig, plt.Figure)
    
    # Get the axes object
    ax = fig.axes[0]
    
    # Check if the plot has the correct title
    assert ax.get_title() == 'Impact Category Correlations'
    
    # Check if the plot has the correct number of ticks
    assert len(ax.get_xticklabels()) == 4  # Four impact categories
    assert len(ax.get_yticklabels()) == 4  # Four impact categories
    
    # Check if the plot has annotations
    assert len(ax.texts) > 0  # Should have correlation values as annotations
    
    plt.close(fig) 