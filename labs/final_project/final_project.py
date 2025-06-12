import sys
sys.path.append('..')

import pandas as pd
import matplotlib.pyplot as plt
from src.data_input import DataInput
from src.calculations import LCACalculator
from src.visualization import LCAVisualizer

# Initialize data input handler
data_input = DataInput()

# Load product data
product_data = data_input.read_data('data/raw/sample_data.csv')
print("Product Data Shape:", product_data.shape)
print("\nSample of Product Data (Key Columns):")
key_columns = ['product_id', 'product_name', 'life_cycle_stage', 'material_type', 
               'quantity_kg', 'energy_consumption_kwh', 'transport_distance_km',
               'waste_generated_kg', 'recycling_rate', 'landfill_rate', 'incineration_rate']
print(product_data[key_columns].head())

# Load impact factors
impact_factors = data_input.read_impact_factors('data/raw/impact_factors.json')
print("\nAvailable Materials:", list(impact_factors.keys()))

# Initialize calculator
calculator = LCACalculator(impact_factors_path='data/raw/impact_factors.json')

# Calculate impacts
impacts = calculator.calculate_impacts(product_data)
print("\nCalculated Impacts Shape:", impacts.shape)
print("\nCalculated Impacts (Essential Columns):")
# Use the new column names with units.
essential_columns = ['product_id', 'product_name', 'life_cycle_stage', 
                     'carbon_impact (kg CO2e)', 'energy_impact (MJ)', 'water_impact (L)']
print(impacts[essential_columns].head())

# Calculate total impacts
total_impacts = calculator.calculate_total_impacts(impacts)
print("\nTotal Impacts by Product:")
print(total_impacts.head())

visualizer = LCAVisualizer()

# Plot carbon impact breakdown by material type, using the new column name.
fig = visualizer.plot_impact_breakdown(impacts, 'carbon_impact (kg CO2e)', 'life_cycle_stage')
plt.show()

# Plot life cycle impacts for Product1
fig = visualizer.plot_life_cycle_impacts(impacts, 'P001')
plt.show()

# Compare two products
g = visualizer.plot_product_comparison(impacts, ['P010', 'P016'])
plt.show()
                                      
# Plot end-of-life breakdown for Product1
fig = visualizer.plot_end_of_life_breakdown(impacts, 'P005')
plt.show()

# Plot impact correlations
fig = visualizer.plot_impact_correlation(impacts)
plt.show()

# Normalize impacts for comparison
normalized_impacts = calculator.normalize_impacts(impacts)
print("\nNormalized Impacts (showing relative scale 0-1):")
print(normalized_impacts[essential_columns].head())

# Compare alternative products
comparison = calculator.compare_alternatives(impacts, ['P009', 'P014'])

# Display the comparison table
print("\n--- Product Comparison: Relative Performance (%) ---")
pd.set_option('display.precision', 1)
print(comparison.to_string(index=False))

# Compare two materials
material_comparison = calculator.compare_materials(impacts, 'Steel', 'Wood')
print("\nMaterial Comparison (Steel vs Wood):")
print(material_comparison.to_string(index=False))

