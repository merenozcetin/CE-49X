"""
Visualization module for LCA tool.
Handles creation of plots and charts for impact analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

class LCAVisualizer:
    def __init__(self):
        plt.style.use('bmh')
        # Custom color palette for general use (e.g., pie charts, bar charts)
        self.colors = [
            '#1A313B', '#155E75', '#4BA07E', '#30B0A1','#6EE7B7', '#FAA255', '#EC613F'
        ]
        
        # Define specific, consistent colors for charts that need them.
        self.eol_plot_colors = {
            'recycling': '#6EE7B7',   # Lightest Green
            'landfill': '#1A313B',    # Darkest
            'incineration': '#EC613F' # Red/Orange
        }
        self.correlation_cmap = LinearSegmentedColormap.from_list(
            'custom_diverging',
            ['#155E75', 'white', '#EC613F'], N=256 # Cool (blue/teal) -> White -> Warm (red/orange)
        )
        
        # The keys here now match the new column names from the calculator.
        self.impact_labels = {
            'carbon_impact (kg CO2e)': 'Carbon Impact (kg CO2e)',
            'energy_impact (MJ)': 'Energy Impact (MJ)',
            'water_impact (L)': 'Water Impact (L)',
            'waste_generated (kg)': 'Waste Generated (kg)'
        }
    
    def plot_impact_breakdown(self, data: pd.DataFrame, impact_type: str, 
                            group_by: str = 'material_type',
                            title: Optional[str] = None) -> plt.Figure:
        """
        Create a pie chart showing impact breakdown by specified grouping.
        
        Args:
            data: DataFrame with impact data
            impact_type: Type of impact to plot (e.g., 'carbon_impact')
            group_by: Column to group by ('material_type' or 'life_cycle_stage')
            title: Optional title for the plot
            
        Returns:
            matplotlib Figure object
        """
        # Diagnostic print to confirm the function is receiving the correct inputs.
        

        fig, ax = plt.subplots(figsize=(12, 8))
        
        impact_data = data.groupby(group_by)[impact_type].sum()
        
        # Sort the data from largest to smallest to ensure darkest color goes to largest slice.
        impact_data = impact_data.sort_values(ascending=False)
        
        # Group slices smaller than 2% into an 'Other' category to prevent clutter.
        threshold = 2.0
        total = impact_data.sum()
        small_slices = impact_data[impact_data / total * 100 < threshold]
        
        if not small_slices.empty and len(small_slices) > 1:
            other_sum = small_slices.sum()
            impact_data = impact_data[impact_data / total * 100 >= threshold]
            impact_data['Other'] = other_sum

        wedges, texts, autotexts = ax.pie(
            impact_data, 
            autopct='%1.1f%%',
            colors=self.colors, # Use the custom color palette.
            pctdistance=0.85,
            startangle=90
        )
        
        ax.legend(wedges, impact_data.index,
                  title=group_by.replace("_", " ").title(),
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize='medium')

        plt.setp(autotexts, size=10, weight="bold", color="white")

        if title:
            ax.set_title(title, fontsize=16, weight='bold')
        else:
            ax.set_title(f'{self.impact_labels[impact_type]} by {group_by.replace("_", " ").title()}', fontsize=16, weight='bold')
            
        ax.axis('equal')  
        plt.tight_layout()
        return fig
    
    def plot_life_cycle_impacts(self, data: pd.DataFrame, 
                              product_id: str) -> plt.Figure:
        """
        Create bar charts showing impacts across life cycle stages for a product.
        
        Args:
            data: DataFrame with impact data
            product_id: Product ID to analyze
            
        Returns:
            matplotlib Figure object
        """
        product_data = data[data['product_id'] == product_id]
        # Get the product name for the main title.
        product_name = product_data['product_name'].iloc[0]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Define consistent colors for each stage.
        stage_colors = {
            'manufacturing': self.colors[0], # Darkest
            'transportation': self.colors[2], # Mid-tone
            'end-of-life': self.colors[4]   # Lighter
        }

        fig.suptitle(f'Life Cycle Impact Analysis for: {product_name}', fontsize=20, weight='bold')

        impact_types = ['carbon_impact (kg CO2e)', 'energy_impact (MJ)', 'water_impact (L)', 'waste_generated (kg)']
        
        for idx, impact_type in enumerate(impact_types):
            ax = axes[idx]
            
            # Group data by stage and sort by impact value.
            stage_data = product_data.groupby('life_cycle_stage')[impact_type].sum().sort_values(ascending=False)
            
            # Map the stage names to our defined colors.
            bar_colors = stage_data.index.map(lambda s: stage_colors.get(s.lower(), self.colors[5]))

            ax.bar(stage_data.index, stage_data.values, color=bar_colors)
            
            # Add data labels on top of each bar for clarity.
            for bar in ax.patches:
                yval = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:,.0f}', 
                    va='bottom', ha='center', fontsize=11, weight='bold', color='#333'
                )

            ax.set_title(self.impact_labels[impact_type], fontsize=14)
            ax.tick_params(axis='x', rotation=0, labelsize=12) # Use horizontal labels.
            ax.set_ylim(0, ax.get_ylim()[1] * 1.15) # Add space for labels.

            # Clean up the axes for a more modern look.
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('') # Remove redundant x-axis labels.
            
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for main title.
        return fig
    
    def plot_product_comparison(self, data: pd.DataFrame, 
                              product_ids: List[str]) -> plt.Figure:
        """
        Create a grouped bar chart comparing multiple products across impact categories.
        
        Args:
            data: DataFrame with impact data
            product_ids: List of product IDs to compare
            
        Returns:
            matplotlib Figure object
        """
        # Aggregate the total impacts for the selected products.
        total_impacts = data[data['product_id'].isin(product_ids)].groupby(['product_id', 'product_name']).agg({
            'carbon_impact (kg CO2e)': 'sum', 'energy_impact (MJ)': 'sum',
            'water_impact (L)': 'sum', 'waste_generated (kg)': 'sum'
        }).reset_index()

        # Melt the DataFrame to prepare it for grouped plotting.
        melted_data = total_impacts.melt(
            id_vars=['product_id', 'product_name'], 
            value_vars=['carbon_impact (kg CO2e)', 'energy_impact (MJ)', 'water_impact (L)', 'waste_generated (kg)'],
            var_name='impact_type', 
            value_name='value'
        )
        
        # Replace the technical impact names with clean, readable labels.
        melted_data['impact_type'] = melted_data['impact_type'].map(self.impact_labels)

        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create the grouped bar plot using seaborn for a clean aesthetic.
        sns.barplot(
            x='impact_type', 
            y='value', 
            hue='product_name', 
            data=melted_data,
            ax=ax,
            palette=[self.colors[0], self.colors[4]] # Use the first and fifth colors.
        )

        # Improve the chart's titles and labels for clarity.
        ax.set_title('Product Environmental Impact Comparison', fontsize=18, weight='bold')
        ax.set_xlabel('Environmental Impact Category', fontsize=12)
        ax.set_ylabel('Total Impact Value', fontsize=12)
        ax.tick_params(axis='x', rotation=0)
        
        # Clean up the visual style.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(title='Product', fontsize='medium')
        
        plt.tight_layout()
        return fig
    
    def plot_end_of_life_breakdown(self, data: pd.DataFrame, 
                                 product_id: str) -> plt.Figure:
        """
        Create a stacked bar chart showing end-of-life management breakdown.
        
        Args:
            data: DataFrame with impact data
            product_id: Product ID to analyze
            
        Returns:
            matplotlib Figure object
        """
        # Filter for end-of-life stage only
        product_data = data[(data['product_id'] == product_id) & 
                          (data['life_cycle_stage'].str.lower() == 'end-of-life')]
        
        if product_data.empty:
            raise ValueError(f"No end-of-life data found for product {product_id}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        eol_data = product_data[['recycling_rate', 'landfill_rate', 'incineration_rate']]
        
        # Use consistent colors for waste management methods
        plot_colors = [self.colors[4], self.colors[0], self.colors[6]]  # Green, Dark, Red/Orange

        eol_data.plot(kind='bar', stacked=True, ax=ax, color=plot_colors)
        
        # Get product name for title
        product_name = product_data['product_name'].iloc[0]
        ax.set_title(f'End-of-Life Management for {product_name}', fontsize=14, weight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Rate', fontsize=12)
        ax.set_ylim(0, 1)
        
        # Remove x-axis ticks since we only have one bar
        ax.set_xticks([])
        
        # Add legend with better labels
        ax.legend(['Recycling', 'Landfill', 'Incineration'], 
                 title='Waste Management Method',
                 loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def plot_impact_correlation(self, data: pd.DataFrame) -> plt.Figure:
        """
        Create a correlation heatmap of different impact categories.
        
        Args:
            data: DataFrame with impact data
            
        Returns:
            matplotlib Figure object
        """
        impact_columns = ['carbon_impact (kg CO2e)', 'energy_impact (MJ)', 'water_impact (L)', 'waste_generated (kg)']
        correlation = data[impact_columns].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use the dedicated, consistent colormap for the heatmap.
        sns.heatmap(correlation, annot=True, cmap=self.correlation_cmap, center=0, ax=ax,
                   annot_kws={'size': 10}, fmt='.2f')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
        ax.set_title('Impact Category Correlations', pad=20, fontsize=12)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        return fig 