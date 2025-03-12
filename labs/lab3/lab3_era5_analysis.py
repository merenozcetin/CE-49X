import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from windrose import WindroseAxes


def load_data(file_path):
    """
    Load the dataset from a CSV file.
    
    Parameters:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame, or None if the file is not found.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        print("Data loaded successfully.")
        print(df.info())
        return df
    except FileNotFoundError:
        print(f"Error: File not found. Ensure the file exists at the specified path: {file_path}")
        return None

def clean_data(df):
    
    # Replace non existent numbers with means
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df = df.round(1)
    return df

def compute_statistics(df, column):
    """
    Compute and print descriptive statistics for the specified column.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column for which to compute statistics.
    """
    min_val = df[column].min()
    max_val = df[column].max()
    mean_val = df[column].mean()
    median_val = df[column].median()
    std_val = df[column].std()
    
    print(f"\nDescriptive statistics for '{column}':")
    print(f"  Minimum: {min_val}")
    print(f"  Maximum: {max_val}")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  Standard Deviation: {std_val:.2f}")

def calculate_wind_speed(df):
    if 'u10m' in df.columns and 'v10m' in df.columns:
        return np.sqrt(df['u10m']**2 + df['v10m']**2)
        
    else:
        raise ValueError("Columns 'u10m' or 'v10m' not found in the dataset.")

def assign_season(month):
    """Assigns seasons based on month."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


def compute_aggregates(df):

    # Calculate wind speed
    df['wind_speed'] = calculate_wind_speed(df)

    # Extract month and season
    df['month'] = df['timestamp'].dt.month
    df['season'] = df['month'].apply(assign_season)

    # Compute monthly averages
    monthly_avg = df.groupby('month')[['wind_speed']].mean()

    # Compute seasonal averages
    seasonal_avg = df.groupby('season')[['wind_speed']].mean()

    return monthly_avg, seasonal_avg

def identify_extreme_weather(df, threshold=95):
    """
    Identify extreme weather days based on wind speed.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'time', 'u10m', 'v10m'.
        threshold (int): Percentile threshold for extreme wind speed (default=95).

    Returns:
        pd.DataFrame: Extreme weather days with high wind speeds.
    """
    df['wind_speed'] = calculate_wind_speed(df)

    # Find top 5 highest wind speed days
    top_wind_days = df.nlargest(5, 'wind_speed')

    # Define extreme threshold (e.g., 95th percentile)
    extreme_threshold = np.percentile(df['wind_speed'], threshold)
    extreme_days = df[df['wind_speed'] > extreme_threshold]

    return top_wind_days, extreme_days

def calculate_diurnal_pattern(df):
    """
    Calculate average wind speed for each hour of the day.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'time', 'u10m', and 'v10m'.

    Returns:
        pd.DataFrame: Average wind speed per hour.
    """
    df['wind_speed'] = calculate_wind_speed(df)
    df['hour'] = df['timestamp'].dt.hour  # Extract hour of the day

    # Compute average wind speed per hour
    diurnal_pattern = df.groupby('hour')['wind_speed'].mean().reset_index()

    return diurnal_pattern

def calculate_monthly_avg(df):
    df['wind_speed'] = calculate_wind_speed(df)
    df['month'] = df['timestamp'].dt.to_period('M')  # Convert to monthly periods

    return df.groupby('month')['wind_speed'].mean().reset_index()
    

def add_season_column(df):
    df['season'] = df['timestamp'].dt.month % 12 // 3 + 1  # Assign seasons (1=Winter, 2=Spring, 3=Summer, 4=Fall)
    season_labels = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    df['season'] = df['season'].map(season_labels)
    return df

def calculate_wind_direction(df):
    """Compute wind direction from u10m and v10m."""
    return (np.arctan2(df['v10m'], df['u10m']) * (180/np.pi)) % 360

def plot_wind_rose(df):
    """Plot Wind Rose Diagram using wind speed and calculated wind direction."""
    ax = WindroseAxes.from_ax()
    ax.bar(df['wind_direction'], df['wind_speed'], normed=True, opening=0.8, edgecolor='black')
    ax.set_legend()
    plt.title("Wind Rose Diagram")
    plt.show()

def main():
    # Update the file path as needed (relative to your script's location)
    file_path1 = '../../datasets/berlin_era5_wind_20241231_20241231.csv'
    file_path2= '../../datasets/munich_era5_wind_20241231_20241231.csv'
    
    # Load the dataset
    df1 = load_data(file_path1)
    df2 = load_data(file_path2)
    if df1 is None or df2 is None:
        return
        
    df1cleaned = clean_data(df1)
    df2cleaned = clean_data(df2)

    
    #opens a new column in the dataset called wind_speed
    df1cleaned['wind_speed'] = calculate_wind_speed(df1cleaned)
    df2cleaned['wind_speed'] = calculate_wind_speed(df2cleaned)

    monthly_avg_berlin, seasonal_avg_berlin = compute_aggregates(df1cleaned)
    monthly_avg_munich, seasonal_avg_munich = compute_aggregates(df2cleaned)

    # Identify extreme weather conditions
    top_wind_berlin, extreme_days_berlin = identify_extreme_weather(df1cleaned)
    top_wind_munich, extreme_days_munich = identify_extreme_weather(df2cleaned)

    # Calculate diurnal wind speed patterns
    diurnal_berlin = calculate_diurnal_pattern(df1cleaned)
    diurnal_munich = calculate_diurnal_pattern(df2cleaned)

    monthly_berlin = calculate_monthly_avg(df1cleaned)
    monthly_munich = calculate_monthly_avg(df2cleaned)

    df_berlin = add_season_column(df1cleaned)
    df_munich = add_season_column(df2cleaned)





  
    print("\n----------------- Berlin Data Stats -----------------\n")
    compute_statistics(df1cleaned, 'u10m')
    compute_statistics(df1cleaned, 'v10m')
    print("\nBerlin Monthly Averages:\n", monthly_avg_berlin)
    print("\nBerlin Seasonal Averages:\n", seasonal_avg_berlin)
    print("\nBerlin Diurnal Wind Speed Pattern:\n", diurnal_berlin)
    print("\nTop 5 Windy Days in Berlin:\n", top_wind_berlin)
    print("\nExtreme Wind Speed Days in Berlin:\n", extreme_days_berlin)
    print("\n----------------- Munich Data Stats -----------------\n")
    compute_statistics(df2cleaned, 'u10m')
    compute_statistics(df2cleaned, 'v10m')
    print("\nMunich Monthly Averages:\n", monthly_avg_munich)
    print("\nMunich Seasonal Averages:\n", seasonal_avg_munich)
    print("\nMunich Diurnal Wind Speed Pattern:\n", diurnal_munich)
    print("\nTop 5 Windy Days in Munich:\n", top_wind_munich)
    print("\nExtreme Wind Speed Days in Munich:\n", extreme_days_munich)
        
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_berlin['month'].astype(str), monthly_berlin['wind_speed'], label="Berlin", marker='o')
    plt.plot(monthly_munich['month'].astype(str), monthly_munich['wind_speed'], label="Munich", marker='s')

    plt.xlabel("Month")
    plt.ylabel("Average Wind Speed (m/s)")
    plt.title("Monthly Average Wind Speed for Berlin & Munich")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()



    # Combine data
    df_berlin['city'] = 'Berlin'
    df_munich['city'] = 'Munich'
    df_combined = pd.concat([df_berlin, df_munich])

    # Plot
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_combined, x='season', y='wind_speed', hue='city', errorbar=None)

    plt.xlabel("Season")
    plt.ylabel("Average Wind Speed (m/s)")
    plt.title("Seasonal Wind Speed Comparison (Berlin vs Munich)")
    plt.legend()
    plt.grid(True)
    plt.show()


    df1cleaned['wind_direction'] = calculate_wind_direction(df1cleaned)
    df2cleaned['wind_direction'] = calculate_wind_direction(df2cleaned)

    # Plot Wind Rose Diagrams for Both Cities
    plot_wind_rose(df1cleaned)
    plot_wind_rose(df2cleaned)

    



    
    
if __name__ == '__main__':
    main()

