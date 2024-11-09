import pandas as pd
import numpy as np

def feature_engineering(df):
    """
    Feature engineering function for yield prediction
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    
    Returns:
    pandas.DataFrame: Processed dataframe with new features
    """
    df = df.copy()
    
    # Temperature features
    df['temp_range_upper'] = df['MaxOfUpperTRange'] - df['MinOfUpperTRange']
    df['temp_range_lower'] = df['MaxOfLowerTRange'] - df['MinOfLowerTRange']
    df['temp_stability_upper'] = df['temp_range_upper'] / (df['AverageOfUpperTRange'] + 1e-6)
    df['temp_stability_lower'] = df['temp_range_lower'] / (df['AverageOfLowerTRange'] + 1e-6)
    df['temp_range_total'] = df['MaxOfUpperTRange'] - df['MinOfLowerTRange']
    df['temp_average_total'] = (df['AverageOfUpperTRange'] + df['AverageOfLowerTRange']) / 2
    
    # Pollinator features
    pollinator_cols = ['honeybee', 'bumbles', 'andrena', 'osmia']
    df['total_pollinators'] = df[pollinator_cols].sum(axis=1)
    df['pollinator_diversity'] = (df[pollinator_cols] > 0).sum(axis=1)
    for col in pollinator_cols:
        df[f'{col}_ratio'] = df[col] / (df['total_pollinators'] + 1e-6)
    
    # Rain and growth features
    df['rain_intensity'] = df['RainingDays'] / (df['AverageRainingDays'] + 1e-6)
    df['temp_rain_interaction'] = df['temp_average_total'] * df['rain_intensity']
    df['growing_condition_index'] = (df['temp_average_total'] * df['total_pollinators'] * 
                                     (1 + df['rain_intensity']))
    
    # Yield-related features
    df['fruit_efficiency'] = df['fruitmass'] / (df['fruitset'] + 1e-6)
    df['seed_density'] = df['seeds'] / (df['fruitmass'] + 1e-6)
    
    # Clone interactions
    df['clone_pollinator_interaction'] = df['clonesize'] * df['total_pollinators']
    df['clone_temp_interaction'] = df['clonesize'] * df['temp_average_total']
    
    return df