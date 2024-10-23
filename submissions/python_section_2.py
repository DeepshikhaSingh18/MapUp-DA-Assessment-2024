import pandas as pd
import numpy as np

from datetime import datetime, timedelta


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    # Create a pivot table to represent the distance matrix
    distance_matrix = df.pivot(index='id_start', columns='id_end', values='distance').fillna(0)

    # Ensure symmetry of the matrix
    distance_matrix = distance_matrix + distance_matrix.transpose()

    # Set diagonal values to 0
    distance_matrix.values[[range(len(distance_matrix))]*2] = 0

    # Calculate cumulative distances along known routes
    for col in distance_matrix.columns:
        for row in distance_matrix.index:
            if distance_matrix.at[row, col] == 0 and row != col:
                # Find known routes from the starting location
                known_routes = distance_matrix.loc[row, distance_matrix.loc[row] != 0]
                # Calculate cumulative distance for unknown routes
                distance_matrix.at[row, col] = known_routes.sum()

    return distance_matrix
df = pd.read_csv('datasets/dataset-2.csv')


def unroll_distance_matrix(distance_matrix)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    lower_triangle = distance_matrix.where(np.tril(np.ones(distance_matrix.shape), k=-1).astype(bool))

    unrolled_series = lower_triangle.stack()
    unrolled_df = unrolled_series.reset_index()

    # Rename columns
    unrolled_df.columns = ['id_start', 'id_end', 'distance']

    return unrolled_df


def find_ids_within_ten_percentage_threshold(result_unrolled_df, reference_value)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    reference_rows = result_unrolled_df[result_unrolled_df['id_start'] == reference_value]
    average_distance = reference_rows['distance'].mean()

    lower_threshold = average_distance * 0.9
    upper_threshold = average_distance * 1.1


    within_threshold_rows = result_unrolled_df[(result_unrolled_df['distance'] >= lower_threshold) & (result_unrolled_df['distance'] <= upper_threshold)]

    result_list = sorted(within_threshold_rows['id_start'].unique())

    return result_list



def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    return df

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    time_ranges_weekdays = [
        {'start_time': datetime.strptime('00:00:00', '%H:%M:%S').time(), 'end_time': datetime.strptime('10:00:00', '%H:%M:%S').time(), 'discount_factor': 0.8},
        {'start_time': datetime.strptime('10:00:00', '%H:%M:%S').time(), 'end_time': datetime.strptime('18:00:00', '%H:%M:%S').time(), 'discount_factor': 1.2},
        {'start_time': datetime.strptime('18:00:00', '%H:%M:%S').time(), 'end_time': datetime.strptime('23:59:59', '%H:%M:%S').time(), 'discount_factor': 0.8}
    ]

    time_ranges_weekends = [
        {'start_time': datetime.strptime('00:00:00', '%H:%M:%S').time(), 'end_time': datetime.strptime('23:59:59', '%H:%M:%S').time(), 'discount_factor': 0.7}
    ]
    

def calculate_discount_factor(row):
    print("temp \n", row)
    current_time = row['start_time']

    # Check if it's a weekday or weekend
    if row['start_day'] in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        time_ranges = time_ranges_weekdays
    else:
        time_ranges = time_ranges_weekends

    for time_range in time_ranges:
        if time_range['start_time'] <= current_time <= time_range['end_time']:
            return time_range['discount_factor']

    df['discount_factor'] = df.apply(calculate_discount_factor, axis=1)

    vehicle_columns = ['moto', 'car', 'rv', 'bus', 'truck']
    for vehicle_column in vehicle_columns:
        df[vehicle_column] = df[vehicle_column] * df['discount_factor']

    df = df.drop(columns=['discount_factor'])

    return df
