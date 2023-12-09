import pandas as pd
import numpy as np
df=pd.read_csv("C:\\Users\\BusinessComputers.in\\Downloads\\MapUp-Data-Assessment-F-main\\MapUp-Data-Assessment-F-main\\datasets\\dataset-3.csv")
print (df)
df1=pd.read_csv("C:\\Users\\BusinessComputers.in\\Downloads\\MapUp-Data-Assessment-F-main\\MapUp-Data-Assessment-F-main\\datasets\\dataset-2.csv")
print (df1)

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here

    # Extract the coordinates from the DataFrame
    coordinates = df[['id_start', 'id_end']].values

    # Calculate pairwise Euclidean distances using NumPy
    distances = np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=-1)

    # Create a DataFrame with distances and set row and column indices
    distance_matrix = pd.DataFrame(distances, index=df['id'], columns=df['id'])

    return distance_matrix

def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    
    # Get the row and column indices from the unroll_distance_matrix
    ids = unroll_distance_matrix.index

    # Initialize lists to store data for the unrolled DataFrame
    id_starts = []
    id_ends = []
    distances = []

    # Iterate through the unroll_distance_matrix and extract information
    for i in ids:
        for j in ids:
            if i != j:
                id_starts.append(i)
                id_ends.append(j)
                distances.append(unroll_distance_matrix.at[i, j])

    # Create the unrolled DataFrame
    df = pd.DataFrame({'id_start': id_starts, 'id_end': id_ends, 'distance': distances})

    return df



def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
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
    # Calculate average distances for each ID
    avg_distances = df.groupby('id')['distance'].mean()

    # Calculate the reference ID's average distance
    reference_avg_distance = avg_distances[reference_id]

    # Find IDs within 10% threshold
    threshold_percentage = 0.10
    lower_threshold = reference_avg_distance * (1 - threshold_percentage)
    upper_threshold = reference_avg_distance * (1 + threshold_percentage)

    # Filter IDs based on the threshold
    selected_ids = avg_distances[(avg_distances >= lower_threshold) & (avg_distances <= upper_threshold)].index

    # Filter the original DataFrame based on selected IDs
    result_df = df[df['id'].isin(selected_ids)]

    return result_df



def calculate_toll_rate(df):
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): Unrolled DataFrame containing columns 'id_start', 'id_end', 'distance', 'vehicle_type'.

    Returns:
        pandas.DataFrame: DataFrame with toll rates
    """
    # Example toll rates for different vehicle types
    toll_rate_mapping = {
        'car': 0.10,   # Example toll rate for cars
        'truck': 0.15, # Example toll rate for trucks
        # Add more vehicle types and corresponding rates as needed
    }

    # Apply toll rates based on vehicle type
    df['toll_rate'] = df['vehicle_type'].map(toll_rate_mapping)

    # Calculate toll amount based on distance and toll rate
    df['toll_amount'] = df['distance'] * df['toll_rate']

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
    

    # Example time-based toll rates for different intervals
    time_interval_rates = {
        (0, 6): 0.08,    # Example rate for midnight to 6 AM
        (6, 12): 0.10,   # Example rate for 6 AM to 12 PM
        (12, 18): 0.12,  # Example rate for 12 PM to 6 PM
        (18, 24): 0.15   # Example rate for 6 PM to midnight
    }

    # Convert timestamp to hour of the day
    df['hour'] = df['timestamp'].dt.hour

    # Function to map hour to the corresponding time interval rate
    def get_time_interval_rate(hour):
        for interval, rate in time_interval_rates.items():
            if interval[0] <= hour < interval[1]:
                return rate
        return 0.0  # Default rate if no interval matches

    # Apply time-based toll rates
    df['time_interval_rate'] = df['hour'].apply(get_time_interval_rate)

    # Calculate toll amount based on distance and time-based toll rate
    df['time_based_toll_amount'] = df['distance'] * df['time_interval_rate']

    # Drop temporary 'hour' column
    df = df.drop(columns=['hour'])

    return df

