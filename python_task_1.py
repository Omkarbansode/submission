
import pandas as pd
df=pd.read_csv("C:\\Users\\BusinessComputers.in\\Downloads\\MapUp-Data-Assessment-F-main\\MapUp-Data-Assessment-F-main\\datasets\\dataset-1.csv")
print (df)


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
    
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car')
    return car_matrix
 


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    car_type_counts = df['car'].value_counts().to_dict()
    return car_type_counts
 


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    bus_indexes = df[df['bus'] > 2 * df['bus'].mean()].index.tolist()
    return bus_indexes
  


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    truck_avg_per_route = df.groupby('route')['truck'].mean()
    routes_above_threshold = truck_avg_per_route[truck_avg_per_route > 7].index.tolist()
    return routes_above_threshold




def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
    modified_matrix = matrix.copy()
    # Your custom multiplication logic here
    modified_matrix = modified_matrix * 2
    return modified_matrix
  


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    completeness_check = df.groupby(['id', 'id_2'])['timestamp'].agg(lambda x: x.max() - x.min() == pd.Timedelta(days=7)).reset_index()
    return completeness_check['timestamp']


