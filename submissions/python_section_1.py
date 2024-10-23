from typing import Dict, List
# pip install polyline
import pandas as pd
import re
import polyline
import numpy as np


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    result = [] # we are creating an empty list to store result
    length = len(lst) # Taking length of the input list

    for i in range(0, length, n):
        temp = [] # temprory list to store current elements

        # Collect next n elements, remaining elements
        for j in range(i, min(i+n, length)):
            temp.append(lst[j])

            # manually reversing the elements and add to the result list
        for k in range(len(temp) - 1, -1, -1):
            result.append(temp[k])
    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    dic = {} 

    # Iterate through each string in the input lst
    for string in lst:
        length = len(string) # taking length of the current string

        # If length is not present in the dictionary, then add it with an empty list
        if length not in dic:
            dic[length] = []

        #If it is already present, then append it to the list
        dic[length].append(string)
#sorting the dictinoary by its keys and return as a new dictionay
    return dict(sorted(dic.items()))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    flattened_dict = {}
    
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            # Recursively flatten the nested dictionary
            nested_flattened_dict = flatten_dict(value, sep)
            # Update the flattened dictionary with the nested keys
            for nested_key, nested_value in nested_flattened_dict.items():
                flattened_dict[f"{key}{sep}{nested_key}"] = nested_value
        elif isinstance(value, list):
            # Handle lists by iterating through each element
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    # Recursively flatten the dictionary in the list
                    nested_flattened_dict = flatten_dict(item, sep)
                    # Update the flattened dictionary with the nested keys
                    for nested_key, nested_value in nested_flattened_dict.items():
                        flattened_dict[f"{key}[{index}]{sep}{nested_key}"] = nested_value
                else:
                    # Otherwise, just add the item with the index
                    flattened_dict[f"{key}[{index}]"] = item
        else:
            # For non-dict and non-list values, add to the flattened dictionary
            flattened_dict[key] = value
    
    return flattened_dict


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    def backtrack(start: int):
        # If we've reached the end of the list, we have a complete permutation
        if start == len(nums):
            result.append(nums[:])  # Appending  a copy of the current permutation in result list
            return
        
        for i in range(start, len(nums)):
            # check for duplicates and skip them
            if i > start and nums[i] == nums[start]:
                continue
            
            # Swap the current element with the start element
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)  # Recurse with the next index
            # Backtrack: swap back to the original configuration
            nums[start], nums[i] = nums[i], nums[start]
    
    nums.sort()  # Sort the numbers to handle duplicates
    result = []
    backtrack(0)  # Start the backtracking process
    return result
    pass


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Define regex patterns for the date formats
    patterns = [
        r'\b(\d{2})-(\d{2})-(\d{4})\b',  # dd-mm-yyyy
        r'\b(\d{2})/(\d{2})/(\d{4})\b',  # mm/dd/yyyy
        r'\b(\d{4})\.(\d{2})\.(\d{2})\b'   # yyyy.mm.dd
    ]
    
    # Combine all patterns into one
    combined_pattern = '|'.join(patterns)
    
    # Find all matches in the text
    matches = re.findall(combined_pattern, text)
    # Format matches into the expected output
    valid_dates = []
    for match in matches:
        if match[0] and match[1] and match[2]:  # Check for dd-mm-yyyy
            valid_dates.append(f"{match[0]}-{match[1]}-{match[2]}")
        elif match[3] and match[4] and match[5]:  # Check for mm/dd/yyyy
            valid_dates.append(f"{match[3]}/{match[4]}/{match[5]}")
        elif match[6] and match[7] and match[8]:  # Check for yyyy.mm.dd
            valid_dates.append(f"{match[6]}.{match[7]}.{match[8]}")
    
    return valid_dates
    pass


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth specified in decimal degrees.
    
    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.

    Returns:
        float: Distance between the two points in meters.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of Earth in meters
    r = 6371000
    return c * r

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
     # Decode the polyline string into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)
    
    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Calculate the distance between consecutive points
    distances = [0]  # First distance is 0
    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i-1]
        lat2, lon2 = df.iloc[i]
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)
    
    df['distance'] = distances

    return df

## Not getting specified output, will do later
def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated[j][n - 1 - i] = matrix[i][j]
    
    # Step 2: Create a new matrix for the transformed result
    transformed = [[0] * n for _ in range(n)]
    # Step 3: Calculate the transformed values
    for i in range(n):
        for j in range(n):
            # Calculate the sum of original row and column indices
            original_row_sum = i
            original_col_sum = j
            index_sum = original_row_sum + original_col_sum
            
            # Multiply the element in the rotated matrix by the index sum
            transformed[i][j] = rotated[i][j] * index_sum
    
    return transformed
 


def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Map weekday names to numbers
    weekday_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }

    # Convert startDay to day of week
    df['start_day'] = df['startDay'].map(weekday_map)

    # Convert timestamp columns to datetime format
    df['start_time'] = pd.to_datetime(df['start_day'].astype(str) + ' ' + df['startTime'], format='%w %H:%M:%S')

    # Convert endDay to day of week
    df['end_day'] = df['endDay'].map(weekday_map)

    # Convert timestamp columns to datetime format
    df['end_time'] = pd.to_datetime(df['end_day'].astype(str) + ' ' + df['endTime'], format='%w %H:%M:%S')

    # Group by (id, id_2)
    grouped = df.groupby(['id', 'id_2'])

    # Function to check completeness for each group
    def check_completeness(group):
        hours_covered = set(group['start_time'].dt.hour.unique())
        days_covered = set(group['start_time'].dt.dayofweek.unique())
        
        # Check if all hours (0-23) and days (0-6) are present
        all_hours = set(range(24))
        all_days = set(range(7))
        
        return not (all_hours.issubset(hours_covered) and all_days.issubset(days_covered))

    # Apply the completeness check and return a boolean series with multi-index
    result = grouped.apply(check_completeness)

    return result

# Load your dataset
df = pd.read_csv('datasets/dataset-1.csv')
