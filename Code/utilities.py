import pandas as pd
import json
import numpy as np

def read_signal(SIGNAL_PATH: str, signal_name: str, sep: str = ';') -> pd.DataFrame:
    """
    Read a signal from a CSV file, convert it to a DataFrame, and process the signal values.
    
    Parameters:
    - SIGNAL_PATH (str): The path to the CSV file containing the signal.
    - signal_name (str): The column name in the CSV file that contains the signal data.
    - sep (str): The separator used in the CSV file. Defaults to ';'.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the processed signal in a column named 'signal'.
                    Any commas in the signal values are replaced with periods, and the values are converted to float.
    """
    try:
        df = pd.read_csv(SIGNAL_PATH, sep=sep)
        df['signal'] = df[signal_name].str.replace(',', '.').astype(float)
        return df[['signal']]
    except Exception as e:
        print(f"Error reading the signal: {e}")
        return None

def save_cleaned_signal(SIGNAL_PATH: str, signal_name: str, cleaned_signal: pd.DataFrame, sep: str = ';') -> None:
    """
    Save the cleaned signal to a new CSV file, replacing the original signal values in the specified column.
    
    Parameters:
    - SIGNAL_PATH (str): The path to the original CSV file containing the signal.
    - signal_name (str): The column name in the CSV file that contains the signal data to be replaced.
    - cleaned_signal (pd.DataFrame): A DataFrame containing the cleaned signal values.
    - sep (str): The separator used in the CSV file. Defaults to ';'.
    """
    df = pd.read_csv(SIGNAL_PATH, sep=sep)
    df[signal_name] = cleaned_signal['cleaned'].apply(lambda x: str(x).replace('.', ','))
    df.to_csv('cleaned_signal.csv', sep=sep, index=False)

def write_data_to_json(file_path: str, data, signal_name: str) -> None:
    """
    Writes signal data to a JSON file. If the file already exists but is empty or malformed, 
    it creates a new JSON structure. If the file does not exist, it creates it and adds the new signal data.
    
    Parameters:
    - file_path (str): The path to the JSON file.
    - data (pd.Series, pd.DataFrame, or list): The signal data to write.
    - signal_name (str): The name of the signal, which will be used as a key in the JSON file.
    """
    try:
        if isinstance(data, pd.Series):
            data = data.tolist()
        elif isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='list')

        existing_data = {}
        try:
            with open(file_path, 'r') as file:
                file_contents = file.read()
                if file_contents:
                    existing_data = json.loads(file_contents)
        except (FileNotFoundError, json.JSONDecodeError):
            print("Creating a new JSON file or overwriting malformed content.")
        
        existing_data[signal_name] = data
        
        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)
        
        print(f"Data for signal '{signal_name}' has been written to {file_path}")
    except Exception as e:
        print(f"Error writing data to JSON: {e}")

def read_signal_from_json(file_path: str, signal_name: str) -> np.ndarray:
    """
    Reads signal data for a specific signal name from a JSON file.
    Adjusted to handle the signal stored as a dictionary under the 'signal' key.
    
    Parameters:
    - file_path (str): The path to the JSON file.
    - signal_name (str): The name of the signal to read.
    
    Returns:
    - np.ndarray: An array of signal values, or None if the signal or file is not found.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            if signal_name in data and "signal" in data[signal_name]:
                signal_series = pd.Series(data[signal_name]["signal"])

                signal_values = signal_series.to_numpy()
                return signal_values
            else:
                print(f"Signal '{signal_name}' not found in the file.")
                return None
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading from JSON: {e}")
        return None
