import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing NA/NaN values in the 'signal' column of the DataFrame by propagating the last valid observation forward.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing a 'signal' column with potential NA/NaN values.

    Returns:
    - pd.DataFrame: The DataFrame with missing values in the 'signal' column filled.
    """
    df['signal'] = df['signal'].ffill()
    return df

def calculate_derivative(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate the first numerical derivative of the 'signal' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing a 'signal' column for which the derivative is calculated.

    Returns:
    - np.ndarray: A numpy array containing the calculated derivative of the 'signal' column.
    """
    derivative = np.diff(df['signal'], prepend=np.nan)
    return derivative

def pad_signal(signal: np.ndarray, padding: int) -> np.ndarray:
    """
    Pad the signal array on both sides with the edge values to handle edge cases during processing.

    Parameters:
    - signal (np.ndarray): The input signal array to be padded.
    - padding (int): The number of elements to pad on both sides of the signal array.

    Returns:
    - np.ndarray: The padded signal array.
    """
    signal_padded = np.pad(signal, (padding, padding), 'edge')
    return signal_padded

def interpolate_signal(start_index: int, end_index: int, signal: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Interpolate the signal between two indices using a specified interpolation method.
    
    Parameters:
    - start_index (int): The starting index for interpolation.
    - end_index (int): The ending index for interpolation, exclusive.
    - signal (np.ndarray): The signal array to interpolate.
    - method (str): The method of interpolation (e.g., 'linear', 'nearest', etc.).
    
    Returns:
    - np.ndarray: The interpolated signal values between start_index and end_index.
    """
    x = np.array([start_index, end_index - 1])
    y = signal[x]
    f = interp1d(x, y, kind=method)
    return f(np.arange(start_index, end_index))

def remove_artifacts(signal: np.ndarray, signal_threshold: float = 40) -> pd.DataFrame:
    """
    Remove artifacts from a signal array by setting values above a threshold to NaN and perform interpolation.
    
    Parameters:
    - signal (np.ndarray): The input signal array.
    - signal_threshold (float): The threshold value above which signal values are considered artifacts.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the cleaned signal and a mask indicating interpolated values.
    """
    df = pd.DataFrame({'signal': signal})
    df = fill_missing_values(df)  # Assuming fill_missing_values is defined elsewhere

    below_threshold_mask = df['signal'] <= signal_threshold

    for index in below_threshold_mask[below_threshold_mask].index:
        if index > 0:
            below_threshold_mask.at[index - 1] = True
        if index + 1 < len(below_threshold_mask):
            below_threshold_mask.at[index + 1] = True

    df.loc[~below_threshold_mask, 'signal'] = np.nan
    cleaned_signal = df['signal'].to_numpy()

    interpolated_mask = pd.isnull(df['signal']).to_numpy()

    return pd.DataFrame({'cleaned': cleaned_signal, 'interpolated': interpolated_mask})

def interpolate_mask(interpolated_mask: np.ndarray, cleaned_signal: np.ndarray, additional_interpolation_method: str) -> np.ndarray:
    """
    Apply interpolation to a signal array specifically in regions indicated by an interpolated mask.
    
    Parameters:
    - interpolated_mask (np.ndarray): A boolean array indicating where interpolation should be applied.
    - cleaned_signal (np.ndarray): The signal array to be interpolated.
    - additional_interpolation_method (str): The method of interpolation to be used.
    
    Returns:
    - np.ndarray: The signal array after applying the specified interpolation to the masked regions.
    """
    i = 0
    while i < len(interpolated_mask):
        if interpolated_mask[i]:
            start_index = i
            while i < len(interpolated_mask) and interpolated_mask[i]:
                i += 1
            end_index = i

            interpolated_values = interpolate_signal(start_index, end_index, 
                                                    cleaned_signal, method=additional_interpolation_method)
            cleaned_signal[start_index:end_index] = interpolated_values
        else:
            i += 1

    return cleaned_signal

def clean_signal(signal: np.ndarray, window_size: int = 9, threshold_multiplier: float = 2, interpolation_method: str = 'linear', 
                 additional_interpolation_method: str = 'next') -> pd.DataFrame:
    """
    Clean a signal by interpolating over sections where the derivative exceeds a certain threshold.
    
    Parameters:
    - signal (np.ndarray): The input signal array to be cleaned.
    - window_size (int): The size of the window used to calculate the derivative and identify sections for interpolation.
    - threshold_multiplier (float): Multiplier used with the standard deviation of the derivative to determine the threshold.
    - interpolation_method (str): The primary method of interpolation for initial cleaning.
    - additional_interpolation_method (str): Additional method of interpolation for further cleaning, if needed.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the cleaned signal, a mask indicating interpolated regions, and a mask for NaN values.
    """
    df = pd.DataFrame({'signal': signal})
    df = fill_missing_values(df)  # Assuming fill_missing_values is defined elsewhere

    derivative = calculate_derivative(df)  # Assuming calculate_derivative is defined elsewhere
    threshold = np.nanstd(derivative) * threshold_multiplier

    padding = window_size // 2
    padded_values = pad_signal(df['signal'], padding)  # Assuming pad_signal is defined elsewhere

    cleaned_signal = padded_values.copy()
    interpolated_mask = np.zeros_like(cleaned_signal, dtype=bool)
    nan_mask = np.zeros_like(cleaned_signal, dtype=bool)

    for i in range(padding, len(padded_values) - padding):
        window = derivative[i-padding:i+padding+1]
        if np.max(window) >= threshold or np.min(window) <= -threshold:
            start_index = i - padding
            end_index = i + padding + 1

            interpolated_values = interpolate_signal(start_index, end_index, 
                                                     cleaned_signal, method=interpolation_method)
            cleaned_signal[start_index:end_index] = interpolated_values
            interpolated_mask[start_index:end_index] = True
            nan_mask[start_index:end_index] = True  # Here should be logic to set NaN, currently just setting True for the mask

    if additional_interpolation_method:
        cleaned_signal = interpolate_mask(interpolated_mask, cleaned_signal, 
                                          additional_interpolation_method)

    cleaned_signal = cleaned_signal[padding:-padding]
    interpolated_mask = interpolated_mask[padding:-padding]
    nan_mask = nan_mask[padding:-padding]

    return pd.DataFrame({'cleaned': cleaned_signal, 'interpolated': interpolated_mask, 'nan_values': nan_mask})

def clean_signal_specific_value(signal: np.ndarray, window_size: int = 9, threshold_multiplier: float = 2, threshold_signal_value: float = 0,
                                interpolation_method: str = 'linear', additional_interpolation_method: str = 'next') -> pd.DataFrame:
    """
    Clean a signal by interpolating over sections where the derivative exceeds a certain threshold and
    setting specific values below a threshold to NaN for further processing.
    
    Parameters:
    - signal (np.ndarray): The input signal array to be cleaned.
    - window_size (int): The size of the window used to calculate the derivative and identify sections for interpolation.
    - threshold_multiplier (float): Multiplier used with the standard deviation of the derivative to determine the threshold.
    - threshold_signal_value (float): Specific signal value threshold below which values are set to NaN before cleaning.
    - interpolation_method (str): The primary method of interpolation for initial cleaning.
    - additional_interpolation_method (str): Additional method of interpolation for further cleaning, if needed.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the cleaned signal, a mask indicating interpolated regions, and a mask for NaN values.
    """
    df = pd.DataFrame({'signal': signal})
    df = fill_missing_values(df)  # Assuming fill_missing_values is defined elsewhere

    derivative = calculate_derivative(df)  # Assuming calculate_derivative is defined elsewhere
    threshold = np.nanstd(derivative) * threshold_multiplier

    padding = window_size // 2
    padded_values = pad_signal(df['signal'], padding)  # Assuming pad_signal is defined elsewhere

    cleaned_signal = padded_values.copy()
    interpolated_mask = np.zeros_like(cleaned_signal, dtype=bool)
    nan_mask = np.zeros_like(cleaned_signal, dtype=bool)

    for i in range(padding, len(padded_values) - padding):
        window_derivative = derivative[i-padding:i+padding+1]
        window_signal = padded_values[i-padding:i+padding+1]

        if (np.max(window_derivative) >= threshold or np.min(window_derivative) <= -threshold) and np.any(window_signal <= threshold_signal_value):
            start_index = i - padding
            end_index = i + padding + 1

            interpolated_values = interpolate_signal(start_index, end_index, 
                                                     cleaned_signal, method=interpolation_method)
            cleaned_signal[start_index:end_index] = interpolated_values
            interpolated_mask[start_index:end_index] = True
            nan_mask[start_index:end_index] = True  # Here should be logic to set NaN, currently just setting True for the mask

    if additional_interpolation_method:
        cleaned_signal = interpolate_mask(interpolated_mask, cleaned_signal, 
                                          additional_interpolation_method)

    cleaned_signal = cleaned_signal[padding:-padding]
    interpolated_mask = interpolated_mask[padding:-padding]
    nan_mask = nan_mask[padding:-padding]

    return pd.DataFrame({'cleaned': cleaned_signal, 'interpolated': interpolated_mask, 'nan_values': nan_mask})

def remove_range(df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    """
    Remove a range of data from a DataFrame based on index values.

    Parameters:
    - df (pd.DataFrame): DataFrame from which to remove data.
    - start_idx (int): Start index of the range to remove.
    - end_idx (int): End index of the range to remove.

    Returns:
    - pd.DataFrame: DataFrame with the specified range removed and index reset.
    """
    return df.drop(index=range(start_idx, end_idx + 1)).reset_index(drop=True)