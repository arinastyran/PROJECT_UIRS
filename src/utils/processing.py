"""Utility functions for data processing and feature extraction."""

import random

import numpy as np
import pandas as pd
import yaml
from scipy import stats


# ---------------- Bag processing functions ---------------- #
def format_to_yaml(column: pd.DataFrame) -> list:
    """Format string to yaml before unpacking.

    Parameters
    ----------
    column : pd.DataFrame
        Series containing string data to be formatted.

    Returns
    -------
    data : list of lists
        List of lists where each inner list contains formatted data from the column.
    """

    data = column.tolist()
    data = [line.replace(', ', '];[') for line in data]
    data = [line.replace('[', '') for line in data]
    data = [line.replace(']', '') for line in data]
    data = [line.split(sep=';') for line in data]
    return data


def unpack_load_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Unpack data containing servo load to separate columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing column with servo load data.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with servo load columns for each wheel.
    """

    # Convert strings to readable format.
    data = format_to_yaml(dataframe['values'])

    # Load YAML data.
    data = [[yaml.safe_load(line) for line in separate_lines] for separate_lines in data]

    # Prepare unpacked data.
    # Load is divided by 100 because it is a percentage value.
    transformed_data = [[abs(dictionary['Amper']) / 100 for dictionary in line] for line in data]
    cols = [''.join(('wheel_load.', str(idx+1))) for idx in range(len(data[0]))]
    load = pd.DataFrame(transformed_data, columns=cols)
    df = pd.concat([dataframe, load], axis=1)
    df.drop(columns=['values'], inplace=True)
    return df


def unpack_ang_vel_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Unpack data containing wheel angular velocities to separate columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing column with wheel angular velocity data.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with angular velocity columns for each wheel.
    """

    # Convert strings to readable format.
    data = format_to_yaml(dataframe['values'])

    # Load YAML data.
    data = [[yaml.safe_load(line) for line in separate_lines] for separate_lines in data]

    # Prepare unpacked data.
    transformed_data = [[abs(dictionary['AngVelocity']) for dictionary in line] for line in data]
    cols = [''.join(('wheel_angular_velocity.', str(idx+1))) for idx in range(len(data[0]))]
    ang_vel = pd.DataFrame(transformed_data, columns=cols)
    df = pd.concat([dataframe, ang_vel], axis=1)
    df.drop(columns=['values'], inplace=True)
    return df


def interpolate_servo_data(dataframe: pd.DataFrame, new_timesteps: pd.Series) -> pd.DataFrame:
    """Interpolate servo data to match IMU.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe containing original servo data. Requires 'Time' column as the first one.
    new_timesteps : pd.Series
        New timestamps for interpolation.

    Returns
    -------
    resampled_dataframe : pd.DataFrame
        DataFrame containing samples mapped to new timestamps.
    """

    resampled_dataframe = pd.DataFrame(new_timesteps, columns=['Time'])
    for col in dataframe.columns[1:]:
        resampled_dataframe[col] = np.interp(new_timesteps, dataframe['Time'], dataframe[col])
    return resampled_dataframe


def calculate_mean_power(wheel_load: pd.DataFrame,
                         wheel_angular_velocity: pd.DataFrame,
                         circular_indexing: bool) -> pd.DataFrame:
    """Calculate average power consumption for left and right side wheels based on index assignment.

    Parameters
    ----------
    wheel_load : pd.DataFrame
        Wheel load data from run.
    wheel_angular_velocity : pd.DataFrame
        Wheel angular velocity data from run.
    circular_indexing : bool
        Wheel indexing format.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with average power consumption columns for each side.
    """

    number_of_wheels = len(wheel_load.columns)
    cols = [''.join(('estimated_power.', str(idx+1))) for idx in range(number_of_wheels)]
    estimated_power = wheel_load.values * wheel_angular_velocity.values
    df = pd.DataFrame(estimated_power, columns=cols, index=wheel_load.index)

    # Properly assign wheels to left and right side.
    if circular_indexing:
        cols_left = [''.join(('estimated_power.', str(idx+1))) for idx in range(number_of_wheels)
                     if idx in (0, 3)]
        cols_right = [''.join(('estimated_power.', str(idx+1))) for idx in range(number_of_wheels)
                      if idx in (1, 2)]
    else:
        cols_left = [''.join(('estimated_power.', str(idx+1))) for idx in range(number_of_wheels)
                     if idx % 2 == 0]
        cols_right = [''.join(('estimated_power.', str(idx+1))) for idx in range(number_of_wheels)
                      if idx % 2 != 0]

    # Calculate mean power for each side.
    power_left = df[cols_left]
    power_right = df[cols_right]
    df['mean_power_left'] = power_left.mean(axis=1)
    df['mean_power_right'] = power_right.mean(axis=1)
    return df


# ---------------- Dataset sampling functions ---------------- #
def sample_sequence(dataframe: pd.DataFrame,
                    selected_columns: list,
                    window_length: int) -> np.ndarray:
    """Draw random windowed sample from run.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Data from run.
    selected_columns : list
        Subset of columns representing chosen signal channels.
    window_length : int
        Number of time steps in window.

    Returns
    -------
    np.ndarray
        Array containing selected channels and time steps.
    """

    df = dataframe[selected_columns]
    max_start_index = len(dataframe) - window_length - 1
    init_timestep = random.randint(0, max_start_index)
    return df.iloc[init_timestep:init_timestep + window_length].to_numpy()


def sequence_run(dataframe: pd.DataFrame, selected_columns: list,
                 window_length: int, window_stride: int = 1) -> list:
    """Convert dataframe to list of windows.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Data from run.
    selected_columns : list
        Subset of columns representing chosen signal channels.
    window_length : int
        Number of time steps in window.
    window_stride : int
        How many time steps to omit from original data between windows.

    Returns
    -------
    df_list : list of np.ndarray
        List of dataframes containing selected channels and time steps.
    """

    df = dataframe[selected_columns]
    num_of_windows = len(dataframe) - window_length + window_stride
    df_list = [df.iloc[init_timestep:init_timestep + window_length].to_numpy()
               for init_timestep in range(0, num_of_windows, window_stride)]
    return df_list


# ---------------- XGBoost specific functions ---------------- #
def get_sample_features(sequence: np.ndarray,
                        time_features: list | None = None,
                        freq_features: list | None = None) -> np.ndarray:
    """Get features for single sequence.

    Parameters
    ----------
    sequence : np.ndarray
        Time series data for which features are to be extracted.
    time_features : list of str, optional
        List of time domain features to extract. If None, no time features are extracted.
    freq_features : list of str, optional
        List of frequency domain features to extract. If None, no frequency features are extracted.

    Returns
    -------
    np.ndarray
        Extracted features as a flat array.
    """

    if time_features is None and freq_features is None:
        raise ValueError('At least one of time_features or freq_features must be provided.')

    engineered_features = []
    if time_features is not None:
        engineered_features.extend(get_time_domain(sequence, time_features))
    if freq_features is not None:
        engineered_features.extend(get_frequency_domain(sequence, freq_features))

    return np.array(engineered_features)


def get_time_domain(sequence: np.ndarray, time_features: list) -> np.ndarray:
    """Get time features.

    Parameters
    ----------
    sequence : np.ndarray
        Time series data for which time features are to be extracted.
    time_features : list of str
        List of time domain features to extract.

    Returns
    -------
    np.ndarray
        Extracted time features as a flattened array.
    """

    engineered_time_features = []

    if 'min' in time_features:
        engineered_time_features.append(np.min(sequence, axis=0))
    if 'max' in time_features:
        engineered_time_features.append(np.max(sequence, axis=0))
    if 'mean' in time_features:
        mean = np.mean(sequence, axis=0)
        engineered_time_features.append(mean)
    if 'std' in time_features:
        engineered_time_features.append(np.std(sequence, axis=0))
    if 'skew' in time_features:
        engineered_time_features.append(stats.skew(sequence, axis=0))
    if 'kurt' in time_features:
        engineered_time_features.append(stats.kurtosis(sequence, axis=0))

    if 'rms' in time_features:
        rms = np.sqrt(np.mean(sequence ** 2, axis=0))
        engineered_time_features.append(rms)
    if 'peak' in time_features:
        peak = np.max(np.abs(sequence), axis=0)
        engineered_time_features.append(peak)
    if 'p2p' in time_features:
        engineered_time_features.append(np.ptp(sequence, axis=0))

    if 'crest' in time_features:
        engineered_time_features.append(peak / rms)  # how extreme the peaks are in a waveform
    if 'form' in time_features:
        engineered_time_features.append(rms / mean)  # the ratio of the RMS (root mean square) value to the average value
    if 'pulse' in time_features:
        engineered_time_features.append(peak / mean)

    return np.array(engineered_time_features).flatten()


def get_frequency_domain(sequence: np.ndarray, freq_features: list) -> np.ndarray:
    """Get frequency features.

    Parameters
    ----------
    sequence : np.ndarray
        Time series data for which frequency features are to be extracted.
    freq_features : list of str
        List of frequency domain features to extract.

    Returns
    -------
    np.ndarray
        Extracted frequency features as a flattened array.
    """

    engineered_freq_features = []

    ft = np.fft.fft(sequence, axis=0)
    s = np.abs(ft ** 2) / len(sequence)

    if 'sum' in freq_features:
        engineered_freq_features.append(np.sum(s, axis=0))
    if 'max' in freq_features:
        engineered_freq_features.append(np.max(s, axis=0))
    if 'mean' in freq_features:
        engineered_freq_features.append(np.mean(s, axis=0))
    if 'peak' in freq_features:
        engineered_freq_features.append(np.max(np.abs(s), axis=0))
    if 'var' in freq_features:
        engineered_freq_features.append(np.var(s, axis=0))

    return np.array(engineered_freq_features).flatten()


# ---------------- General processing functions ---------------- #
def generalize_classes(surface_classes: list) -> list:
    """Generalize surface classes.

    Parameters
    ----------
    surface_classes : list of str
        List of original surface classes.

    Returns
    -------
    list of str
        List of generalized surface classes.
    """
    return ['group_0' if label in ('long_carpet', 'artificial_grass')
            else 'group_2' if label in ('pvc_foamboard', 'osb',
                                        'foam_underlayment', 'eva_foam_tiles')
            else 'group_1' for label in surface_classes]
