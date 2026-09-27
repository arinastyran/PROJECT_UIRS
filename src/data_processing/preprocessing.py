"""Scripts to convert ROS bag files to CSV format and preprocess the data.

Data directory tree must obey the following structure:
{<class_label>, <test_id>}/<kinematics>/<bag_name>

This module uses bagpy library that is compatible with ROS 1 only!
If your data comes from a ROS 2 platform, you need to convert bag files to ROS 1.
"""

import argparse
import functools as ft
import json
import sys
from pathlib import Path

import pandas as pd
from bagpy import bagreader
from tqdm import tqdm

from utils.processing import (
    unpack_load_data,
    unpack_ang_vel_data,
    interpolate_servo_data,
    calculate_mean_power,
)


def convert_rosbags(bag_dir: Path) -> None:
    """Convert ROS bag files in the specified directory to CSV format.

    Parameters
    ----------
    bag_dir : pathlib.Path
        Path to the directory containing ROS bag files.
    """

    bag_paths = list(bag_dir.rglob('*.bag'))

    if not bag_paths:
        print(f"Error: No .bag files found in directory: {bag_dir}!")
        print("Please check the directory path and ensure it contains ROS bag files.")
        sys.exit(1)

    print(f"Found {len(bag_paths)} bag files to process.")
    print("Extracting data from ROS bag files...")

    for bag_path in tqdm(bag_paths):
        b = bagreader(str(bag_path))

        for topic in b.topics:
            b.message_by_topic(topic)


def process_bag(bag_path: Path,
                circular_indexing: bool) -> tuple[str, pd.DataFrame, dict[str, str]]:
    """Apply preprocessing pipeline to data from a single ROS bag file.

    Parameters
    ----------
    bag_path : pathlib.Path
        Path to the ROS bag file.
    circular_indexing : bool
        Flag indicating whether servo indexing is circular (True) or alternating (False).

    Returns
    -------
    bag_name : str
        ROS bag file name.
    bag_df : pd.DataFrame
        Processed data.
    label : dict of str
        Dictionary containing class label/test identificator and kinematics.
    """

    # Split path into chunks.
    subfolders = bag_path.parts
    kinematics, surface = subfolders[-2], subfolders[-3]
    bag_name = '_'.join(subfolders[-3:])

    # Set servo indexing flag.
    if kinematics == '4W':
        circular_indexing = True

    # Read IMU and servo data.
    imu_df = pd.read_csv(bag_path.joinpath('imu-data.csv'))
    imu_time_reference = pd.read_csv(bag_path.joinpath('imu-time_ref.csv'))
    wheel_load_df = pd.read_csv(bag_path.joinpath('Servo_data.csv'))
    wheel_velocity_df = pd.read_csv(bag_path.joinpath('wheel_feedback.csv'))

    # Select IMU data rows that have timestamp reference.
    matching_timestamps = imu_df['header.seq'].isin(imu_time_reference['header.seq'])
    imu_df = imu_df[matching_timestamps]

    # Select servo data that lies within IMU data timeframe bounds.
    timeframe_min, timeframe_max = imu_df['Time'].min(), imu_df['Time'].max()
    above_min_timestamp = wheel_load_df['Time'] >= timeframe_min
    below_max_timestamp = wheel_load_df['Time'] <= timeframe_max
    wheel_load_df = wheel_load_df[above_min_timestamp & below_max_timestamp]
    above_min_timestamp = wheel_velocity_df['Time'] >= timeframe_min
    below_max_timestamp = wheel_velocity_df['Time'] <= timeframe_max
    wheel_velocity_df = wheel_velocity_df[above_min_timestamp & below_max_timestamp]

    # Unpack load and angular velocity values.
    wheel_load_df = unpack_load_data(wheel_load_df)
    wheel_velocity_df = unpack_ang_vel_data(wheel_velocity_df)

    # Adjust servo timesteps relative to IMU data.
    wheel_load_df['Time'] -= timeframe_min
    wheel_velocity_df['Time'] -= timeframe_min

    # Interpolate servo data to match IMU 100 Hz rate.
    new_timesteps = imu_time_reference['time_ref.secs'] + imu_time_reference['time_ref.nsecs'] / 1e9
    new_timesteps -= new_timesteps.min()
    imu_df['Time'] = new_timesteps
    resampled_wheel_load_df = interpolate_servo_data(wheel_load_df, new_timesteps)
    resampled_wheel_velocity_df = interpolate_servo_data(wheel_velocity_df, new_timesteps)

    # Merge all data into a single dataframe.
    dfs = [imu_df, resampled_wheel_load_df, resampled_wheel_velocity_df]
    df = ft.reduce(lambda left, right: pd.merge(left, right, how='outer', on='Time'), dfs)

    # Clean the resulting dataframe.
    imu_columns = ['linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z',
                   'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z']
    wheel_load_columns = [col for col in df.columns if 'wheel_load' in col]
    wheel_angular_velocity_columns = [col for col in df.columns if 'wheel_angular_velocity' in col]
    df = df[['Time'] + imu_columns + wheel_load_columns + wheel_angular_velocity_columns]

    # Trim the first and last 10% of rows for more coherent data.
    clip_var = int(len(df) * .1)
    df = df.iloc[clip_var:-clip_var].reset_index(drop=True)
    df['Time'] -= df['Time'].min()

    # Center Z axis acceleration around 0.
    df['linear_acceleration.z'] -= df['linear_acceleration.z'].mean()

    # Calculate estimated power values.
    power = calculate_mean_power(df[wheel_load_columns],
                                 df[wheel_angular_velocity_columns],
                                 circular_indexing)
    bag_df = pd.concat([df, power], axis=1)

    # Create label dictionary.
    label = {'surface': surface, 'kinematics': kinematics}

    return bag_name, bag_df, label


def merge_csvs(bag_dir: Path) -> None:
    """Merge CSV files from ROS bag data into a single dataframe per bag.

    Parameters
    ----------
    bag_dir : pathlib.Path
        Path to the directory containing ROS bag files.
    """

    # Flag indicating whether servo indexing is circular (True) or alternating (False).
    # In alternating indexing left indexes are even and right indexes are odd.
    circular_indexing = False

    # Set working directories.
    target_dir = bag_dir.parents[0] / 'processed'
    target_dir.mkdir(parents=True, exist_ok=True)

    # Case specific method for gathering dataset directories.
    bag_paths = [folder for folder in bag_dir.rglob('*W/*') if folder.is_dir()]

    if not bag_paths:
        print(f"\nError: No bag directories found matching pattern '*W/*' in: {bag_dir}!")
        print("Expected directory structure: {surface}/{kinematics}/{bag_name}/")
        print("Where kinematics should end with 'W' (e.g., '2W', '4W').")
        sys.exit(1)

    print(f"\nFound {len(bag_paths)} bag directories to process.")
    print("Processing data from ROS topics...")
    # Loop over bags and process data.
    labels = {}

    for bag_path in tqdm(bag_paths):
        bag_name, bag_df, label = process_bag(bag_path, circular_indexing)

        # Write dataframe to CSV file.
        bag_df.to_csv(target_dir.joinpath(f'{bag_name}.csv'))

        labels[bag_name] = label

    # Dump labels to JSON.
    with open(bag_dir.parents[0] / 'labels.json', 'w', encoding='utf-8') as fp:
        json.dump(labels, fp)


def main():
    """Main script for extracting data from ROS bag files and applying preprocessing pipeline."""

    parser = argparse.ArgumentParser(description='Convert ROS bags to CSV and preprocess them.')
    parser.add_argument('--bag-dir',
                        default='data/train_set/raw',
                        type=Path,
                        help='Path to dataset directory relative to root.')
    parser.add_argument('--merge-files',
                        default=True,
                        type=bool,
                        help='Whether to merge extracted CSV files into single dataframes per bag.')
    args = parser.parse_args()

    if not args.bag_dir.exists():
        print(f"Error: Directory does not exist: {args.bag_dir}!")
        sys.exit(1)

    convert_rosbags(args.bag_dir)

    if args.merge_files:
        merge_csvs(args.bag_dir)


if __name__ == "__main__":
    main()
