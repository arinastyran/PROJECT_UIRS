"""Main script for producing exploratory data analysis plots."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from utils.processing import (
    unpack_load_data,
    unpack_ang_vel_data,
    calculate_mean_power,
    generalize_classes,
)
from utils.training import load_config, set_seed
from utils.visualization import plot_signal, plot_many, plot_correlation, IMAGE_DIR, TABLE_DIR


def analyze_run(bag_path: Path, output_dir: Path) -> None:
    """Local dataset analysis comprising of one run.

    Parameters
    ----------
    bag_path : Path
        Path to the directory containing the bag data.
    output_dir : Path
        Directory to save the plots.
    """

    print(f"\nAnalyzing run data in: {bag_path}")

    # Set up paths.
    image_dir = output_dir.joinpath(IMAGE_DIR)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Read IMU data.
    imu_dataframe = pd.read_csv(bag_path.joinpath('imu-data.csv'))
    imu_lin_acc_cols = ['linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z']
    imu_ang_vel_cols = ['angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z']

    # Read and unpack servo load and angular velocity values.
    wheel_load_dataframe = pd.read_csv(bag_path.joinpath('Servo_data.csv'))
    wheel_velocity_dataframe = pd.read_csv(bag_path.joinpath('wheel_feedback.csv'))
    wheel_load_dataframe = unpack_load_data(wheel_load_dataframe)
    wheel_velocity_dataframe = unpack_ang_vel_data(wheel_velocity_dataframe)
    wheel_angular_velocity_cols = [col for col in wheel_velocity_dataframe.columns
                                   if 'wheel_angular_velocity' in col]
    wheel_load_cols = [col for col in wheel_load_dataframe.columns if 'wheel_load' in col]

    # Calculate estimated power values.
    power_dataframe = calculate_mean_power(wheel_load_dataframe[wheel_load_cols],
                                           wheel_velocity_dataframe[wheel_angular_velocity_cols],
                                           True)
    power_dataframe = pd.concat([wheel_load_dataframe['Time'], power_dataframe], axis=1)
    power_cols = [col for col in power_dataframe.columns if 'mean_power' in col]

    # Plot all manuscript figures.
    plot_signal(imu_dataframe,
                imu_lin_acc_cols,
                y_label=r'IMU acceleration [$\mathregular{m/s^2}$]',
                alpha=0.5,
                output_dir=image_dir)
    plot_signal(imu_dataframe,
                imu_ang_vel_cols,
                y_label=r'IMU angular velocity [$\mathregular{rad/s}$]',
                alpha=0.5,
                output_dir=image_dir)
    plot_many(wheel_velocity_dataframe, wheel_angular_velocity_cols,
              y_label=r'servo angular velocity [$\mathregular{rpm/min}$]', output_dir=image_dir)
    plot_many(wheel_load_dataframe, wheel_load_cols,
              y_label='servo fractional load', output_dir=image_dir)
    plot_signal(power_dataframe, power_cols, y_label='power estimate', output_dir=image_dir)


def analyze_dataset(dataset_params: dict, output_dir: Path) -> None:
    """
    Global dataset analysis including:
        1. mutual information score between features and target,
        2. feature correlation analysis.

    Parameters
    ----------
    dataset_params : dict
        Dataset parameters from configuration file.
    output_dir : Path
        Directory to save the analysis results.
    """

    # Set up paths.
    data_dir = Path(dataset_params['data_dir'])
    labels_file = Path(dataset_params['labels_file'])
    image_dir = output_dir.joinpath(IMAGE_DIR)
    image_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir.joinpath(TABLE_DIR)
    table_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset.
    with open(labels_file, encoding='utf-8') as fp:
        labels = json.load(fp)

    dataset = [(data_dir.joinpath(f'{key}.csv'), values['surface'])
               for key, values in labels.items()]
    filenames = [run[0] for run in dataset]
    target_classes = [run[1] for run in dataset]

    # Encode target classes.
    le = LabelEncoder()

    if dataset_params['generalized_classes']:
        generalized_classes = generalize_classes(target_classes)
        le.fit(generalized_classes)
        y = le.transform(generalized_classes)
    else:
        le.fit(target_classes)
        y = le.transform(target_classes)

    # Read files to dataframe.
    df_list = [pd.read_csv(file, index_col=[0]) for file in filenames]

    for df, label in zip(df_list, y):
        df['target'] = label
    final_df = pd.concat(df_list, axis=0)
    columns = []

    for col_group in dataset_params['df_columns']:
        for cols in col_group.values():
            columns.extend(cols)

    x = final_df[columns]
    y = final_df['target']

    # Calculate mutual information score between features and target.
    print("\nComputing mutual information score...")
    mutual_information_score = mutual_info_classif(x.drop(columns=['Time']), y)
    mutual_information_df = pd.DataFrame(mutual_information_score.reshape(1, -1),
                                         columns=x.columns[1:],
                                         index=['Mutual Information'])
    mutual_information_df.to_json(table_dir / 'mutual_information.json')

    # Calculate and plot feature correlation matrix.
    print("\nComputing correlation matrix...")
    correlation_matrix = x.drop(columns=['Time']).corr()
    plot_correlation(correlation_matrix, image_dir)


def main():
    """Main script for dataset analysis."""

    parser = argparse.ArgumentParser(description='Dataset Analysis for Surface Classification')
    parser.add_argument('--config-file',
                        default='exploratory_analysis.yaml',
                        type=str,
                        help='YAML configuration file path.')
    parser.add_argument('--output-dir',
                        default='results',
                        type=Path,
                        help='Output directory path.')
    args = parser.parse_args()

    config_path = Path('configs').joinpath(args.config_file)
    analysis_params = load_config(config_path)
    dataset_params = analysis_params['dataset_params']
    bag_path = Path(analysis_params['bag_path'])
    set_seed(analysis_params['seed'])

    analyze_run(bag_path, args.output_dir)
    analyze_dataset(dataset_params, args.output_dir)


if __name__ == "__main__":
    main()
