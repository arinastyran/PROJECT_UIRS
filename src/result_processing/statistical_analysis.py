"""Main script for statistical analysis of results."""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

from utils.visualization import build_directory_dict


def main():
    """Main statistical analysis function."""

    parser = argparse.ArgumentParser(description='Run statistical analysis on computed results.')
    parser.add_argument('--result-dir',
                        default='results',
                        type=Path,
                        help='Base directory for experiment outputs')
    args = parser.parse_args()

    # Collect all directories and file names.
    main_results_dir = args.result_dir.joinpath('logs', 'tuning')
    table_dir = args.result_dir.joinpath('logs', 'tables')
    table_dir.mkdir(parents=True, exist_ok=True)
    experiment_directory_dict = build_directory_dict(main_results_dir)

    for model, class_level in experiment_directory_dict.items():
        for classes, experiment_configurations in class_level.items():
            ks_information_dict = {}
            for kinematics, filenames in experiment_configurations.items():
                configuration_results_dir = main_results_dir.joinpath(model, classes, kinematics)
                results = {}

                imu_filepath = configuration_results_dir / f'{filenames[0]}.json'
                with open(imu_filepath, encoding='utf-8') as fp:
                    results['imu'] = json.load(fp)

                both_filepath = configuration_results_dir / f'{filenames[2]}.json'
                with open(both_filepath, encoding='utf-8') as fp:
                    results['both'] = json.load(fp)

                imu_scores = [result['weighted avg']['f1-score']
                              for result in results['imu'].values()]
                both_scores = [result['weighted avg']['f1-score']
                               for result in results['both'].values()]

                ks_statistic = ks_2samp(imu_scores, both_scores)
                ks_information_dict[kinematics] = {'statistic': ks_statistic.statistic,
                                                   'pvalue': ks_statistic.pvalue}

            ks_information_df = pd.DataFrame(ks_information_dict)
            ks_information_df.to_json(table_dir.joinpath(f'{model}_{classes}_ks_information.json'))


if __name__ == "__main__":
    main()
