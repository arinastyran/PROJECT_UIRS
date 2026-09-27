"""Main script for plotting tuning results."""

import argparse
from pathlib import Path

from utils.visualization import setup_matplotlib, build_directory_dict, plot_tuning_results


def main():
    """Main tuning results plotting function."""

    parser = argparse.ArgumentParser(description='Run multiple training instances.')
    parser.add_argument('--result-dir',
                        default='results',
                        type=Path,
                        help='Base directory for experiment outputs.')
    args = parser.parse_args()

    # Set up matplotlib.
    setup_matplotlib({'lines.linewidth': 0.5})

    # Collect all directories and file names.
    experiment_directory_dict = build_directory_dict(args.result_dir.joinpath('logs', 'tuning'))

    for model, class_level in experiment_directory_dict.items():
        for classes, experiment_configurations in class_level.items():
            plot_tuning_results(model, classes, experiment_configurations, args.result_dir)


if __name__ == "__main__":
    main()
