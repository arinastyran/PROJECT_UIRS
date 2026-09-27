"""Shared plotting configuration and methods for all visualization scripts."""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler


# Color definitions
NICER_GREEN = '#159C48'
NICER_BLUE = '#00A0FF'
ORANGE = '#FBBC04'
PINK = '#DB00CF'
MAD_PURPLE = '#732BF5'
LIGHT_GREEN = '#66C2A5'


# Default figure parameters
DEFAULT_FIGURE_PARAMS = {
    'figure.figsize': [4, 3],
    'axes.prop_cycle': cycler('color',
                              [NICER_BLUE, NICER_GREEN, ORANGE, PINK, MAD_PURPLE, LIGHT_GREEN]),
    'lines.linewidth': 1.5,
    'font.size': 10,
}
DPI = 1000


# Experiment constants
T_95 = 2.228  # t-value for 95% confidence interval
THRESHOLDS = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]


# Directories
IMAGE_DIR = Path('figures')
TABLE_DIR = os.path.join('logs', 'tables')


# ---------------- General plotting functions ---------------- #
def setup_matplotlib(custom_params=None) -> None:
    """Configure matplotlib with default or custom parameters.

    Parameters
    ----------
    custom_params : dict, optional
        Additional or override parameters.
    """

    params = DEFAULT_FIGURE_PARAMS.copy()

    if custom_params:
        params.update(custom_params)

    plt.rcParams.update(params)


# ---------------- Exploratory analysis plotting functions ---------------- #
def plot_signal(dataframe: pd.DataFrame, columns: list, output_dir: Path,
                title: str | None = None, y_label: str | None = None, alpha: float = 1.0) -> None:
    """Plot time series signals in a single figure.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Data from run.
    columns : list of str
        Which signals to plot.
    output_dir : pathlib.Path
        Directory to save the plot.
    title : str
        Plot title.
    y_label : str
        y-axis label.
    alpha : float
        Plot transparency.
    """

    setup_matplotlib({'figure.figsize': [8, 3], 'font.size': 12})
    fig, ax = plt.subplots(layout='constrained')
    time = dataframe['Time'] - dataframe['Time'].min()

    for col in columns:
        ax.plot(time, dataframe[col], label=col, alpha=alpha)

    if title is not None:
        ax.title(title)

    if y_label is not None:
        ax.set_ylabel(y_label)

    ax.set_xlabel('time [s]')
    fig.legend(loc='outside upper right', ncol=len(columns))
    figure_name = y_label.lower().split('[')[0].strip().replace(' ', '_') if y_label else 'signal_plot'
    plt.savefig(output_dir.joinpath(f'{figure_name}.png'), dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_many(dataframe: pd.DataFrame, columns: list, output_dir: Path,
              y_label: str | None = None, alpha: float = 1.0) -> None:
    """Plot time series signals in separate axes.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Data from run.
    columns : list of str
        Which signals to plot.
    output_dir : pathlib.Path
        Directory to save the plot.
    y_label : str
        y-axis label.
    alpha : float
        Plot transparency.
    """

    setup_matplotlib({'figure.figsize': [8, 3]})
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    time = dataframe['Time'] - dataframe['Time'].min()

    for i, ax in enumerate(axes.flat):
        col = columns[i]
        ax.plot(time, dataframe[col], label=col, alpha=alpha)
        ax.set_title(f"{col}")

    if y_label is not None:
        fig.supylabel(y_label)

    fig.supxlabel('time [s]')
    plt.tight_layout()
    figure_name = y_label.lower().split('[')[0].strip().replace(' ', '_') if y_label else 'signal_plot'
    plt.savefig(output_dir.joinpath(f'{figure_name}.png'), dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_correlation(correlation_matrix: pd.DataFrame, output_dir: Path) -> None:
    """Plot correlation matrix of dataframe features.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Data from run.
    output_dir : pathlib.Path
        Directory to save the plot.
    """

    setup_matplotlib({'font.size': 8})
    sns.heatmap(correlation_matrix,
                vmin=-1,
                vmax=1,
                cmap='Blues',
                annot=True,
                linewidths=0.5,
                fmt='.2f')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(output_dir.joinpath('corr_plot.png'), dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_odom_error(odometry_errors: pd.DataFrame, assigned_labels: list, image_path: Path) -> None:
    """Plot odometry error results.

    Parameters
    ----------
    odometry_errors : pd.DataFrame
        Dataframe with odometry errors.
    assigned_labels : list of int
        List of assigned class labels.
    image_path : pathlib.Path
        Path to save the plot.
    """

    original_labels = ['laminate flooring',
                       'short carpet',
                       'long carpet',
                       'artificial grass',
                       'pvc foamboard',
                       'linoleum',
                       'ceramic tiles',
                       'osb',
                       'foam underlayment',
                       'eva foam tiles']

    setup_matplotlib()
    sns.scatterplot(
        x=original_labels,
        y=odometry_errors,
        hue=assigned_labels,
        palette=[NICER_BLUE, NICER_GREEN, ORANGE],
        s=40,
    )
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.yticks(ticks=np.arange(-6, 6, 3) / 10)
    plt.ylim(-0.6, 0.3)
    plt.xlabel('original labels')
    plt.ylabel('scaled odometry error')
    plt.grid()
    plt.legend(loc='lower right')
    plt.savefig(image_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_threshold_analysis(threshold_data_path: Path, output_dir: Path) -> None:
    """Plot threshold analysis results.

    Parameters
    ----------
    threshold_data_path : pathlib.Path
        Path to the JSON file with threshold analysis results.
    output_dir : pathlib.Path
        Directory to save the plot.
    """

    with open(threshold_data_path, 'r', encoding='utf-8') as f:
        threshold_history = json.load(f)

    # Calculate average F1-scores and standard deviations
    average_f1 = []
    ci_f1 = []

    for threshold in THRESHOLDS:
        scores = threshold_history[str(threshold)]
        average_f1.append(np.mean(scores))
        ci_f1.append(T_95 * np.std(scores) / np.sqrt(len(scores)))

    plt.figure(figsize=(8, 3))
    plt.errorbar(THRESHOLDS, average_f1, yerr=ci_f1,
                 marker='o', capsize=4, capthick=1,
                 linewidth=1, markersize=4, c='k')
    plt.xlabel('importance threshold')
    plt.ylabel('average F1-score')
    plt.grid()
    plt.xscale('log')
    plt.xlim([5e-06, 1e-01])
    plt.tight_layout()
    plt.savefig(output_dir.joinpath('threshold_analysis.png'), dpi=DPI, bbox_inches='tight')
    plt.close()


# ---------------- Main plotting functions ---------------- #
def plot_cv_results(model: str, number_of_classes: str,
                    experiment_configurations: dict, result_dir: Path) -> None:
    """Plot cross-validation results with confidence intervals.

    Parameters
    ----------
    model : str
        Model type.
    number_of_classes : str
        Number of class labels in the experiment.
    experiment_configurations : dict
        Dictionary of experiment configuration parameters.
    result_dir : pathlib.Path
        Path to directory storing experiment results.
    """

    # Get figure directory path.
    image_dir = result_dir.joinpath('figures', 'cv')
    image_dir.mkdir(parents=True, exist_ok=True)

    for kinematics, filenames in experiment_configurations.items():
        # Load results.
        results_dir = result_dir.joinpath('logs', 'cv', model, number_of_classes, kinematics)
        results = load_cv_results(filenames, results_dir)

        # Gather labels.
        labels = experiments_to_labels(filenames)

        # Plot results.
        plt.figure()
        figure_name = f'cv_{model}_{number_of_classes}_{kinematics}.png'.lower()
        output_path = image_dir.joinpath(figure_name)

        for result, label in zip(results.values(), labels):
            df = pd.DataFrame(result)
            res_array = np.array(df.loc['f1_score'].values.tolist()).T
            x = np.arange(1, res_array.shape[0] + 1)

            # Calculate statistics.
            average_f1 = res_array.mean(axis=1)
            std_dev_f1 = res_array.std(axis=1)
            ci = T_95 * std_dev_f1 / np.sqrt(10)

            # Plot with confidence interval.
            plt.plot(x, average_f1, label=label)
            plt.fill_between(x, (average_f1 - ci), (average_f1 + ci), alpha=.2)

        # Configure plot.
        plt.xlim(1, 100)
        plt.ylim(0.4, 1)
        plt.xlabel('epoch')
        plt.ylabel('average F1-score')
        plt.grid(which='major', axis='both', linewidth=1)
        plt.grid(which='minor', axis='both', linewidth=0.4)
        plt.minorticks_on()
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close()


def plot_tuning_results(model: str, number_of_classes: str,
                        experiment_configurations: dict, result_dir: Path) -> None:
    """Plot tuning results with confidence intervals.

    Parameters
    ----------
    model : str
        Model type.
    number_of_classes : str
        Number of class labels in the experiment.
    experiment_configurations : dict
        Dictionary of experiment configuration parameters.
    result_dir : pathlib.Path
        Path to directory storing experiment results.
    """

    # Set up paths.
    image_dir = result_dir.joinpath('figures', 'tuning')
    image_dir.mkdir(parents=True, exist_ok=True)
    figure_name = f'tuning_{model}_{number_of_classes}.png'.lower()
    output_path = image_dir.joinpath(figure_name)

    plt.figure()
    width = 0.2
    ind = np.arange(3)  # the x locations for the groups

    sorted_by_channel = defaultdict(list)
    for kinematics, filenames in experiment_configurations.items():
        sorted_by_channel['imu'].append((kinematics, filenames[0]))
        sorted_by_channel['est. power'].append((kinematics, filenames[1]))
        sorted_by_channel['both'].append((kinematics, filenames[2]))

    for i, (label, channel_data) in enumerate(sorted_by_channel.items()):
        # Load results.
        results = {}
        for kinematics, filename in channel_data:
            results_dir = result_dir.joinpath('logs',
                                              'tuning',
                                              model,
                                              number_of_classes,
                                              kinematics)
            results[filename] = load_tuning_results(filename, results_dir)
        average_f1 = [result['mean']['weighted avg']['f1-score'] for result in results.values()]
        ci_f1 = [result['ci']['weighted avg']['f1-score'] for result in results.values()]

        plt.bar(ind + i * width,
                average_f1,
                width=width,
                yerr=ci_f1,
                label=label,
                capsize=4,
                alpha=0.6)

    # Configure plot.
    plt.xticks(ticks=np.arange(len(experiment_configurations)) + width,
               labels=['4W', '6W', '4W, 6W'])
    plt.yticks(ticks=np.arange(5, 11) / 10)
    plt.ylim(0.5, 1)
    plt.xlabel('configuration')
    plt.ylabel('average F1 score')
    plt.grid(which='major', axis='y', linewidth=1)
    plt.grid(which='minor', axis='y', linewidth=0.4)
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=False)
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


# ---------------- Helper functions ---------------- #
def build_directory_dict(root: Path) -> dict:
    """Builds a nested dictionary of directories and files using pathlib.

    Parameters
    ----------
    root : pathlib.Path
        Root directory to start the search.

    Returns
    -------
    result : dict
        Nested dictionary with structure {model: {classes: {kinematics: [filenames]}}}.
    """

    result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for json_file in root.rglob('*.json'):
        # Get relative path parts.
        relative = json_file.relative_to(root)
        parts = relative.parts

        # Expected model/classes/kinematics/file.json.
        if len(parts) >= 4:
            model = parts[-4]
            classes = parts[-3]  # could be version, category, etc.
            kinematics = parts[-2]
            filename = parts[-1]

            if not any(substring in filename for substring in ('mean', 'ci')):
                result[model][classes][kinematics].append(filename.removesuffix('.json'))

    for model, model_level in result.items():
        for classes, class_level in model_level.items():
            for files in class_level.values():
                files.sort(key=len)

            sorted_keys = sorted(class_level, key=len)
            sorted_dict = {key: class_level[key] for key in sorted_keys}
            result[model][classes] = sorted_dict

    # Convert nested defaultdicts to regular dicts
    return {
        model: {
            classes: dict(kinematics)
            for classes, kinematics in class_levels.items()
        }
        for model, class_levels in result.items()
    }


def load_cv_results(filenames: list, results_dir: Path) -> dict:
    """Load CV results from JSON files.

    Parameters
    ----------
    filenames : list of str
        List of filenames (with extension) to load.
    results_dir : pathlib.Path
        Directory where results are stored.

    Returns
    -------
    results : dict
        Dictionary of loaded results.
    """

    results = {}
    for filename in filenames:
        filepath = results_dir / f'{filename}.json'
        with open(filepath, encoding='utf-8') as fp:
            results[filename] = json.load(fp)
    return results


def load_tuning_results(filename: str, results_dir: Path) -> dict:
    """Load CV results from JSON files.

    Parameters
    ----------
    filename : str
        Filenames (without extension) to load.
    results_dir : pathlib.Path
        Directory where results are stored.

    Returns
    -------
    results : dict of dicts
        Dictionary of loaded results.
    """

    result = {}
    filepath_mean = results_dir / f'{filename}_mean.json'

    with open(filepath_mean, encoding='utf-8') as fp:
        result['mean'] = json.load(fp)

    filepath_ci = results_dir / f'{filename}_ci.json'

    with open(filepath_ci, encoding='utf-8') as fp:
        result['ci'] = json.load(fp)

    return result


def experiments_to_labels(files: list) -> list:
    """Map experiment filenames to plot labels.

    Parameters
    ----------
    files : list of str
        List of filenames.

    Returns
    -------
    labels : list of str
        Corresponding plot labels.
    """

    labels = []
    for filename in files:
        if 'imu' in filename:
            if 'servo' in filename:
                labels.append('both')
            else:
                labels.append('imu')
        else:
            labels.append('est. power')
    return labels
