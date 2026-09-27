"""Surface Classification XGB Tuning Script.

This script performs tuning for an XGBoost model on surface classification task
given a set of input signals in order to evaluate its expected performance.

Examples
--------
$ python surface_classification_xgb_tuning.py --config-file xgb_tuning.yaml --experiment-name 4w_imu
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Subset
from xgboost import XGBClassifier

from utils.datasets import XGBTrainingDataset
from utils.processing import generalize_classes
from utils.training import (
    load_config,
    extract_experiment_name,
    set_seed,
    seed_worker,
    get_device,
    average_over_splits,
    WINDOW_LENGTH,
    PARAM_GRID,
    THRESHOLD,
)


def xgb_training(experiment_name: str,
                 experiment_params: dict,
                 training_params: dict,
                 dataset_params: dict,
                 output_dir: Path) -> None:
    """Perform cross-validation for XGBoost surface classification.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment for identification and logging purposes.
    experiment_params : dict
        Dictionary containing experiment configuration with keys:
        - 'channels': channel subset specification,
        - 'kinematics': list of kinematic configurations to include.
    training_params : dict
        Dictionary containing training parameters with keys:
        - 'num_epochs': number of training epochs,
        - 'batch_size': CV process batch size,
        - 'seed': random seed for reproducibility.
    dataset_params : dict
        Dictionary containing dataset configuration with keys:
        - 'data_dir': path to data directory,
        - 'labels_file': path to labels JSON file,
        - 'data_frequency': data sampling frequency,
        - 'time_features': list of time domain features to include in the analysis,
        - 'freq_features': list of frequency domain features to include in the analysis,
        - 'generalized_classes': whether to use generalized class labels.
    output_dir : pathlib.Path
        Path where the analysis results will be saved as JSON.
    """

    print(f"Running experiment: {experiment_name}")
    print(f"Subset: {experiment_params['channels']}, ",
          f"Configurations: {experiment_params['kinematics']}")

    g = set_seed(training_params['seed'])
    device = get_device()

    # Set up paths.
    data_dir = Path(dataset_params['data_dir'])
    labels_file = Path(dataset_params['labels_file'])

    # Load dataset.
    with open(labels_file, encoding='utf-8') as fp:
        labels = json.load(fp)

    dataset = [(data_dir.joinpath(f'{key}.csv'), values['surface'])
               for key, values in labels.items()
               if values['kinematics'] in experiment_params['kinematics']]
    x = [run[0] for run in dataset]
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

    classes = le.classes_
    num_classes = len(classes)

    cv_data = XGBTrainingDataset(x,
                                 y,
                                 data_frequency=dataset_params['data_frequency'],
                                 window_length=WINDOW_LENGTH,
                                 subset=experiment_params['channels'],
                                 time_features=dataset_params['time_features'],
                                 freq_features=dataset_params['freq_features'])

    history = {}
    history_features = {}
    sss = StratifiedShuffleSplit(n_splits=40, test_size=0.2)

    for i, (training_index, test_index) in enumerate(sss.split(x, y)):
        # Initialize the model in each split.
        xgb_model = XGBClassifier(device=device, tree_method='hist',
                                  objective='multi:softprob', num_class=num_classes)

        train_subset = Subset(cv_data, training_index)
        test_subset = Subset(cv_data, test_index)
        train_dataloader = DataLoader(train_subset,
                                      batch_size=len(train_subset),
                                      worker_init_fn=seed_worker,
                                      generator=g,
                                      shuffle=True)
        test_dataloader = DataLoader(test_subset,
                                     batch_size=len(test_subset),
                                     worker_init_fn=seed_worker,
                                     generator=g)

        # Extract the whole datasets to variables.
        x_train, y_train = next(iter(train_dataloader))
        x_test, y_true = next(iter(test_dataloader))

        # Find the best hyperparameters.
        clf_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=PARAM_GRID, cv=5,
                                        scoring='f1_weighted', n_jobs=4, verbose=4)
        clf_search.fit(x_train, y_train)

        # Find the most important features from the best estimator.
        importances = clf_search.best_estimator_.feature_importances_
        idx = np.arange(len(importances))

        # Take features above threshold.
        best_features = idx[importances > THRESHOLD]

        # Fit the estimator on the best sets of hyperparameters and features.
        xgb_tuned = XGBClassifier(device=device,
                                  tree_method='hist',
                                  objective='multi:softprob',
                                  num_class=num_classes,
                                  **clf_search.best_params_)
        xgb_tuned.fit(x_train[:, best_features], y_train)
        y_pred = xgb_tuned.predict(x_test[:, best_features])
        history[i + 1] = classification_report(y_true, y_pred, output_dict=True)
        history_features[i + 1] = dict(zip(cv_data.engineered_features[best_features].tolist(),
                                           importances[best_features].tolist()))

    base_filename = '_'.join(['xgb', str(num_classes)]
                             + experiment_params['kinematics']
                             + experiment_params['channels']).lower()

    with open(output_dir.joinpath(f'{base_filename}.json'), 'w', encoding='utf-8') as fp:
        json.dump(history, fp)

    with open(output_dir.joinpath(f'{base_filename}_features.json'), 'w', encoding='utf-8') as fp:
        json.dump(history_features, fp)

    # Average classification reports over splits and save to JSON files.
    average_over_splits(history, base_filename, output_dir)


def main():
    """Main script for loading configuration file and running the experiment."""

    parser = argparse.ArgumentParser(description='CNN Cross-Validation for Surface Classification')
    parser.add_argument('--config-file',
                        type=str,
                        required=True,
                        help='YAML configuration file name.')
    parser.add_argument('--experiment-name',
                        type=str,
                        required=True,
                        help='Experiment ID to run.')
    parser.add_argument('--output-dir',
                        default='results/logs/tuning/xgb',
                        type=Path,
                        help='Output directory path.')
    args = parser.parse_args()

    config_path = Path('configs').joinpath(args.config_file)
    all_params = load_config(config_path)
    experiments, training_params, dataset_params = all_params.values()
    experiment_name = args.experiment_name
    experiment_params = next(
        (params['experiment_params'] for params in experiments
         if extract_experiment_name(params['experiment_params']) == experiment_name)
    )

    number_of_classes = '3' if dataset_params['generalized_classes'] else '10'
    kinematics_set = '_'.join(experiment_params['kinematics']).lower()
    output_dir = args.output_dir.joinpath(number_of_classes, kinematics_set)
    output_dir.mkdir(parents=True, exist_ok=True)

    xgb_training(experiment_name, experiment_params, training_params,
                 dataset_params, output_dir)
    time.sleep(1)


if __name__ == "__main__":
    main()
