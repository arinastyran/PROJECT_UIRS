"""XGBoost threshold analysis script.

This script performs cross-validation to measure the imapct of feature importance
thresholding on the classification performance of an XGBoost model.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Subset
from xgboost import XGBClassifier

from utils.datasets import XGBTrainingDataset
from utils.processing import generalize_classes
from utils.training import load_config, set_seed, seed_worker, WINDOW_LENGTH, PARAM_GRID
from utils.visualization import plot_threshold_analysis, THRESHOLDS, IMAGE_DIR, TABLE_DIR


def xgb_threshold_analysis(experiment_name: str,
                           experiment_params: dict,
                           training_params: dict,
                           dataset_params: dict,
                           output_path: Path) -> None:
    """Perform cross-validation for XGBoost threshold analysis.

    The analysis uses a predefined set of thresholds and evaluates model performance at each level.

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
        - 'seed': random seed for reproducibility.
    dataset_params : dict
        Dictionary containing dataset configuration with keys:
        - 'data_dir': path to data directory,
        - 'labels_file': path to labels JSON file,
        - 'data_frequency': data sampling frequency,
        - 'time_features': list of time domain features to include in the analysis,
        - 'freq_features': list of frequency domain features to include in the analysis,
        - 'generalized_classes': whether to use generalized class labels.
    output_path : pathlib.Path
        Path where the analysis results will be saved as JSON.
    """

    print(f"Running experiment: {experiment_name}")
    print(f"Subset: {experiment_params['channels']}, ",
          f"Configurations: {experiment_params['kinematics']}")

    g = set_seed(training_params['seed'])

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

    threshold_history = defaultdict(list)
    sss = StratifiedShuffleSplit(n_splits=40, test_size=0.2)

    for (training_index, test_index) in sss.split(x, y):
        # Initialize the model in each split.
        xgb_model = XGBClassifier(objective='multi:softprob', num_class=num_classes)

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
                                        scoring='f1_weighted', n_jobs=-1, verbose=4)
        clf_search.fit(x_train, y_train)

        # Find the most important features from the best estimator.
        importances = clf_search.best_estimator_.feature_importances_
        idx = np.arange(len(importances))

        # Test different thresholds.
        for threshold in THRESHOLDS:
            best_features = idx[importances > threshold]

            # Fit the estimator on the best sets of hyperparameters and features.
            xgb_tuned = XGBClassifier(objective='multi:softprob',
                                      num_class=num_classes,
                                      **clf_search.best_params_)
            xgb_tuned.fit(x_train[:, best_features], y_train)
            y_pred = xgb_tuned.predict(x_test[:, best_features])
            f1 = f1_score(y_true, y_pred, average='weighted')
            threshold_history[threshold].append(f1)

    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(threshold_history, f)


def main():
    """Main script for loading configuration file and running the experiment."""

    all_params = load_config(Path('configs/threshold_analysis.yaml'))
    experiment_name, experiment_params, training_params, dataset_params = all_params.values()
    output_dir = Path('results')
    image_dir = output_dir.joinpath(IMAGE_DIR)
    image_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir.joinpath(TABLE_DIR)
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir.joinpath('threshold_analysis.json')

    xgb_threshold_analysis(experiment_name,
                           experiment_params,
                           training_params,
                           dataset_params,
                           table_path)
    time.sleep(1)

    plot_threshold_analysis(table_path, image_dir)


if __name__ == "__main__":
    main()
