"""Surface Classification CNN Tuning Script.

This script performs tuning for a CNN model on surface classification task
given a set of input signals in order to evaluate its expected performance.

Examples
--------
$ python surface_classification_cnn_tuning.py --config-file cnn_tuning.yaml --experiment-name 4w_imu
"""

import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.models import CNNSurfaceClassifier
from utils.datasets import CNNTrainingDataset
from utils.processing import generalize_classes
from utils.training import (
    EarlyStopper,
    ProgressReporter,
    load_config,
    extract_experiment_name,
    set_seed,
    seed_worker,
    get_device,
    get_input_size,
    step,
    average_over_splits,
    WINDOW_LENGTH,
)


def cnn_tuning(experiment_name: str,
               experiment_params: dict,
               training_params: dict,
               dataset_params: dict,
               output_dir: Path,
               progress_reporter: ProgressReporter | None = None) -> None:
    """Perform tuning for CNN surface classification.

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
        - 'generalized_classes': whether to use generalized class labels.
    output_dir : Path
        Path where the analysis results will be saved as JSON.
    progress_reporter : ProgressReporter
        Custom training reporting interface.
    """

    print(f"Running experiment: {experiment_name}")
    print(f"Subset: {experiment_params['channels']}, ",
          f"Configurations: {experiment_params['kinematics']}")

    g = set_seed(training_params['seed'])
    device = get_device()
    input_size = get_input_size(experiment_params['channels'])

    # Set up paths.
    data_dir = Path(dataset_params['data_dir'])
    labels_file = Path(dataset_params['labels_file'])

    # Load dataset.
    with open(labels_file, encoding='utf-8') as fp:
        labels = json.load(fp)

    dataset = [(data_dir.joinpath(key + '.csv'), values['surface'])
               for key, values in labels.items()
               if values['kinematics'] in experiment_params['kinematics']]
    x = [run[0] for run in dataset]
    target_classes = [run[1] for run in dataset]

    # Encode target classes.
    lb = LabelBinarizer()

    if dataset_params['generalized_classes']:
        generalized_classes = generalize_classes(target_classes)
        lb.fit(generalized_classes)
        y = lb.transform(generalized_classes)
    else:
        lb.fit(target_classes)
        y = lb.transform(target_classes)

    classes = lb.classes_
    num_classes = len(classes)
    y = y.reshape(-1, num_classes)

    cv_data = CNNTrainingDataset(x,
                                 y,
                                 data_frequency=dataset_params['data_frequency'],
                                 window_length=WINDOW_LENGTH,
                                 subset=experiment_params['channels'])

    history = {}
    criterion = nn.CrossEntropyLoss()
    sss = StratifiedShuffleSplit(n_splits=40, test_size=0.2)
    n_splits = sss.get_n_splits()

    # Initialize progress reporter
    if progress_reporter:
        progress_reporter.set_total_steps(n_splits, training_params['num_epochs'])

    # Initialize progress bar.
    fold_pbar = tqdm(total=n_splits, desc='CV Folds', disable=progress_reporter is not None)

    for i, (training_index, test_index) in enumerate(sss.split(x, y)):
        if progress_reporter:
            progress_reporter.update_fold(i + 1)

        split_train_loss = []
        split_val_loss = []
        split_test_loss = []
        split_f1 = []

        # Initialize the model in each split.
        cnn_model = CNNSurfaceClassifier(input_size=input_size, output_size=num_classes).to(device)
        best_model = cnn_model.state_dict()

        # Initialize optimizer in each split.
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3, eps=1e-6, weight_decay=1e-3)

        # Initialize scheduler in each split.
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        # Initialize early stopping.
        early_stopper = EarlyStopper()

        # Separate hold-out fold.
        train_index, val_index = train_test_split(training_index,
                                                  test_size=0.2,
                                                  stratify=y[training_index])

        train_dataloader = DataLoader(Subset(cv_data, train_index),
                                      batch_size=training_params['batch_size'],
                                      worker_init_fn=seed_worker,
                                      generator=g,
                                      shuffle=True)
        val_dataloader = DataLoader(Subset(cv_data, val_index),
                                    batch_size=training_params['batch_size'],
                                    worker_init_fn=seed_worker,
                                    generator=g)
        test_dataloader = DataLoader(Subset(cv_data, test_index),
                                     batch_size=training_params['batch_size'],
                                     worker_init_fn=seed_worker,
                                     generator=g)

        epoch_pbar = tqdm(total=training_params['num_epochs'],
                          desc=f'Fold {i+1}/{n_splits}',
                          disable=progress_reporter is not None)

        for epoch in range(training_params['num_epochs']):
            idx = 0
            running_train_loss = 0.0
            running_val_loss = 0.0

            cnn_model.train()
            for idx, train_batch in enumerate(train_dataloader):
                train_loss, _ = step(cnn_model, train_batch, criterion,
                                     device, train=True, optimizer=optimizer)
                running_train_loss += train_loss

            avg_train_loss = running_train_loss.detach().cpu().item() / (idx + 1)
            split_train_loss.append(avg_train_loss)
            scheduler.step()

            cnn_model.eval()
            with torch.no_grad():
                for idx, val_batch in enumerate(val_dataloader):
                    val_loss, _ = step(cnn_model, val_batch, criterion, device)
                    running_val_loss += val_loss

            avg_val_loss = running_val_loss.detach().cpu().item() / (idx + 1)
            split_val_loss.append(avg_val_loss)

            if early_stopper.early_stop(avg_val_loss):
                break

            if early_stopper.counter == 0:
                best_model = cnn_model.state_dict()

            # Update progress.
            if progress_reporter:
                progress_reporter.update_epoch(epoch + 1, avg_train_loss, avg_val_loss)
            else:
                epoch_pbar.set_postfix({'Train Loss': f'{avg_train_loss:.2E}',
                                        'Val Loss': f'{avg_val_loss:.2E}',
                                        'F1': f'{split_f1[-1]:.3f}'})
                epoch_pbar.update(1)

            epoch_pbar.close()

        cnn_model.load_state_dict(best_model)
        y_true, y_pred = [], []
        running_test_loss = 0.0

        cnn_model.eval()
        with torch.no_grad():
            for idx, test_batch in enumerate(test_dataloader):
                test_loss, test_outputs = step(cnn_model, test_batch, criterion, device)
                running_test_loss += test_loss

                y_true.extend(torch.argmax(test_batch[1], dim=1).cpu().numpy())
                y_pred.extend(torch.argmax(test_outputs, dim=1).cpu().numpy())

        avg_test_loss = running_test_loss.detach().cpu().item() / (idx + 1)
        split_test_loss.append(avg_test_loss)
        split_f1.append(f1_score(y_true, y_pred, average='weighted'))

        if not progress_reporter:
            fold_pbar.update(1)

        history[i + 1] = classification_report(y_true, y_pred, output_dict=True)

    fold_pbar.close()

    base_filename = '_'.join(['cnn', str(num_classes)]
                             + experiment_params['kinematics']
                             + experiment_params['channels']).lower()

    with open(output_dir.joinpath(f'{base_filename}.json'), 'w', encoding='utf-8') as fp:
        json.dump(history, fp)

    # Average classification reports over splits and save to JSON files.
    average_over_splits(history, base_filename, output_dir)

    if progress_reporter:
        progress_reporter.set_completed()


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
                        default='results/logs/tuning/cnn',
                        type=Path,
                        help='Output directory path.')
    parser.add_argument('--progress-dir',
                        default=None,
                        type=Path,
                        help='Progress directory path (optional).')
    args = parser.parse_args()

    # Setup progress reporter if progress directory is provided
    progress_reporter = None

    if args.progress_dir is not None:
        progress_file = args.progress_dir.joinpath(f'{args.experiment_name}_progress.json')
        progress_reporter = ProgressReporter(progress_file)

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

    cnn_tuning(experiment_name, experiment_params, training_params,
               dataset_params, output_dir, progress_reporter)
    time.sleep(1)


if __name__ == "__main__":
    main()
