"""Helper objects used in training and inference of surface prediction models."""

import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from utils.visualization import T_95


WINDOW_LENGTH = 200  # Number of samples in each window.
# XGBoost hyperparameter space
PARAM_GRID = {'n_estimators': [100, 200, 300],
              'learning_rate': [0.01, 0.1, 0.3],
              'max_depth': [3, 4, 5],
              'subsample': [0.6, 0.8, 1.0],
              'colsample_bytree': [0.6, 0.8, 1.0],
              'min_child_weight': [1, 3, 5],
              'reg_alpha': [0.1, 0.5],
              'reg_lambda': [0.1, 0.5]}
THRESHOLD = 1e-3


class EarlyStopper:
    """Helper class for early stopping during training of neural networks."""
    def __init__(self, patience: int = 10, min_delta: float = 1e-5) -> None:
        """
        Parameters
        ----------
        patience : int, default=10
            Number of epochs with no improvement after which training will be stopped.
        min_delta : float, default=1e-5
            Minimum change in the monitored quantity to qualify as an improvement.

        Attributes
        ----------
        counter : int
            Counter for epochs without improvement.
        min_validation_loss : float
            Minimum validation loss observed so far.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float) -> bool:
        """Check if training should be stopped based on validation loss.

        Parameters
        ----------
        validation_loss : float
            Current validation loss to compare with the minimum observed loss.

        Returns
        -------
        bool
            True if training should be stopped, False otherwise.
        """

        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ProgressReporter:
    """Report progress to file for monitoring by launcher."""

    def __init__(self, progress_file):
        self.progress_file = Path(progress_file)
        self.progress_data = {
            'total_steps': 0,
            'current_step': 0,
            'status': 'initializing',
            'current_fold': 0,
            'total_folds': 0,
            'current_epoch': 0,
            'total_epochs': 0,
            'train_loss': 0.0,
            'val_loss': 0.0,
        }
        self._write_progress()

    def _write_progress(self):
        """Write current progress to file."""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f)
        except Exception:
            # If we can't write progress, don't crash the training
            pass

    def set_total_steps(self, total_folds, total_epochs):
        """Set total steps for progress tracking."""
        self.progress_data['total_steps'] = total_folds * total_epochs
        self.progress_data['total_folds'] = total_folds
        self.progress_data['total_epochs'] = total_epochs
        self.progress_data['status'] = 'training'
        self._write_progress()

    def update_fold(self, current_fold):
        """Update current fold."""
        self.progress_data['current_fold'] = current_fold
        self._write_progress()

    def update_epoch(self, current_epoch, train_loss=None, val_loss=None):
        """Update current epoch and losses."""
        self.progress_data['current_epoch'] = current_epoch
        self.progress_data['current_step'] = ((self.progress_data['current_fold'] - 1) *
                                              self.progress_data['total_epochs'] + current_epoch)

        if train_loss is not None:
            self.progress_data['train_loss'] = float(train_loss)
        if val_loss is not None:
            self.progress_data['val_loss'] = float(val_loss)

        self._write_progress()

    def set_completed(self):
        """Mark as completed."""
        self.progress_data['status'] = 'completed'
        self.progress_data['current_step'] = self.progress_data['total_steps']
        self._write_progress()

    def set_failed(self):
        """Mark as failed."""
        self.progress_data['status'] = 'failed'
        self._write_progress()


class ProgressTracker:
    """Track progress of multiple experiments using file-based communication."""

    def __init__(self, progress_dir, experiment_names):
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_names = experiment_names
        self.progress_bars = {}
        self.stop_monitoring = False
        self.monitor_thread = None

        # Create progress files for each experiment
        for exp_name in experiment_names:
            progress_file = self.progress_dir / f"{exp_name}_progress.json"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_steps': 0,
                    'current_step': 0,
                    'status': 'initializing',
                    'current_fold': 0,
                    'total_folds': 0,
                    'current_epoch': 0,
                    'total_epochs': 0,
                    'train_loss': 0.0,
                    'val_loss': 0.0
                }, f)

    def start_monitoring(self):
        """Start monitoring progress in a separate thread."""
        self.monitor_thread = threading.Thread(target=self._monitor_progress, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring_func(self):
        """Stop monitoring progress."""
        self.stop_monitoring = True
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)

        # Close all progress bars
        for pbar in self.progress_bars.values():
            pbar.close()

    def _monitor_progress(self):
        """Monitor progress files and update progress bars."""
        while not self.stop_monitoring:
            for exp_name in self.experiment_names:
                progress_file = self.progress_dir / f"{exp_name}_progress.json"

                try:
                    if progress_file.exists():
                        with open(progress_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # Create progress bar if it doesn't exist
                        if exp_name not in self.progress_bars:
                            if data['total_steps'] > 0:
                                self.progress_bars[exp_name] = tqdm(
                                    total=data['total_steps'],
                                    desc=f"{exp_name}",
                                    position=len(self.progress_bars),
                                    leave=True
                                )

                        # Update existing progress bar
                        elif exp_name in self.progress_bars:
                            pbar = self.progress_bars[exp_name]

                            # Update total if it changed
                            if pbar.total != data['total_steps'] and data['total_steps'] > 0:
                                pbar.total = data['total_steps']
                                pbar.refresh()

                            # Update progress
                            if data['current_step'] > pbar.n:
                                pbar.update(data['current_step'] - pbar.n)

                            # Update description with current status
                            status_info = []
                            if data['status'] == 'training':
                                status_info.append(f"Fold {data['current_fold']}/{data['total_folds']}")
                                status_info.append(f"Epoch {data['current_epoch']}/{data['total_epochs']}")
                                if data['train_loss'] > 0:
                                    status_info.append(f"Loss: {data['train_loss']:.2E}")
                            elif data['status'] == 'completed':
                                status_info.append("✅ COMPLETED")
                            elif data['status'] == 'failed':
                                status_info.append("❌ FAILED")

                            desc = f"{exp_name}"
                            if status_info:
                                desc += f" - {' | '.join(status_info)}"

                            pbar.set_description(desc)

                except (json.JSONDecodeError, FileNotFoundError, KeyError):
                    # Skip if file is being written or doesn't exist yet
                    continue

            time.sleep(0.5)  # Update every 500ms


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.

    Parameters
    ----------
    config_path : pathlib.Path
        Path to the configuration file located in the 'configs' directory.

    Returns
    -------
    config : dict
        Configuration parameters as a dictionary.
    """

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def extract_experiment_name(experiment_params: dict) -> str:
    """Extract a unique name for the experiment based on its parameters.

    Parameters
    ----------
    experiment_params : dict
        Dictionary containing experiment parameters.

    Returns
    -------
    str
        Unique experiment name.
    """
    return '_'.join(experiment_params['kinematics'] + experiment_params['channels']).lower()


def set_seed(seed: int) -> torch.Generator:
    """Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value to set for random number generators.

    Returns
    -------
    generator : torch.Generator
        A torch Generator object initialized with the given seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id) -> None:
    """Set the seed for worker processes to ensure reproducibility."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> str:
    """Get the device to run the model on.

    Returns
    -------
    str
        The device to use for computations (CPU or GPU).
    """
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def get_input_size(channels: list) -> int:
    """Get the input size based on the subset of modalities.

    Parameters
    ----------
    channels : list of str
        List of input modalities, e.g., ['imu', 'servo'].

    Returns
    -------
    int
        The total input size based on the selected modalities.
    """

    if channels is None:
        return 0

    size_map = {'imu': 6, 'servo': 2}
    return sum(size_map[input] for input in channels)


def run_training_instance(script_path: Path,
                          config: str,
                          experiment_name: str,
                          output_dir: Path,
                          progress_dir: Path | None = None) -> dict:
    """Run a single training instance with the given configuration.

    Parameters
    ----------
    script_path : pathlib.Path
        Path to the training script to execute.
    config : str
        Configuration file name.
    experiment_name : str
        Name of the experiment.
    output_dir : pathlib.Path
        Directory to save the output results.
    progress_dir : pathlib.Path, optional
        Directory to save progress files for monitoring.

    Returns
    -------
    dict
        A dictionary containing the configuration, experiment name, return code, and success status.
    """

    # Build the command
    cmd = [sys.executable,
           script_path,
           '--config',
           config,
           '--experiment-name',
           experiment_name,
           '--output-dir',
           output_dir]

    if progress_dir is not None:
        cmd += ['--progress-dir', progress_dir]

    print(f"Starting training with config: {experiment_name}")
    process = subprocess.run(cmd, check=False, text=True)

    return {
        'config': config,
        'experiment_name': experiment_name,
        'returncode': process.returncode,
        'success': process.returncode == 0
    }


def step(model: torch.nn.Module, batch: tuple,
         criterion: torch.nn.Module, device: torch.device,
         train: bool = False, optimizer: torch.optim.Optimizer | None = None) -> tuple:
    """Perform a single training or validation step.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train or validate.
    batch : tuple
        A tuple containing input features and target labels.
    criterion : torch.nn.Module
        Loss function to compute the loss.
    device : torch.device
        Device to perform computations on (CPU or GPU).
    train : bool, default=False
        If True, perform a training step; if False, perform a validation step.
    optimizer : torch.optim.Optimizer, optional
        Optimizer for updating model parameters during training.

    Returns
    -------
    tuple
        Loss value and model outputs.
    """

    batch_x, batch_y = batch
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    batch_x = batch_x.permute(0, 2, 1)
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)

    if train:
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss, outputs


def average_over_splits(results: dict, filename: str, output_dir: Path) -> None:
    """Averages tuning classification reports over splits.

    Parameters
    ----------
    results : dict
        Dictionary containing classification reports for each split.
    filename : str
        Base filename for saving the averaged reports.
    output_dir : pathlib.Path
        Directory to save the output JSON files.
    """

    # Update classification reports.
    for result_dict in results.values():
        result_dict.update({'accuracy': {'precision': None,
                                         'recall': None,
                                         'f1-score': result_dict['accuracy'],
                                         'support': result_dict['macro avg']['support']}})

    # Load reports to DataFrames.
    reports = [pd.DataFrame(result_dict).transpose() for result_dict in results.values()]

    # Calculate statistics
    result_arrays = np.array([report.to_numpy()[:, :3] for report in reports])
    df_mean, df_ci = reports[0].copy(), reports[0].copy()
    df_mean.iloc[:, :3] = result_arrays.mean(axis=0)
    df_ci.iloc[:, :3] = T_95 * result_arrays.std(axis=0) / np.sqrt(len(reports))

    # Dump statistics to JSON files.
    df_mean.to_json(path_or_buf=output_dir.joinpath(f'{filename}_mean.json'), orient='index')
    df_ci.to_json(path_or_buf=output_dir.joinpath(f'{filename}_ci.json'), orient='index')
