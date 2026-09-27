"""Module for datasets used in training and inference of surface prediction models."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.processing import get_sample_features, sample_sequence, sequence_run
from utils.training import WINDOW_LENGTH


class BaseDataset(Dataset, ABC):
    """Base class for datasets used in training and inference of surface prediction models.

    This class initializes the dataset parameters such as frequency, sampling rate, lookback period,
    stride, and window length. It also defines the modalities to be used in the dataset.
    """
    def __init__(self, data_frequency: float = 100.0,
                 window_length: int = 200, subset: list | None = None) -> None:
        """
        Parameters
        ----------
        data_frequency : float, default=100.0
            Frequency of collected data [Hz].
        window_length : int, default=200
            Network input length.
        subset : list of str, optional
            List containing data sources to be used in the dataset.

        Attributes
        ----------
        selected_modalities : list of str
            List of selected modalities based on the subset.
        """

        self.data_frequency = data_frequency
        self.window_length = window_length

        if subset is None:
            self.subset = ['imu', 'servo']
        else:
            self.subset = subset

        modalities = {
            'imu': [
                'linear_acceleration.x',
                'linear_acceleration.y',
                'linear_acceleration.z',
                'angular_velocity.x',
                'angular_velocity.y',
                'angular_velocity.z',
            ],
            'servo': [
                'mean_power_left',
                'mean_power_right',
            ],
        }
        self.selected_modalities = []
        for source in self.subset:
            self.selected_modalities.extend(modalities[source])

    @abstractmethod
    def __len__(self) -> int:
        return 0

    @abstractmethod
    def __getitem__(self, idx: int):
        raise IndexError


class CNNTrainingDataset(BaseDataset):
    """Dataset for training CNN."""
    def __init__(self, runs: list | np.ndarray,
                 labels: np.ndarray, data_frequency: float = 100.0,
                 window_length: int = WINDOW_LENGTH, subset: list | None = None) -> None:
        """
        Parameters
        ----------
        runs : list of str or np.ndarray of str
            List or array of time series file names.
        labels : np.ndarray
            ndarray of one-hot encoded surface labels.
        """

        super().__init__(data_frequency, window_length, subset)

        self.runs = runs
        self.labels = labels

    def __len__(self) -> int:
        """Returns the number of runs in the dataset."""
        return len(self.runs)

    def __getitem__(self, idx: int) -> tuple:
        """Retrieves a single, windowed time series from dataset."""

        run = self.runs[idx]
        time_series = pd.read_csv(run, index_col=[0]).drop(labels='Time', axis=1)

        x = sample_sequence(time_series, self.selected_modalities, self.window_length)
        y = self.labels[idx]

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)


class XGBTrainingDataset(CNNTrainingDataset):
    """Dataset for training XGBoost.

    It extends CNNTrainingDataset and includes engineered features for the XGBoost models.
    """
    def __init__(self, runs: list | np.ndarray,
                 labels: np.ndarray, data_frequency: float = 100.0,
                 window_length: int = WINDOW_LENGTH, subset: list | None = None,
                 time_features: list | None = None, freq_features: list | None = None) -> None:
        """
        Parameters
        ----------
        time_features : list of str, optional
            List of time domain features to be extracted from the time series.
        freq_features : list of str, optional
            List of frequency domain features to be extracted from the time series.
        """

        super().__init__(runs, labels, data_frequency, window_length, subset)

        self.time_features = time_features
        self.freq_features = freq_features

    @property
    def engineered_features(self) -> np.ndarray:
        """Returns the list of engineered features (dataframe column names) for XGBoost."""

        t_features = [channel + '_t_' + feature
                      for feature in self.time_features
                      for channel in self.selected_modalities]
        f_features = [channel + '_f_' + feature
                      for feature in self.freq_features
                      for channel in self.selected_modalities]
        return np.array(t_features + f_features)

    def __getitem__(self, idx: int) -> tuple:
        """Overrides __getitem__ to return engineered features for XGBoost."""

        run = self.runs[idx]
        time_series = pd.read_csv(run, index_col=[0]).drop(labels='Time', axis=1)

        x = sample_sequence(time_series, self.selected_modalities, self.window_length)
        x_hat = get_sample_features(x, self.time_features, self.freq_features)
        y = self.labels[idx]

        return x_hat, y


class InferenceDataset(BaseDataset):
    """Dataset for CNN inference on a single recorded run."""
    def __init__(self,
                 run: str,
                 data_frequency: float = 100.0,
                 window_length: int = WINDOW_LENGTH,
                 subset: list | None = None) -> None:
        """
        Parameters
        ----------
        run : str
            Time series file name.
        """

        super().__init__(data_frequency, window_length, subset)

        self.run = pd.read_csv(run, index_col=[0]).drop(labels='Time', axis=1)
        self.windows = sequence_run(self.run, self.selected_modalities, self.window_length)

    def __len__(self) -> int:
        """Returns the number of windows retrieved from a run."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Retrieves a single, windowed time series from dataset."""
        return torch.tensor(self.windows[idx], dtype=torch.float)
