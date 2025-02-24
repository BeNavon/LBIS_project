"""
This script is used to plot a real time gait phase estimation using the trained model.
The script has the following functions:
    - preprocess the test data (using causal filters etc.)
    - forward prediction of the gait phase using a given model. 
    - plot the real time gait phase estimation and the true gait phase on the same plot.
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
import torch
import torch.nn as nn

def prep_data(data, filter_order=2, cutoff=25, fs=100):
    """
    Function to preprocess the data.
    Args:
        data: np.array of shape (n_samples, n_channels)
        filter_order: int, order of the filter
        cutoff: int, cutoff frequency of the filter
        fs: int, sampling frequency
    Returns:
        data: np.array of shape (n_samples, n_channels)
    """
    # causal filter
    b, a = signal.butter(filter_order, cutoff/(fs/2), btype='low', analog=False)
    data = signal.lfilter(b, a, data, axis=0)
    return data

def apply_filter(data, cutoff=25, fs=200, order=4):
    b, a = butter(order, cutoff / (fs / 2), btype='low', analog=False)
    return lfilter(b, a, data, axis=0)

def forward(model, data):
    """
    Function to predict the gait phase using a given model.
    Args:
        model: torch model
        data: np.array of shape (n_samples, n_channels)
    Returns:
        pred: np.array of shape (n_samples, 1)
    """
    data = torch.tensor(data).float()
    pred = model(data)
    return pred.detach().numpy()
