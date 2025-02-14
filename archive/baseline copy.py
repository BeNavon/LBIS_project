import os
import glob
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

###############################################################################
# 1. Custom Dataset Class
###############################################################################

class InertialDataset(Dataset):
    def __init__(self, root_dir, sensor='imu', 
                 sequence_length=128, 
                 mode_filter=None,
                 transform=None):
        """
        Args:
            root_dir (str): Path to the root directory containing subject folders.
            sensor (str): Sensor name to load (e.g., 'imu'). The code assumes that
                          files for this sensor are stored in 
                          <subject>/<date>/<mode>/<sensor>/*.mat.
            sequence_length (int): Number of time steps to extract per sample.
            mode_filter (list or None): Optionally, a list of locomotion modes to include.
                (e.g. ['treadmill', 'ramp']). If None, all modes are included.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.sensor = sensor
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Define the mapping from locomotion mode to label.
        self.mode_mapping = {
            'treadmill': 0,
            'levelground': 1,
            'ramp': 2,
            'stair': 3,
            'static': 4
        }
        
        # Use glob to search for all .mat files for the given sensor.
        # Assumes file structure: <root_dir>/<subject>/<date>/<mode>/<sensor>/*.mat
        search_path = os.path.join(self.root_dir, '*', '*', '*', self.sensor, '*.mat')
        self.file_list = glob.glob(search_path, recursive=True)
        if mode_filter is not None:
            self.file_list = [f for f in self.file_list if self._get_mode_from_path(f) in mode_filter]
            
        if len(self.file_list) == 0:
            raise RuntimeError(f'No files found with sensor "{self.sensor}" in {self.root_dir}!')
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Get file path and extract label (locomotion mode) from folder structure.
        file_path = self.file_list[idx]
        mode = self._get_mode_from_path(file_path)
        try:
            label = self.mode_mapping[mode]
        except KeyError:
            raise ValueError(f"Unknown mode '{mode}' found in path: {file_path}")

        # Load sensor data from the .mat file.
        data = self._load_mat_file(file_path)
        
        # (Optional) You may want to apply additional synchronization here using the
        # timestamp information if you plan to combine sensors.
        
        # Ensure data is a 2D numpy array of shape (num_samples, num_channels)
        # (Assume that if a header column is present it is the first column.)
        if data.ndim != 2:
            raise ValueError(f"Data loaded from {file_path} is not 2D.")
            
        # If the number of columns is one more than expected (i.e. header + channels),
        # drop the first column. Otherwise, assume data has only sensor channels.
        # (You can adjust this logic as needed.)
        if data.shape[1] > 1:
            # Here we assume the first column is the header (timestamps).
            data = data[:, 1:]
            
        # data now should be of shape (num_samples, num_channels).
        num_samples = data.shape[0]
        
        # If the trial is longer than sequence_length, randomly crop a contiguous segment.
        # If it is shorter, pad with zeros at the end.
        if num_samples >= self.sequence_length:
            start_idx = random.randint(0, num_samples - self.sequence_length)
            data_window = data[start_idx:start_idx + self.sequence_length, :]
        else:
            pad_length = self.sequence_length - num_samples
            data_window = np.pad(data, ((0, pad_length), (0, 0)), mode='constant')
        
        # Optionally apply a transform
        if self.transform:
            data_window = self.transform(data_window)
        else:
            # Convert to float32 tensor
            data_window = torch.tensor(data_window, dtype=torch.float32)
        
        # Return sample and label as a tensor.
        return data_window, torch.tensor(label, dtype=torch.long)
    
    def _get_mode_from_path(self, file_path):
        """
        Extracts the locomotion mode from the file path.
        Assumes the folder structure: <subject>/<date>/<mode>/<sensor>/<file>
        so that the mode is the third folder from the root.
        """
        parts = os.path.normpath(file_path).split(os.sep)
        # Example: ['AB09', '10_21_2018', 'treadmill', 'imu', 'imu_01_01.mat']
        # The mode is at index -3.
        if len(parts) < 5:
            raise ValueError(f"Unexpected file path structure: {file_path}")
        return parts[-3].lower()
    
    def _load_mat_file(self, file_path):
        """
        Loads a MATLAB file and returns a 2D numpy array containing the sensor data.
        Assumes the .mat file contains a table with a 'Header' column plus one or more
        sensor channel columns. The method below shows two strategies:
          1. If the .mat file contains a structured array with named fields, we drop the
             'Header' field.
          2. Otherwise, if the file is stored as a numeric array (with header as first column),
             we simply drop the first column.
        You may need to modify this function to match your file format exactly.
        """
        mat_contents = sio.loadmat(file_path)
        # Exclude MATLAB metadata keys.
        data_keys = [k for k in mat_contents.keys() if not k.startswith('__')]
        if len(data_keys) == 0:
            raise ValueError(f"No data found in {file_path}.")
        
        # If the sensor name appears as a key, use it.
        if self.sensor in data_keys:
            data = mat_contents[self.sensor]
        else:
            # Otherwise, use the first non-metadata key.
            data = mat_contents[data_keys[0]]
            
        # Check if data is a structured array (has named fields).
        if hasattr(data, 'dtype') and data.dtype.names is not None:
            # Convert structured array to a numpy array by concatenating fields (except 'Header').
            field_names = data.dtype.names
            data_list = []
            for field in field_names:
                if field.lower() == 'header':
                    continue
                # data[field] will have shape (N, 1) or (N,)
                field_data = np.array(data[field]).squeeze()
                data_list.append(field_data)
            # Stack columns to get shape (N, num_channels)
            data = np.stack(data_list, axis=1)
        else:
            # Otherwise, assume data is numeric.
            # If more than one column, drop the first column assuming it is the header.
            if data.ndim == 2 and data.shape[1] > 1:
                data = data[:, 1:]
            # Else, leave as is.
        
        return data

###############################################################################
# 2. Baseline Network Definition
###############################################################################

class BaselineInertialNet(nn.Module):
    def __init__(self, num_channels, num_classes=5, sequence_length=128):
        super(BaselineInertialNet, self).__init__()
        # Convolutional layers to extract temporal features.
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Calculate the length of the sequence after the conv and pooling layers.
        def calc_output_length(length, kernel_size, pool_size):
            conv_length = length - (kernel_size - 1)
            return conv_length // pool_size
        
        out_length = calc_output_length(sequence_length, 3, 2)  # after first block
        out_length = calc_output_length(out_length, 3, 2)         # after second block
        
        flattened_size = 64 * out_length
        
        # Fully connected layers for classification.
        self.fc1 = nn.Linear(flattened_size, 100)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(100, num_classes)
    
    def forward(self, x):
        # x is expected to have shape: (batch_size, sequence_length, num_channels)
        # Rearrange to (batch_size, num_channels, sequence_length) for Conv1d.
        x = x.transpose(1, 2)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten.
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

###############################################################################
# 3. Example Usage
###############################################################################

if __name__ == "__main__":
    # Set parameters.
    root_directory = "/dataset"  # Update this to the root of your data.
    sensor_to_use = "imu"                     # You can change this to any available sensor.
    sequence_length = 128
    batch_size = 8

    # Create dataset. (Optionally, you can filter for specific modes using mode_filter.)
    dataset = InertialDataset(root_dir=root_directory, sensor=sensor_to_use, 
                              sequence_length=sequence_length, mode_filter=None)
    
    # For example, get one sample to determine the number of channels.
    sample_data, sample_label = dataset[0]
    # sample_data shape should be (sequence_length, num_channels)
    num_channels = sample_data.shape[1]
    print(f"Sample data shape: {sample_data.shape}, label: {sample_label.item()}")
    
    # Create DataLoader.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Instantiate the baseline network.
    num_classes = 5  # Based on our mode mapping.
    model = BaselineInertialNet(num_channels=num_channels, num_classes=num_classes, 
                                sequence_length=sequence_length)
    
    # Print model summary.
    print(model)
    
    # Example forward pass with one batch.
    for batch_data, batch_labels in dataloader:
        # batch_data: (batch_size, sequence_length, num_channels)
        outputs = model(batch_data)
        print("Output shape:", outputs.shape)  # Expected: (batch_size, num_classes)
        break  # Just one batch for demonstration.
