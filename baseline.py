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
# 1. Dataset Definition: Treadmill Gait Phase Estimation from Thigh & Shank IMUs
###############################################################################

class TreadmillGaitDataset(Dataset):
    def __init__(self, root_dir, sequence_length=128, subject_filter=None, transform=None):
        """
        Args:
            root_dir (str): Path to the folder containing all .mat files.
            sequence_length (int): Fixed number of time steps per sample.
            subject_filter (list or None): If provided, only include files whose
                subject (extracted from the filename) is in this list.
            transform (callable, optional): Optional transform to apply on the data.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        
        # List all .mat files in the folder.
        self.file_list = glob.glob(os.path.join(self.root_dir, '*.mat'))
        if len(self.file_list) == 0:
            raise RuntimeError(f"No .mat files found in {self.root_dir}!")
        
        # If a subject filter is provided, restrict the files to those subjects.
        if subject_filter is not None:
            self.file_list = [f for f in self.file_list 
                               if self._get_subject_from_path(f) in subject_filter]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Get the file path and subject id.
        file_path = self.file_list[idx]
        subject_id = self._get_subject_from_path(file_path)
        
        # Load sensor data and gait phase from the .mat file.
        sensor_data_full, gait_phase = self._load_mat_file(file_path)
        # sensor_data_full is expected to be shape (N, 25)
        if sensor_data_full.ndim != 2 or sensor_data_full.shape[1] != 25:
            raise ValueError(f"Unexpected sensor data shape {sensor_data_full.shape} in file: {file_path}")
        
        # Drop the header (first column) so that sensor data becomes (N, 24).
        sensor_data_full = sensor_data_full[:, 1:]
        # Now, select only the shank and thigh channels.
        # After dropping header: columns 0-5: foot, 6-11: shank, 12-17: thigh, 18-23: trunk.
        # We select columns 6:18 to obtain shank (6 columns) and thigh (6 columns) = 12 channels.
        sensor_data = sensor_data_full[:, 6:18]  # shape (N, 12)
        
        num_samples = sensor_data.shape[0]
        # Determine a window of fixed length.
        if num_samples >= self.sequence_length:
            # Randomly select a start index so that the window fits in the trial.
            start_idx = random.randint(0, num_samples - self.sequence_length)
            window_data = sensor_data[start_idx:start_idx + self.sequence_length, :]
            # Use the gait phase at the last time step in the window as the target.
            window_label = gait_phase[start_idx + self.sequence_length - 1]
        else:
            # If too short, pad sensor data with zeros at the end.
            pad_length = self.sequence_length - num_samples
            window_data = np.pad(sensor_data, ((0, pad_length), (0, 0)), mode='constant')
            # Use the last available gait phase as the target.
            window_label = gait_phase[-1]
        
        if self.transform:
            window_data = self.transform(window_data)
        else:
            window_data = torch.tensor(window_data, dtype=torch.float32)
        
        # For regression, label is a float tensor (you could also normalize it as needed).
        window_label = torch.tensor(window_label, dtype=torch.float32)
        
        # Return the sensor window, the gait phase label, and the subject id.
        return window_data, window_label, subject_id
    
    def _get_subject_from_path(self, file_path):
        """
        Extract the subject id from the file name.
        Assumes the file name is formatted like "AB09_treadmill_trial1.mat"
        where "AB09" is the subject id.
        """
        base = os.path.basename(file_path)  # e.g., "AB09_treadmill_trial1.mat"
        subject_id = base.split('_')[0]
        return subject_id.strip()
    
    def _load_mat_file(self, file_path):
        """
        Loads a MATLAB file and returns a tuple: (sensor_data, gait_phase)
        - sensor_data: A numeric array of shape (N, 25) (including header)
        - gait_phase: A numeric vector of length N.
        
        Assumes that the .mat file contains two variables: one containing the sensor
        data (the first non-metadata variable) and one named "gait_phase".
        """
        mat_contents = sio.loadmat(file_path)
        # Exclude MATLAB metadata keys.
        data_keys = [k for k in mat_contents.keys() if not k.startswith('__')]
        if 'gait_phase' not in data_keys:
            raise ValueError(f"'gait_phase' variable not found in {file_path}.")
        
        # Extract gait_phase.
        gait_phase = np.squeeze(mat_contents['gait_phase'])
        # Remove gait_phase from the keys to identify the sensor data variable.
        data_keys.remove('gait_phase')
        if len(data_keys) == 0:
            raise ValueError(f"No sensor data found in {file_path}.")
        # Use the first remaining key as sensor data.
        sensor_data = np.array(mat_contents[data_keys[0]])
        return sensor_data, gait_phase

###############################################################################
# 2. Baseline Neural Network for Gait Phase Regression
###############################################################################

class BaselineGaitPhaseNet(nn.Module):
    def __init__(self, num_channels=12, sequence_length=128):
        """
        Args:
            num_channels (int): Number of sensor channels (12 for shank+thigh).
            sequence_length (int): Number of time steps per input sample.
        """
        super(BaselineGaitPhaseNet, self).__init__()
        
        # Two 1D convolutional blocks.
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool  = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn2   = nn.BatchNorm1d(64)
        
        # Calculate the length of the sequence after two conv+pool blocks.
        def calc_output_length(L, kernel_size, pool_size):
            conv_L = L - (kernel_size - 1)
            return conv_L // pool_size
        
        out_length = calc_output_length(sequence_length, 3, 2)  # After first block.
        out_length = calc_output_length(out_length, 3, 2)         # After second block.
        flattened_size = 64 * out_length
        
        # Fully connected layers for regression.
        self.fc1 = nn.Linear(flattened_size, 100)
        self.dropout = nn.Dropout(p=0.5)
        # Final output: one continuous value (the estimated gait phase).
        self.fc2 = nn.Linear(100, 1)
    
    def forward(self, x):
        # x: (batch_size, sequence_length, num_channels)
        # Rearrange to (batch_size, num_channels, sequence_length) for Conv1d.
        x = x.transpose(1, 2)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten.
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No activation; regression output.
        return x.squeeze(1)  # Return shape (batch_size,)

###############################################################################
# 3. Example Usage with One-Subject-Out Cross Validation
###############################################################################

if __name__ == "__main__":
    # Parameters.
    root_directory = "data"  # Folder containing the .mat files.
    sequence_length = 128
    batch_size = 8
    
    # Gather all .mat files and determine all subject IDs.
    all_files = glob.glob(os.path.join(root_directory, '*.mat'))
    def get_subject(file_path):
        base = os.path.basename(file_path)
        return base.split('_')[1].strip()
    all_subjects = sorted(list(set(get_subject(f) for f in all_files)))
    print("All subjects in dataset:", all_subjects)
    
    # One-Subject-Out Cross Validation:
    # For demonstration, we leave out the first subject as the test subject.
    test_subject = all_subjects[0]
    train_subjects = [sub for sub in all_subjects if sub != test_subject]
    print(f"Test subject: {test_subject}")
    print(f"Training subjects: {train_subjects}")
    
    # Create datasets.
    train_dataset = TreadmillGaitDataset(root_dir=root_directory, 
                                           sequence_length=sequence_length,
                                           subject_filter=train_subjects)
    test_dataset = TreadmillGaitDataset(root_dir=root_directory, 
                                          sequence_length=sequence_length,
                                          subject_filter=[test_subject])
    
    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Instantiate the network.
    model = BaselineGaitPhaseNet(num_channels=12, sequence_length=sequence_length)
    print(model)
    
    # For demonstration, run one forward pass on a batch from the training loader.
    for batch_data, batch_labels, batch_subjects in train_loader:
        # batch_data: shape (batch_size, sequence_length, 12)
        # batch_labels: shape (batch_size,)
        outputs = model(batch_data)
        print("Input batch shape:", batch_data.shape)
        print("Output (estimated gait phase) shape:", outputs.shape)
        print("Gait phase labels:", batch_labels)
        break

    # In a full training pipeline, you would define an optimizer and loss function:
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.MSELoss()
    # Then, iterate over train_loader for training and use test_loader for validation.
