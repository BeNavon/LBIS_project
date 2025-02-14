import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

###############################################################################
# 1. Custom Dataset for Right-Leg Gait Phase Estimation (CSV version)
###############################################################################
class GaitPhaseDataset(Dataset):
    def __init__(self, root_dir, sequence_length=128, subjects=None, transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset. Files are expected under:
                dataset/<subject>/<date>/treadmill/imu/*.csv
                and corresponding gait cycle files under:
                dataset/<subject>/<date>/treadmill/gcRight/*.csv.
            sequence_length (int): Number of time steps in each sample window.
            subjects (list of str or None): List of subject IDs (e.g., ['AB09', 'AB10']).
                If None, all subjects are used.
            transform (callable, optional): Optional transform to be applied on the IMU window.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform

        # Locate all treadmill IMU CSV files.
        search_path = os.path.join(root_dir, '*', '*', 'treadmill', 'imu', '*.csv')
        self.imu_files = glob.glob(search_path, recursive=True)
        if subjects is not None:
            # Assume the subject folder is the first folder in the file path.
            self.imu_files = [f for f in self.imu_files if f.split(os.sep)[-5] in subjects]
        
        if len(self.imu_files) == 0:
            raise RuntimeError("No IMU files found. Please check your dataset directory and folder structure.")
    
    def __len__(self):
        return len(self.imu_files)
    
    def __getitem__(self, idx):
        # Get the path to the IMU CSV file.
        imu_path = self.imu_files[idx]
        # Derive the corresponding gcRight CSV file path by replacing the folder name:
        # e.g., .../treadmill/imu/imu_01_01.csv  →  .../treadmill/gcRight/imu_01_01.csv
        gcRight_path = imu_path.replace(os.sep + 'imu' + os.sep, os.sep + 'gcRight' + os.sep)
        
        # Load CSV files (skip header row).
        imu_data = self._load_csv_file(imu_path)
        gcRight_data = self._load_csv_file(gcRight_path)
        
        # Drop the header column (timestamp) from both.
        imu_data = imu_data[:, 1:]
        gcRight_data = gcRight_data[:, 1:]
        
        # Select only shank and thigh channels from the IMU data.
        # After dropping the header, the column ordering is assumed to be:
        # [foot_Accel (3), foot_Gyro (3), shank_Accel (3), shank_Gyro (3),
        #  thigh_Accel (3), thigh_Gyro (3), trunk_Accel (3), trunk_Gyro (3)]
        # We keep shank (columns 6 to 11) and thigh (columns 12 to 17) → 12 channels total.
        shank = imu_data[:, 6:12]
        thigh = imu_data[:, 12:18]
        imu_selected = np.concatenate([shank, thigh], axis=1)  # Shape: (N, 12)
        
        # Synchronize lengths (truncate to the minimum length among files).
        min_length = min(imu_selected.shape[0], gcRight_data.shape[0])
        imu_selected = imu_selected[:min_length, :]
        gcRight_data = gcRight_data[:min_length, :]
        
        # Randomly extract a window of fixed length.
        if min_length > self.sequence_length:
            start_idx = random.randint(0, min_length - self.sequence_length)
        else:
            start_idx = 0  # Alternatively, you could pad sequences that are too short.
        end_idx = start_idx + self.sequence_length
        imu_window = imu_selected[start_idx:end_idx, :]  # Shape: (sequence_length, 12)
        
        # Use the gait phase (HeelStrike value) from gcRight at the center of the window.
        center_idx = start_idx + self.sequence_length // 2
        gait_phase_right = gcRight_data[center_idx, 0]  # HeelStrike value (0–100)
        gait_phase_right = gait_phase_right / 100.0       # Normalize to [0, 1]
        target = np.array([gait_phase_right], dtype=np.float32)
        
        # Optionally apply a transform; otherwise, convert to torch tensors.
        if self.transform:
            imu_window = self.transform(imu_window)
        else:
            imu_window = torch.tensor(imu_window, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        
        return imu_window, target

    def _load_csv_file(self, file_path):
        """
        Loads a CSV file using NumPy, skipping the header row.
        """
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        return data

###############################################################################
# 2. Baseline Convolutional Network for Right-Leg Gait Phase Estimation
###############################################################################
class GaitPhaseEstimator(nn.Module):
    def __init__(self, num_channels=12, sequence_length=128, output_dim=1):
        """
        Args:
            num_channels (int): Number of input channels (here, 12: shank and thigh).
            sequence_length (int): Length of the input window (number of time steps).
            output_dim (int): Dimension of the regression output (here, 1: right-leg gait phase).
        """
        super(GaitPhaseEstimator, self).__init__()
        # Convolutional layers for temporal feature extraction.
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool  = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn2   = nn.BatchNorm1d(64)
        
        # Compute the output sequence length after conv and pooling layers.
        def calc_output_length(L, kernel_size, pool_size):
            conv_L = L - (kernel_size - 1)
            return conv_L // pool_size
        
        out_length = calc_output_length(sequence_length, 3, 2)  # after first block
        out_length = calc_output_length(out_length, 3, 2)         # after second block
        flattened_size = 64 * out_length
        
        # Fully connected layers for regression.
        self.fc1 = nn.Linear(flattened_size, 100)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(100, output_dim)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_channels)
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
# 3. One-Subject-Out Cross-Validation Training & Evaluation
###############################################################################
if __name__ == "__main__":
    # Set device (CPU or GPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- User parameters ---
    root_directory = "data"  # Update with your dataset's root directory.
    sequence_length = 128
    batch_size = 8
    num_epochs = 10  # Adjust number of epochs as needed.
    
    # List all subject folders (assumed to be top-level folders under root_directory).
    all_subjects = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
    all_subjects.sort()
    print("Subjects found:", all_subjects)
    
    # One-subject-out cross-validation.
    for test_subject in all_subjects:
        train_subjects = [s for s in all_subjects if s != test_subject]
        print(f"\n--- One-Subject-Out Split ---")
        print(f"Training on subjects: {train_subjects}")
        print(f"Testing on subject: {test_subject}")
        
        # Create dataset splits.
        train_dataset = GaitPhaseDataset(root_dir=root_directory, sequence_length=sequence_length, subjects=train_subjects)
        test_dataset  = GaitPhaseDataset(root_dir=root_directory, sequence_length=sequence_length, subjects=[test_subject])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize the model.
        model = GaitPhaseEstimator(num_channels=12, sequence_length=sequence_length, output_dim=1)
        model.to(device)
        
        # Define optimizer and loss function (using MSE for regression).
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # --- Training Loop ---
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_data, batch_targets in train_loader:
                batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_data.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{num_epochs} – Training Loss: {epoch_loss:.4f}")
        
        # --- Evaluation on Test Data ---
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_targets in test_loader:
                batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_targets)
                test_loss += loss.item() * batch_data.size(0)
        test_loss /= len(test_loader.dataset)
        print(f"Test Loss for subject {test_subject}: {test_loss:.4f}")
        
        # For demonstration purposes, we run one subject-out split.
        break
