#%% Imports and seed
import os
import glob
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Set random seed for reproducibility
torch.manual_seed(22)
np.random.seed(22)
random.seed(22)

#%% Custom Dataset for Right-Leg Gait Phase Estimation (CSV version)
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
            # Filter files based on subject IDs
            filtered_files = []
            for subject in subjects:
                subject_path = os.path.join(root_dir, subject, '*', 'treadmill', 'imu', '*.csv')
                filtered_files.extend(glob.glob(subject_path))
                print(f"Before filtering - Number of files: {len(self.imu_files)}")
                print(f"Filtering for subject: {subject}")
            self.imu_files = filtered_files
            print(f"After filtering - Number of files: {len(self.imu_files)}")
            print("Sample paths:")
            for f in self.imu_files[:2]:  # Print first 2 paths as examples
                print(f"- {f}")
                print(f"  Subject: {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))}")
        if len(self.imu_files) == 0:
            raise RuntimeError("No IMU files found. Please check your dataset directory and folder structure.")

    
    def __len__(self):
        return len(self.imu_files)
    
    def __getitem__(self, idx):
        # Get the IMU CSV file path.
        imu_path = self.imu_files[idx]
        # Derive the corresponding gcRight CSV file path by replacing 'imu' with 'gcRight'
        gcRight_path = imu_path.replace(os.sep + 'imu' + os.sep, os.sep + 'gcRight' + os.sep)
        
        # Load CSV files (skip the header row)
        imu_data = self._load_csv_file(imu_path)
        gcRight_data = self._load_csv_file(gcRight_path)
        
        # Drop the timestamp column (first column)
        imu_data = imu_data[:, 1:]
        gcRight_data = gcRight_data[:, 1:]
        
        # Select only shank and thigh channels from IMU data.
        # CSV column order (after dropping timestamp) is:
        # [foot_Accel_X, foot_Accel_Y, foot_Accel_Z,
        #  foot_Gyro_X, foot_Gyro_Y, foot_Gyro_Z,
        #  shank_Accel_X, shank_Accel_Y, shank_Accel_Z,
        #  shank_Gyro_X, shank_Gyro_Y, shank_Gyro_Z,
        #  thigh_Accel_X, thigh_Accel_Y, thigh_Accel_Z,
        #  thigh_Gyro_X, thigh_Gyro_Y, thigh_Gyro_Z,
        #  trunk_Accel_X, trunk_Accel_Y, trunk_Accel_Z,
        #  trunk_Gyro_X, trunk_Gyro_Y, trunk_Gyro_Z]
        # We keep shank (columns 6 to 11) and thigh (columns 12 to 17)
        shank = imu_data[:, 6:12]
        thigh = imu_data[:, 12:18]
        imu_selected = np.concatenate([shank, thigh], axis=1)  # Shape: (N, 12)
        
        # Synchronize lengths: truncate all signals to the minimum available length.
        min_length = min(imu_selected.shape[0], gcRight_data.shape[0])
        imu_selected = imu_selected[:min_length, :]
        gcRight_data = gcRight_data[:min_length, :]
        
        # Randomly extract a window of fixed length.
        if min_length > self.sequence_length:
            start_idx = random.randint(0, min_length - self.sequence_length)
        else:
            start_idx = 0  # Alternatively, pad shorter sequences.
        end_idx = start_idx + self.sequence_length
        imu_window = imu_selected[start_idx:end_idx, :]  # (sequence_length, 12)
        
        # Use the HeelStrike value from gcRight at the center of the window.
        center_idx = start_idx + self.sequence_length // 2
        heel_strike = gcRight_data[center_idx, 0]  # HeelStrike value (0-100)
        # Normalize to [0, 1]
        heel_strike_norm = heel_strike / 100.0
        target = np.array([heel_strike_norm], dtype=np.float32)
        
        # Optionally apply a transform; otherwise, convert to torch tensors.
        if self.transform:
            imu_window = self.transform(imu_window)
        else:
            imu_window = torch.tensor(imu_window, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        
        return imu_window, target

    def _load_csv_file(self, file_path):
        """Loads a CSV file using NumPy (skipping the header row)."""
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        return data

#%% Baseline CNN Model for Gait Phase Estimation (Regression)
class BaselineGaitPhaseCNN(nn.Module):
    def __init__(self, num_channels=12, sequence_length=128, output_dim=1, dropout_rate=0.5):
        """
        Args:
            num_channels (int): Number of input channels (here, 12 from shank+thigh).
            sequence_length (int): Length of the input window.
            output_dim (int): Dimension of the regression output (1 value: normalized HeelStrike).
            dropout_rate (float): Dropout probability.
        """
        super(BaselineGaitPhaseCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Calculate the output length after two conv+pool layers.
        def calc_out_length(L, kernel_size, pool_size):
            conv_L = L - (kernel_size - 1)
            return conv_L // pool_size
        
        out_length = calc_out_length(sequence_length, 3, 2)  # after first block
        out_length = calc_out_length(out_length, 3, 2)         # after second block
        flattened_size = 64 * out_length
        
        self.fc1 = nn.Linear(flattened_size, 100)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(100, output_dim)  # Linear output for regression
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_channels)
        x = x.transpose(1, 2)  # (batch_size, num_channels, sequence_length)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#%% Training and Evaluation Functions (Regression)
def train_model(model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * data.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_history.append(epoch_val_loss)
        
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_weights)
    return model, train_loss_history, val_loss_history

def test_model(model, device, test_loader, criterion):
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            running_test_loss += loss.item() * data.size(0)
    test_loss = running_test_loss / len(test_loader.dataset)
    return test_loss

#%% Main: One-Subject-Out Cross-Validation Setup and Training
# Define dataset root directory
dataset_root = r"data"  # e.g., "C:\path\to\dataset" or "./dataset"

# Get list of subject folders
all_subjects = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
all_subjects.sort()
print("Subjects found:", all_subjects)

# perform one subject-out split. #TODO: # loop over all subjects
test_subject = all_subjects[0]
train_subjects = all_subjects[1:]
print(f"\nTraining on subjects: {train_subjects}")
print(f"Testing on subject: {test_subject}")

# Hyperparameters
sequence_length = 128
batch_size = 128
num_epochs = 30
learning_rate = 1e-3
dropout_rate = 0.5

# Create training and validation datasets (here we split train further into train/val)
train_dataset_full = GaitPhaseDataset(root_dir=dataset_root, sequence_length=sequence_length, subjects=train_subjects)
# Split training data into training and validation (e.g., 80/20 split)
num_train = int(0.8 * len(train_dataset_full))
num_val = len(train_dataset_full) - num_train
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset_full, [num_train, num_val])

# Create test dataset (from test_subject)
test_dataset = GaitPhaseDataset(root_dir=dataset_root, sequence_length=sequence_length, subjects=[test_subject])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, optimizer, and scheduler
model = BaselineGaitPhaseCNN(num_channels=12, sequence_length=sequence_length, output_dim=1, dropout_rate=dropout_rate)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on device:", device)

# Train the model
model, train_loss_hist, val_loss_hist = train_model(model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs)

# Plot training and validation loss
plt.figure(figsize=(10,5))
plt.plot(train_loss_hist, label='Train Loss')
plt.plot(val_loss_hist, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate on test set
test_loss = test_model(model, device, test_loader, criterion)
print(f"Test Loss (MSE): {test_loss:.4f}")
