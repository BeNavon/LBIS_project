## Imports and seed
import os
import glob
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split

# Set random seed for reproducibility
torch.manual_seed(22)
np.random.seed(22)
random.seed(22)

## TensorDataset Conversion for NPY Files
path_to_load = r'npys/'  # Update this with your npy directory
train_data_tot_np = np.load(r'{}X_train.npy'.format(path_to_load))
train_labels_tot_np = np.load(r'{}y_train.npy'.format(path_to_load))
val_data_tot_np = np.load(r'{}X_train.npy'.format(path_to_load))
val_labels_tot_np = np.load(r'{}y_train.npy'.format(path_to_load))
test_data_tot_np = np.load(r'{}X_test.npy'.format(path_to_load))
test_labels_tot_np = np.load(r'{}y_test.npy'.format(path_to_load))

# The labels are sinusoidal representations: [ sin(2*pi*phi/100), cos(2*pi*phi/100) ].
# Convert labels to float32.
train_dataset = TensorDataset(torch.tensor(train_data_tot_np, dtype=torch.float32),
                              torch.tensor(train_labels_tot_np, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(val_data_tot_np, dtype=torch.float32),
                              torch.tensor(val_labels_tot_np, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(test_data_tot_np, dtype=torch.float32),
                              torch.tensor(test_labels_tot_np, dtype=torch.float32))

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

## Baseline CNN Model for Gait Phase Estimation (Regression)
class BaselineGaitPhaseCNN(nn.Module):
    def __init__(
        self,
        num_channels=12,      # Number of input channels (e.g., 12 for shank+thigh IMU)
        sequence_length=200,  # Updated window length: 200 samples (matches your npy data)
        output_dim=2,         # Output dimension: 2 for [sin, cos]
        conv_filters=[32, 64],# Filters for each conv block
        kernel_size=3,        # Kernel size for all conv layers
        stride=1,             # Stride for all conv layers
        padding=0,            # Padding for all conv layers
        dilation=1,           # Dilation for all conv layers
        pool_size=2,          # Max-pooling factor
        hidden_units=100,     # Units in the first fully connected layer
        dropout_rate=0.5,     # Dropout probability
        activation='relu'     # Activation: 'relu', 'sigmoid', or 'tanh'
    ):
        super(BaselineGaitPhaseCNN, self).__init__()
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.activation_choice = activation.lower()
        
        def get_activation_fn(act):
            if act == 'relu':
                return nn.ReLU()
            elif act == 'sigmoid':
                return nn.Sigmoid()
            elif act == 'tanh':
                return nn.Tanh()
            else:
                raise ValueError("Unsupported activation. Choose 'relu', 'sigmoid', or 'tanh'.")
        
        act_fn = get_activation_fn(self.activation_choice)
        self.conv_blocks = nn.ModuleList()
        in_channels = num_channels
        L = sequence_length

        def conv_output_length(L_in, kernel_size, stride, padding, dilation):
            return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        for out_channels in conv_filters:
            block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          dilation=dilation),
                nn.BatchNorm1d(out_channels),
                act_fn,
                nn.MaxPool1d(pool_size)
            )
            self.conv_blocks.append(block)
            L_conv = conv_output_length(L, kernel_size, stride, padding, dilation)
            L_pool = L_conv // pool_size
            L = L_pool
            in_channels = out_channels
        
        flattened_size = in_channels * L
        
        self.fc1 = nn.Linear(flattened_size, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_units, output_dim)
        self.fc_activation = get_activation_fn(self.activation_choice)
        
    def forward(self, x):
        # Check if input is in [batch, sequence_length, channels]
        # If so, transpose to [batch, channels, sequence_length]
        if x.dim() == 3 and x.shape[1] == self.sequence_length:
            x = x.transpose(1, 2)
        # Otherwise, assume input is already channels-first.
        for block in self.conv_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

## Training and Evaluation Functions (with Early Stopping)
def train_model(model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience=5):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    epochs_since_improvement = 0
    
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
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            
        if epochs_since_improvement >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

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

## Main: One-Subject Evaluation
if __name__ == "__main__":
    # For one-subject evaluation, we use the npy-based TensorDataset created above.
    # Hyperparameters
    sequence_length = 200  # Must match the window size of your preprocessed npy data
    batch_size = 128
    num_epochs = 30
    learning_rate = 1e-3
    dropout_rate = 0.5
    patience = 5  # Early stopping patience

    hyperparams = {
        "optimizer": "adam",   # Options: "adam", "sgd"
        "loss_function": "mse"   # Options: "mse", "mae", "huber"
    }

    # Instantiate the model with output_dim=2 for sinusoidal representation.
    model = BaselineGaitPhaseCNN(num_channels=12, sequence_length=sequence_length, output_dim=2, dropout_rate=dropout_rate)
    
    # Choose the loss function
    loss_fn = hyperparams["loss_function"].lower()
    if loss_fn == "mse":
        criterion = nn.MSELoss()
    elif loss_fn == "mae":
        criterion = nn.L1Loss()
    elif loss_fn == "huber":
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError("Unsupported loss function. Choose 'mse', 'mae', or 'huber'.")

    # Choose the optimizer
    optim_choice = hyperparams["optimizer"].lower()
    if optim_choice == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optim_choice == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer. Choose 'adam' or 'sgd'.")

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on device:", device)

    # DataLoaders (already created above)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the model with early stopping
    model, train_loss_hist, val_loss_hist = train_model(
        model, device, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience=patience
    )

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_hist, label='Train Loss')
    plt.plot(val_loss_hist, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate on test set
    test_loss = test_model(model, device, test_loader, criterion)
    print(f"Test Loss (MSE): {test_loss:.4f}")
