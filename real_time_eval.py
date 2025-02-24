"""
This goal of this script is to plot a real time gait phase estimation using a trained model.
"""
# %%
import numpy as np
import pandas as pd
import os
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
# from baseline import BaselineGaitPhaseCNN
from preprocessing.preprocess import sliding_window_with_label, apply_filter, calc_phase_from_label
import json
import random

# Set random seed for reproducibility
torch.manual_seed(22)
np.random.seed(22)
random.seed(22)

# Load the scenario json file
scenario = 'scenario_1'
with open('preprocessing/preprocess_scenarios.json', 'r') as file:
    preprocess_scenarios = json.load(file)
    is_filter = preprocess_scenarios[scenario]['is_filter']
    filter_type = preprocess_scenarios[scenario]['filter_type']
    cutoff = 25                 # cutoff frequency for the filter (Hz)
    is_normalize = preprocess_scenarios[scenario]['is_normalize']
    window_size = preprocess_scenarios[scenario]['window_size']
    overlap = preprocess_scenarios[scenario]['overlap']
print("{} - filter: {}, type: {}, normalize: {}, window size: {}, overlap: {}".format(scenario, is_filter, filter_type, is_normalize, window_size, overlap))


## Baseline CNN Model for Gait Phase Estimation (Regression)
class BaselineGaitPhaseCNN(nn.Module):
    def __init__(
        self,
        num_channels=12,      # Number of input channels (e.g., 12 for shank+thigh IMU)
        sequence_length=window_size,  # Input time window length
        output_dim=2,         # Regression output dimension now 2 for [sin, cos]
        conv_filters=[32, 64],# Filters for each conv block (2 blocks in this example)
        kernel_size=3,        # Kernel size for all conv layers
        stride=1,             # Stride for all conv layers
        padding=0,            # Padding for all conv layers
        dilation=1,           # Dilation for all conv layers
        pool_size=2,          # Max-pooling factor
        hidden_units=100,     # Units in the first fully connected layer
        dropout_rate=0.5,     # Dropout probability
        activation='relu'     # Activation function: 'relu', 'sigmoid', or 'tanh'
    ):

        super(BaselineGaitPhaseCNN, self).__init__()
        
        self.num_channels = num_channels
        self.activation_choice = activation.lower()
        self.conv_blocks = nn.ModuleList()
        
        # Track current number of channels and current sequence length.
        in_channels = num_channels
        L = sequence_length

        # Function to calculate output length after a 1D convolution (with dilation)
        def conv_output_length(L_in, kernel_size, stride, padding, dilation):
            return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        # Helper function to choose an activation function
        def get_activation_fn(act):
            if act == 'relu':
                return nn.ReLU()
            elif act == 'sigmoid':
                return nn.Sigmoid()
            elif act == 'tanh':
                return nn.Tanh()
            else:
                raise ValueError("Unsupported activation function. Choose 'relu', 'sigmoid', or 'tanh'.")
        
        act_fn = get_activation_fn(self.activation_choice)
        
        # Build each convolutional block
        for out_channels in conv_filters:
            # Convolution + BatchNorm + Activation + MaxPool
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
            
            # Update sequence length after convolution and pooling
            L_conv = conv_output_length(L, kernel_size, stride, padding, dilation)
            L_pool = L_conv // pool_size
            L = L_pool
            
            # Update for next block
            in_channels = out_channels
        
        # Flattened size after all convolutional blocks
        flattened_size = in_channels * L
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_units, output_dim)
        
        # Activation for FC layers using the same activation choice
        self.fc_activation = get_activation_fn(self.activation_choice)
        
    def forward(self, x):

        # print("Input shape:", x.shape)
        
        # If the printed shape shows (batch_size, 12, window_size), then no rearrangement is necessary. 
        # However, if you see (batch_size, window_size, 12), then you would need to transpose x using:
        # x = x.transpose(1, 2)
        
        # Pass through each convolutional block
        for block in self.conv_blocks:
            x = block(x)
        
        # Flatten the features using torch.flatten
        x = torch.flatten(x, start_dim=1)
        
        # Fully connected layers
        x = self.fc_activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) 
        # x = F.normalize(x, p=2, dim=1) # Normalize the output to unit length        
        return x
    

# %%
def calculate_delta_phi_vector(y1,y2,y1_hat,y2_hat):
    # Ensure inputs are NumPy arrays (in case they are lists)
    y1, y2, y1_hat, y2_hat = map(np.asarray, [y1, y2, y1_hat, y2_hat])
    # Compute angles using element-wise operations
    delta_phi = np.arctan2(y2_hat * y1 - y1_hat * y2, y1_hat * y1 + y2_hat * y2)
    delta_phi = delta_phi / (2 * np.pi)
    return delta_phi  # Returns a NumPy array

def calculate_sRMSE(y_pred,y_actual):
    # this code uses the assumption that y_pred and y_actual are in the dimensions: [num_examples,2]
    y1 = y_actual[:,0]
    y2 = y_actual[:,1]
    y1_hat = y_pred[:,0]
    y2_hat = y_pred[:,1]
    # Ensure inputs are NumPy arrays (in case they are lists)
    y1, y2, y1_hat, y2_hat = map(np.asarray, [y1, y2, y1_hat, y2_hat])
    # compute delta_phi:
    delta_phi_values = calculate_delta_phi_vector(y1, y2, y1_hat, y2_hat)
    # calculate the sRMSE :
    sRMSE = np.sqrt(np.power(delta_phi_values, 2).mean())
    return sRMSE

def get_y_pred_and_actual_on_validation(model, device, val_loader):
    model.eval()  # Set model to evaluation mode
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            # Move data to CPU and convert to NumPy
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    y_pred = np.vstack(all_preds)
    y_actual = np.vstack(all_targets)
    return y_pred, y_actual

def draw_angles(y_pred,y_actual):
    angles_pred = np.arctan2(y_pred[:, 1], y_pred[:, 0]) / (2 * np.pi)
    angles_actual = np.arctan2(y_actual[:, 1], y_actual[:, 0]) / (2 * np.pi)
    # make sure that the phases are between 0 and 1
    angles_pred = (angles_pred + 1) % 1
    angles_actual = (angles_actual + 1) % 1
    # Scatter plot of actual vs. predicted angles
    plt.figure(figsize=(6, 6))
    plt.scatter(angles_actual, angles_pred, alpha=0.5, edgecolors='k')
    # Reference line (perfect predictions)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
    # Labels and title
    plt.xlabel("Actual Angles (cycles)")
    plt.ylabel("Predicted Angles (cycles)")
    plt.title("Predicted vs. Actual Angles")
    plt.legend()
    plt.grid(True)
    # Show the plot
    plt.show()

def compute_loss(model, data_loader, loss_type, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0

    # Choose the correct loss function
    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "mae":
        criterion = nn.L1Loss()
    elif loss_type == "huber":
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError("Unsupported loss type. Choose 'mse', 'mae', or 'huber'.")

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * data.size(0)

    return total_loss / len(data_loader.dataset)



# Load the test mean and std for normalization (if needed)
mean_std = np.load('preprocessing/mean_std_train.npy')
mean_train = mean_std[0]
std_train = mean_std[1]

# Load the validation file for example
imu_df = pd.read_csv(os.path.join('dataset', 'AB10', 'treadmill', 'imu', 'treadmill_01_01_data.csv'))
gc_df = pd.read_csv(os.path.join('dataset', 'AB10', 'treadmill', 'gcRight', 'treadmill_01_01_data.csv'))

# Preallocate the validation data
X_validation_data = np.empty((0, 12, window_size))
y_validation_data = np.empty((0, 2))

# remove unnecessary columns
gc_df = gc_df.drop(columns=["ToeOff"])
imu_df = imu_df.drop(columns=['foot_Accel_X', 'foot_Accel_Y', 'foot_Accel_Z', 'foot_Gyro_X', 'foot_Gyro_Y', 'foot_Gyro_Z', 'trunk_Accel_X', 'trunk_Accel_Y', 'trunk_Accel_Z', 'trunk_Gyro_X', 'trunk_Gyro_Y', 'trunk_Gyro_Z'])

# remove the first and last samples that have no proper label defined (until the first Heel Strike occurance + after the last Toe Off occurance)
gc_df = gc_df.loc[gc_df.index[gc_df["HeelStrike"].gt(0)].min() : gc_df.index[gc_df["HeelStrike"] == 100].max()]
imu_df = imu_df[imu_df["Header"].isin(gc_df["Header"])] # remove the rows that are not in the gc data

# Apply the cosine and sine functions to the HeelStrike column
gc_df['cos_gait_phase'] = np.cos(gc_df['HeelStrike'] * 2 * np.pi / 100)
gc_df['sin_gait_phase'] = np.sin(gc_df['HeelStrike'] * 2 * np.pi / 100)

# remove header and other columns
gc_df.drop(columns=["Header","HeelStrike"], inplace=True)
imu_df.drop(columns=['Header'], inplace=True)

gc_df.reset_index(drop=True, inplace=True)
imu_df.reset_index(drop=True, inplace=True)

# Apply a filter to the IMU data (choose between a causal and non-causal filter i.e. with phase or zero phase lag filters)
# filter_type = "causal" # or "non-causal" - defined in the scenario parameters settings section
filtered_df = pd.DataFrame(apply_filter(imu_df.values, filter_type='causal', cutoff=25, order=4), columns=imu_df.columns) if is_filter else imu_df

# Normalize the input data (is_normalize = True or False)
filtered_df = (filtered_df - mean_train) / std_train if is_normalize else filtered_df

# Split the data into windows (by window size and overlap)
X_windows, y_labels = sliding_window_with_label(filtered_df, gc_df, window_size=window_size, overlap = window_size-10) # this ensures that the windows will run one by one as they do in real time...

# Concatenate the data to the multidimensional array according to the validation data matrix
X_validation_data = np.concatenate((X_validation_data, X_windows), axis=0)
y_validation_data = np.concatenate((y_validation_data, y_labels), axis=0)

val_dataset = TensorDataset(torch.tensor(X_validation_data, dtype=torch.float32),
                              torch.tensor(y_validation_data, dtype=torch.float32))


# Set a device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the checkpoint
checkpoint = torch.load('model_checkpoint.pth')

# Reinitialize the model with the saved parameters
model_params = checkpoint['model_params']
eval_model = BaselineGaitPhaseCNN(**model_params)

# # Align the keys in the state dictionary
# new_state_dict = {}
# for key, value in checkpoint['model_state_dict'].items():
#     new_key = key.replace('conv_blocks.0.', 'conv1.').replace('conv_blocks.1.', 'conv2.')
#     new_state_dict[new_key] = value

# # Load the aligned state dictionary into the model
# eval_model.load_state_dict(new_state_dict)
eval_model.eval()

batch_size = 128 #TODO: Set batch size.
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

with torch.no_grad():
    for data, targets in val_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = eval_model(data)
# %% Plot the outputs and targets on the same plot as true gait phase and prediction

plt.figure()
for i in range(2):
    plt.subplot(2,1,i+1)
    plt.plot(targets.detach().numpy()[100:4000,i])
    plt.plot(outputs.detach().numpy()[100:4000,i])
    plt.legend(['True Gait Phase', 'Predicted Gait Phase'])
plt.show()

# print(data.detach().numpy())
print(targets.detach().numpy())
print(outputs.detach().numpy())

# %%
# get predicted and actual labels for validation data:
y_pred, y_actual = get_y_pred_and_actual_on_validation(eval_model, device, val_loader)
final_val_loss = compute_loss(eval_model, val_loader, "mse", device)
sRMSE = calculate_sRMSE(y_pred,y_actual)
# Print results
print(f"Recalculated Validation Loss: {final_val_loss:.4f}")
print(f"Validation sRMSE: {100*sRMSE:.4f} %")
draw_angles(y_pred,y_actual)

# %% This can be copied to the baseline model script
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt
from preprocessing.preprocess import sliding_window_with_label, apply_filter
import json

# Load the scenario json file
scenario = 'scenario_1'
with open('preprocessing/preprocess_scenarios.json', 'r') as file:
    preprocess_scenarios = json.load(file)
    is_filter = preprocess_scenarios[scenario]['is_filter']
    filter_type = preprocess_scenarios[scenario]['filter_type']
    cutoff = 25                 # cutoff frequency for the filter (Hz)
    is_normalize = preprocess_scenarios[scenario]['is_normalize']
    window_size = preprocess_scenarios[scenario]['window_size']
    overlap = preprocess_scenarios[scenario]['overlap']
print("{} - filter: {}, type: {}, normalize: {}, window size: {}, overlap: {}".format(scenario, is_filter, filter_type, is_normalize, window_size, overlap))


def draw_real_time_plot(y_pred, y_actual):
    angles_pred = np.arctan2(y_pred[:, 1], y_pred[:, 0]) / (2 * np.pi)
    angles_actual = np.arctan2(y_actual[:, 1], y_actual[:, 0]) / (2 * np.pi)
    # make sure that the phases are between 0 and 1
    angles_pred = (angles_pred + 1) % 1
    angles_actual = (angles_actual + 1) % 1
    plt.figure()
    plt.subplot(3,1,1), plt.grid()
    plt.plot(100*angles_actual[100:4000])
    plt.plot(100*angles_pred[100:4000])
    plt.xticks([100, 1000, 2000, 3000, 4000], ['0.5', '5', '10', '15', '20'])
    plt.ylabel("Gait Phase (%)")
    plt.subplot(3,1,2), plt.grid()
    plt.plot(y_actual[100:4000,0])
    plt.plot(y_pred[100:4000,0])
    plt.xticks([100, 1000, 2000, 3000, 4000], ['0.5', '5', '10', '15', '20'])
    plt.ylabel("Cosine of phase")
    plt.subplot(3,1,3), plt.grid()
    plt.plot(y_actual[100:4000,1])
    plt.plot(y_pred[100:4000,1])
    plt.ylabel("Sin of phase")
    plt.xticks([100, 1000, 2000, 3000, 4000], ['0.5', '5', '10', '15', '20'])
    plt.legend(['True Gait Phase', 'Predicted Gait Phase'], loc='lower center', bbox_to_anchor=(0.5, -0.8), ncol=2)
    plt.xlabel("Time (sec)")
    plt.show()
    pass

def load_validation_for_real_time() -> torch.utils.data.DataLoader:
    # Load the test mean and std for normalization (if needed)
    mean_std = np.load('preprocessing/mean_std_train.npy')
    mean_train = mean_std[0]
    std_train = mean_std[1]

    # Load the validation file for example
    imu_df = pd.read_csv(os.path.join('dataset', 'AB10', 'treadmill', 'imu', 'treadmill_01_01_data.csv'))
    gc_df = pd.read_csv(os.path.join('dataset', 'AB10', 'treadmill', 'gcRight', 'treadmill_01_01_data.csv'))

    # Preallocate the validation data
    X_validation_data = np.empty((0, 12, window_size))
    y_validation_data = np.empty((0, 2))

    # remove unnecessary columns
    gc_df = gc_df.drop(columns=["ToeOff"])
    imu_df = imu_df.drop(columns=['foot_Accel_X', 'foot_Accel_Y', 'foot_Accel_Z', 'foot_Gyro_X', 'foot_Gyro_Y', 'foot_Gyro_Z', 'trunk_Accel_X', 'trunk_Accel_Y', 'trunk_Accel_Z', 'trunk_Gyro_X', 'trunk_Gyro_Y', 'trunk_Gyro_Z'])

    # remove the first and last samples that have no proper label defined (until the first Heel Strike occurance + after the last Toe Off occurance)
    gc_df = gc_df.loc[gc_df.index[gc_df["HeelStrike"].gt(0)].min() : gc_df.index[gc_df["HeelStrike"] == 100].max()]
    imu_df = imu_df[imu_df["Header"].isin(gc_df["Header"])] # remove the rows that are not in the gc data

    # Apply the cosine and sine functions to the HeelStrike column
    gc_df['cos_gait_phase'] = np.cos(gc_df['HeelStrike'] * 2 * np.pi / 100)
    gc_df['sin_gait_phase'] = np.sin(gc_df['HeelStrike'] * 2 * np.pi / 100)

    # remove header and other columns
    gc_df.drop(columns=["Header","HeelStrike"], inplace=True)
    imu_df.drop(columns=['Header'], inplace=True)

    gc_df.reset_index(drop=True, inplace=True)
    imu_df.reset_index(drop=True, inplace=True)

    # Apply a filter to the IMU data (choose between a causal and non-causal filter i.e. with phase or zero phase lag filters)
    # filter_type = "causal" # or "non-causal" - defined in the scenario parameters settings section
    filtered_df = pd.DataFrame(apply_filter(imu_df.values, filter_type='causal', cutoff=25, order=4), columns=imu_df.columns) if is_filter else imu_df

    # Normalize the input data (is_normalize = True or False)
    filtered_df = (filtered_df - mean_train) / std_train if is_normalize else filtered_df

    # Split the data into windows (by window size and overlap)
    X_windows, y_labels = sliding_window_with_label(filtered_df, gc_df, window_size=window_size, overlap = window_size-1) # this ensures that the windows will run one by one as they do in real time...

    # Concatenate the data to the multidimensional array according to the validation data matrix
    X_validation_data = np.concatenate((X_validation_data, X_windows), axis=0)
    y_validation_data = np.concatenate((y_validation_data, y_labels), axis=0)

    val_dataset = TensorDataset(torch.tensor(X_validation_data, dtype=torch.float32),
                                torch.tensor(y_validation_data, dtype=torch.float32))
    batch_size = 128 
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    return val_loader


real_time_val_loader = load_validation_for_real_time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set a device

# get predicted and actual labels for validation data:
y_pred, y_actual = get_y_pred_and_actual_on_validation(model, device, real_time_val_loader)

final_val_loss = compute_loss(model, val_loader, "mse", device)
sRMSE = calculate_sRMSE(y_pred,y_actual)
# Print results
print(f"Recalculated Validation Loss: {final_val_loss:.4f}")
print(f"Validation sRMSE: {100*sRMSE:.4f} %")
draw_angles(y_pred,y_actual)

# Plot the outputs and targets on the same plot as true gait phase and prediction
draw_real_time_plot(y_pred=y_pred, y_actual=y_actual)