import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import random
import pandas as pd  # needed for _load_as_text below

def _load_as_text(file_path):
    """
    Fallback function to load a file as a text table using pandas.
    
    This function uses a regex separator (splitting on one or more tabs or commas),
    skips the header row, and uses on_bad_lines='skip' to ignore malformed lines.
    It first attempts to read with UTF-8 encoding; if that fails, it falls back to latin1.
    After reading, it converts all columns to numeric (non-numeric values become NaN)
    and fills missing values before converting to a float32 NumPy array.
    """
    try:
        # Attempt with UTF-8 encoding.
        df = pd.read_csv(
            file_path,
            sep=r'[\t,]+',
            engine='python',
            skiprows=1,      # Skip header row
            header=None,
            encoding='utf-8',
            on_bad_lines='skip'
        )
    except UnicodeDecodeError as e:
        print(f"Unicode decode error for {file_path} with utf-8: {e}. Trying latin1 encoding.")
        try:
            df = pd.read_csv(
                file_path,
                sep=r'[\t,]+',
                engine='python',
                skiprows=1,
                header=None,
                encoding='latin1',
                on_bad_lines='skip'
            )
        except Exception as e2:
            raise ValueError(f"Failed to load {file_path} as text with pandas using latin1: {e2}")
    except Exception as e:
        raise ValueError(f"Failed to load {file_path} as text with pandas: {e}")
    
    # Convert every column to numeric, coercing errors to NaN.
    df = df.apply(pd.to_numeric, errors='coerce')
    # Replace NaN values with 0 (or another appropriate value).
    df = df.fillna(0)
    
    try:
        data = df.to_numpy(dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Could not convert data from {file_path} to float32: {e}")
    
    return data


class InertialDataset(Dataset):
    def __init__(self, root_dir, sensor='imu', sequence_length=128,
                 mode_filter=None, transform=None, expected_channels=None):
        """
        Args:
            root_dir (str): Root folder (e.g., "dataset").
            sensor (str): Sensor name (e.g., 'imu').
            sequence_length (int): Number of time steps per sample.
            mode_filter (list or None): List of locomotion modes to include (e.g. ["levelground"]).
            transform (callable, optional): Optional transform to apply.
            expected_channels (int, optional): Force all returned samples to have this many channels.
                If a sample has fewer columns, it is padded with zeros; if more, it is trimmed.
        """
        self.root_dir = root_dir
        self.sensor = sensor
        self.sequence_length = sequence_length
        self.transform = transform
        self.expected_channels = expected_channels  # e.g., 4 or 24
        
        # Mapping for locomotion mode labels.
        self.mode_mapping = {
            'treadmill': 0,
            'levelground': 1,
            'ramp': 2,
            'stair': 3,
            'static': 4
        }
        
        # Search for files: dataset/AB*/<date>/levelground/<sensor>/*.mat
        search_path = os.path.join(self.root_dir, "AB*", "*", "levelground", self.sensor, "*.mat")
        all_files = glob.glob(search_path, recursive=True)
        
        # Exclude files from any "osimxml" folder.
        self.file_list = [f for f in all_files if "osimxml" not in f.lower()]
        
        # Apply mode filter if provided.
        if mode_filter is not None:
            self.file_list = [f for f in self.file_list if self._get_mode_from_path(f) in mode_filter]
        
        if len(self.file_list) == 0:
            raise RuntimeError(f'No files found with sensor "{self.sensor}" in {self.root_dir}!')
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        mode = self._get_mode_from_path(file_path)
        try:
            label = self.mode_mapping[mode]
        except KeyError:
            raise ValueError(f"Unknown mode '{mode}' found in path: {file_path}")
        
        # Load sensor data.
        data = self._load_mat_file(file_path)
        
        # Enforce fixed channel dimension if expected_channels is set.
        if self.expected_channels is not None:
            current_channels = data.shape[1]
            if current_channels < self.expected_channels:
                pad_width = self.expected_channels - current_channels
                data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
            elif current_channels > self.expected_channels:
                data = data[:, :self.expected_channels]
        
        # Ensure data is 2D.
        if data.ndim != 2:
            raise ValueError(f"Data loaded from {file_path} is not 2D.")
        
        num_samples = data.shape[0]
        # Crop (or pad in time) to the desired sequence length.
        if num_samples >= self.sequence_length:
            start_idx = random.randint(0, num_samples - self.sequence_length)
            data_window = data[start_idx:start_idx + self.sequence_length, :]
        else:
            pad_length = self.sequence_length - num_samples
            data_window = np.pad(data, ((0, pad_length), (0, 0)), mode='constant')
        
        # Apply any transform, then convert to tensor.
        if self.transform:
            data_window = self.transform(data_window)
        else:
            data_window = torch.tensor(data_window, dtype=torch.float32)
        
        return data_window, torch.tensor(label, dtype=torch.long)
    
    def _get_mode_from_path(self, file_path):
        parts = os.path.normpath(file_path).split(os.sep)
        if len(parts) < 4:
            raise ValueError(f"Unexpected file path structure: {file_path}")
        return parts[-3].lower()
    
    def _load_mat_file(self, file_path):
        """
        Tries to load the file using loadmat; if no numeric fields are found,
        falls back to text reading via _load_as_text.
        """
        try:
            mat_contents = sio.loadmat(file_path)
        except Exception as e:
            print(f"Failed to load {file_path} as a MAT file ({e}), trying as text.")
            return _load_as_text(file_path)
        
        data_keys = [k for k in mat_contents.keys() if not k.startswith('__')]
        if len(data_keys) == 0:
            raise ValueError(f"No data found in {file_path}.")
        
        if self.sensor in data_keys:
            data = mat_contents[self.sensor]
        else:
            data = mat_contents[data_keys[0]]
            
        if hasattr(data, 'dtype') and data.dtype.names is not None:
            field_names = data.dtype.names
            numeric_fields = []
            for field in field_names:
                if field.lower() == 'header':
                    continue
                field_data = np.atleast_1d(np.array(data[field]).squeeze())
                try:
                    field_data_float = field_data.astype(np.float32)
                    numeric_fields.append(field_data_float)
                except Exception as e:
                    try:
                        field_data_float = np.array([float(x) for x in field_data])
                        field_data_float = field_data_float.astype(np.float32)
                        numeric_fields.append(field_data_float)
                    except Exception as e2:
                        print(f"Skipping field {field} in {file_path}: cannot convert to float ({e2})")
                        continue
            
            if len(numeric_fields) == 0:
                print(f"No numeric fields found in structured array for {file_path}. Falling back to text reading.")
                return _load_as_text(file_path)
            
            min_length = min(arr.shape[0] for arr in numeric_fields)
            numeric_fields = [arr[:min_length] for arr in numeric_fields]
            data = np.stack(numeric_fields, axis=1)
        else:
            if data.dtype == np.object_:
                print(f"Data in {file_path} is of object type, falling back to text reading.")
                return _load_as_text(file_path)
            if data.ndim == 2 and data.shape[1] > 1:
                data = data[:, 1:]
                
        return np.asarray(data, dtype=np.float32)

# Example Usage:
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    root_directory = "dataset"  # Update path as needed.
    sensor_to_use = "imu"
    sequence_length = 128
    batch_size = 8
    expected_channels = 4  # <-- Set this to the number you want all samples to have.

    dataset = InertialDataset(root_dir=root_directory,
                              sensor=sensor_to_use,
                              sequence_length=sequence_length,
                              mode_filter=["levelground"],
                              expected_channels=expected_channels)

    # Print info for one sample.
    sample_data, sample_label = dataset[0]
    print(f"Sample data shape: {sample_data.shape}, Label: {sample_label.item()}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch_data, batch_labels in dataloader:
        print("Batch data shape:", batch_data.shape)  # Should be (batch_size, sequence_length, expected_channels)
        print("Batch labels:", batch_labels)
        break


import matplotlib.pyplot as plt

# --- Visualize a single sample from the dataset ---
# Get one sample (data and label)
sample_data, sample_label = dataset[0]
# sample_data is a torch.Tensor of shape [128, 4]. Convert it to a NumPy array.
sample_np = sample_data.numpy()

# Create a time axis (for example, one point per time step)
time = np.arange(sample_np.shape[0])

# Create subplots: one for each channel
num_channels = sample_np.shape[1]
fig, axs = plt.subplots(num_channels, 1, figsize=(12, 2*num_channels), sharex=True)

# If there's only one channel, axs might not be a list; ensure it's a list:
if num_channels == 1:
    axs = [axs]

for i in range(num_channels):
    axs[i].plot(time, sample_np[:, i], label=f'Channel {i+1}')
    axs[i].set_ylabel(f'Ch {i+1}')
    axs[i].legend(loc='upper right')
    
axs[-1].set_xlabel('Time step')
plt.suptitle(f'Sample Label: {sample_label.item()}')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Visualize a batch of samples (optional) ---
# Get one batch from the DataLoader
batch_data, batch_labels = next(iter(dataloader))
# batch_data is of shape [batch_size, sequence_length, channels]
batch_np = batch_data.numpy()

# Plot the first sample in the batch as an example.
sample_idx = 0
sample_np = batch_np[sample_idx]
time = np.arange(sample_np.shape[0])
fig, axs = plt.subplots(num_channels, 1, figsize=(12, 2*num_channels), sharex=True)
if num_channels == 1:
    axs = [axs]

for i in range(num_channels):
    axs[i].plot(time, sample_np[:, i], label=f'Channel {i+1}')
    axs[i].set_ylabel(f'Ch {i+1}')
    axs[i].legend(loc='upper right')

axs[-1].set_xlabel('Time step')
plt.suptitle(f'Batch Sample {sample_idx}, Label: {batch_labels[sample_idx].item()}')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
