import numpy as np
import xarray as xr
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler

filepath = '~/Desktop/TST/Dataset/MissouriUpstream.nc'

with xr.open_dataset(filepath, group='Reach_Timeseries') as ds:

    width             = np.array(ds.W)
    average_area      = np.array(ds.A)
    surface_elevation = np.array(ds.H)
    surface_slop      = np.array(ds.S)
    discharge         = np.array(ds.Q)

# data = np.stack((width, average_area, surface_elevation, surface_slop, discharge), axis=1)
data = np.stack(discharge, axis=1)[0]

# Get only the first observation of each set of observations
# data = np.array([datum[:,0] for datum in data])

# Split into train and test
train_data, test_data = data[:-179].reshape(-1,1), data[-179:].reshape(-1,1)

# Preprocess using a scaler if scaling = True
scaling = True
if scaling:
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data).flatten().tolist()
    test_data = scaler.fit_transform(test_data).flatten().tolist()

# Method to split into sequences
def split_data(sequence_length, dataset, n_input_feats):

    X = []
    y = []

    for i in range(len(dataset) - sequence_length):

        X.append(dataset[i:(i+sequence_length)])
        y.append(dataset[i+sequence_length])

    X = np.array(X)
    y = np.array(y)

    return torch.tensor(X, dtype=torch.float64).view(-1, sequence_length, n_input_feats), torch.tensor(y, dtype=torch.float64).view(-1, n_input_feats)

# Splitting data
X_train_data, y_train_data = split_data(30, train_data, 1)
X_test_data, y_test_data   = split_data(30, test_data, 1)

# To tensor datasets
train_data_tensor = TensorDataset(X_train_data, y_train_data)
test_data_tensor  = TensorDataset(X_test_data, y_test_data)