import numpy as np
import xarray as xr
import torch
from torch.utils.data import TensorDataset

filepath = '~/Desktop/TST/Dataset/MissouriUpstream.nc'

with xr.open_dataset(filepath, group='Reach_Timeseries') as ds:

    width             = np.array(ds.W)
    average_area      = np.array(ds.A)
    surface_elevation = np.array(ds.H)
    surface_slop      = np.array(ds.S)
    discharge         = np.array(ds.Q)

data = np.stack((width, average_area, surface_elevation, surface_slop, discharge), axis=1)

# Get only the first observation of each set of observations
data = np.array([datum[:,0] for datum in data])

# Grab the first four columns as inputs
inps = data[:, :-1]

# Grab the last column as the target (discharge)
tgts = data[:, -1]

# Transform the numpy arrays to tensors
inps = torch.from_numpy(inps)
tgts = torch.from_numpy(tgts)

# Split into train and test
train_inps, test_inps = inps[:-179], inps[-179:]
train_tgts, test_tgts = tgts[:-179], tgts[-179:]

# Create tensor datasets
train_data_tensor = TensorDataset(train_inps, train_tgts)
test_data_tensor  = TensorDataset(test_inps, test_tgts)