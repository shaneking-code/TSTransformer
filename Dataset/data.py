import numpy as np
import xarray as xr
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler

filepath = './Dataset/MissouriUpstream.nc'

with xr.open_dataset(filepath, group='Reach_Timeseries') as ds:

    width             = np.array(ds.W)
    average_area      = np.array(ds.A)
    surface_elevation = np.array(ds.H)
    surface_slop      = np.array(ds.S)
    discharge         = np.array(ds.Q)

data = np.stack(discharge, axis=1)[0]

# Split into train and test (70/30 split)

train_data, test_data = data[:-179].reshape(-1,1), data[-179:].reshape(-1,1)

# Preprocess using a scaler if scaling = True

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data).flatten().tolist()
test_data = scaler.fit_transform(test_data).flatten().tolist()

# Method to split into sequences

"""
    CODE CITATION:
        Title: SRC/TGT Sequence Split
        Author: Jeff Heaton
        Link: https://github.com/jeffheaton/app_deep_learning/blob/884ba74e722fd63931f0a5e283bc9e2ad25d116d//t81_558_class_10_3_transformer_timeseries.ipynb
"""

def split_data(sequence_length, dataset, n_input_feats):

    src = []
    tgt = []

    for i in range(len(dataset) - sequence_length):

        src.append(dataset[i:(i+sequence_length)])
        tgt.append(dataset[i+sequence_length])

    src = np.array(src)
    tgt = np.array(tgt)

    return torch.tensor(src, dtype=torch.float32).view(-1, sequence_length, n_input_feats), torch.tensor(tgt, dtype=torch.float32).view(-1, n_input_feats)

"""
    END CITED CODE
"""

# Splitting data

src_train_data, tgt_train_data = split_data(30, train_data, 1)
src_test_data, tgt_test_data   = split_data(30, test_data, 1)

# To tensor datasets

train_data_tensor = TensorDataset(src_train_data, tgt_train_data)
test_data_tensor  = TensorDataset(src_test_data, tgt_test_data)