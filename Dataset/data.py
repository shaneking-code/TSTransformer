import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

filepath = '~/Desktop/TST/Dataset/MissouriUpstream.nc'

with xr.open_dataset(filepath, group='Reach_Timeseries') as ds:
    width             = np.array(ds.W)
    average_area      = np.array(ds.A)
    surface_elevation = np.array(ds.H)
    surface_slop      = np.array(ds.S)
    discharge         = np.array(ds.Q)
    plt.plot(list(range(len(ds.Q))), ds.Q)
    plt.show()

dataset = np.stack((width, average_area, surface_elevation, surface_slop, discharge), axis=1)
dataset = np.array([data[:,0] for data in dataset])
