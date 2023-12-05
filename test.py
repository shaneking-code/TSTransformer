from hyperparameters import *
import torch
from Dataset.data import test_data_tensor, scaler, y_test_data, y_train_data
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = torch.load("optmodel.pth")
#model = torch.load('model.pth')
model.eval()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load testing data into DataLoader
test_loader = DataLoader(test_data_tensor, batch_size=batch_size)

# Set up loss
criterion = nn.MSELoss()

losses = []
predictions = []
actuals = []

# Generate predictions
with torch.no_grad():
    for i, (src, tgt) in enumerate(test_loader):

        src = src.to(torch.float32).to(device)
        tgt = tgt.to(torch.float32).to(device)
        tgt_batch = model(src)
        
        actuals.extend(tgt.squeeze().tolist())
        predictions.extend(tgt_batch.squeeze().tolist())
        losses.append(criterion(tgt_batch, tgt))

# Using scaler, map actuals, predictions into unscaled domain
inverse_y_train_data = scaler.inverse_transform(np.array(y_train_data).reshape(-1,1)).flatten().tolist()
inverse_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten().tolist()
inverse_y_test_data = scaler.inverse_transform(np.array(y_test_data).reshape(-1,1)).flatten().tolist()

plot = False
if plot:
    yrange1 = inverse_y_train_data + inverse_y_test_data
    yrange2 = inverse_predictions
    xrange = list(range(len(yrange1)))

    shifted_xrange = range(len(xrange) - len(inverse_predictions), len(xrange))
    plt.plot(xrange, yrange1, color="b", alpha=0.2)
    plt.plot(shifted_xrange, yrange2, color="g")
    plt.savefig("./output.png", dpi=300)

# Nash-Sutcliffe Efficiency
def NSE(simulated, observed):

    num = torch.sum((observed - simulated)**2)
    den = torch.sum((observed - torch.mean(observed))**2)
    
    return 1 - (num / den)

# Normalized Root-Mean-Squared Efficiency
def NRMSE(simulated, observed):

    rmse = torch.sqrt(torch.mean((simulated - observed)**2))
    div = torch.mean(observed)

    return rmse / div

ipred_tensor = torch.tensor(inverse_predictions)
iytest_tensor = torch.tensor(inverse_y_test_data)

nse = NSE(ipred_tensor, iytest_tensor).item()
nrmse = NRMSE(ipred_tensor, iytest_tensor).item()
print("NSE   = ", round(nse, 3))
print("NRMSE = ", round(nrmse, 3))