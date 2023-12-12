from hyperparameters import *
from Dataset.data import scaler
import torch
from Dataset.data import test_data_tensor, scaler, tgt_test_data, tgt_train_data
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
optimal = True
if optimal:
    model = torch.load("optmodel.pth")
else:
    model = torch.load('model.pth')

# Set to eval mode
model.eval()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load testing data into DataLoader
test_loader = DataLoader(test_data_tensor, batch_size=batch_size)

# Set up loss if needed
criterion = nn.MSELoss()

# Set up tracking for predictions and actuals
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

# Using scaler, map actuals, predictions into unscaled domain
inverse_y_train_data = scaler.inverse_transform(np.array(tgt_train_data).reshape(-1,1)).flatten().tolist()
inverse_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten().tolist()
inverse_y_test_data = scaler.inverse_transform(np.array(tgt_test_data).reshape(-1,1)).flatten().tolist()

# Prediction section
tgt = list(test_loader)[-1][-1]
src = list(test_loader)[-1][-2]
tgt = tgt.to(torch.float32)
src = src.to(torch.float32)

# Prepare src by appending tgt[-1] to get the next tgt
src = src[1:]
src = torch.cat((src, src[-1].unsqueeze(0)))
src[-1] = torch.cat((src[-1][1:], tgt[-1].unsqueeze(0)))

# Generate 1st prediction
out = model(src[-1].unsqueeze(0)).squeeze().tolist()
out = [out]

# Generate 2nd prediction
src[-1] = torch.cat((src[-1][1:], torch.tensor(out).unsqueeze(0)))
out2 = model(src[-1].unsqueeze(0)).squeeze().tolist()
out.append(out2)

# Generate third prediction
src[-1] = torch.cat((src[-1][1:], torch.tensor(out[-1]).unsqueeze(0).unsqueeze(0)))
out3 = model(src[-1].unsqueeze(0)).squeeze().tolist()
out.append(out3)

# Scale out back into the correct domain
out = scaler.inverse_transform(np.array(out).reshape(-1,1)).flatten().tolist()

plot = True
if plot:
    yrange1 = inverse_y_train_data + inverse_y_test_data
    yrange2 = inverse_predictions
    xrange = list(range(len(yrange1)))

    out.insert(0, yrange2[-1])

    shifted_xrange = range(len(xrange) - len(inverse_predictions), len(xrange))
    pred_xrange = range(len(xrange) - 1, len(xrange) + len(out) - 1)
    plt.plot(xrange, yrange1, color="b", alpha=0.2, label="True")
    plt.plot(shifted_xrange, yrange2, color="g", label="Predicted")
    plt.plot(pred_xrange, out, color="r", label="Simmed")
    plt.xlabel("Day")
    plt.ylabel("Discharge")
    plt.legend()
    plt.show()
    #plt.savefig("./simmed_output.png", dpi=300)
    #plt.savefig("./output.png", dpi=300)

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
print("NSE   = ", round(nse, 6))
print("NRMSE = ", round(nrmse, 6))