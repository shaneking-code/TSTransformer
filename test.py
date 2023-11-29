from hyperparameters import batch_size
import torch
from Dataset.data import test_data_tensor, scaler, y_test_data, y_train_data
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

model = torch.load('model.pth')
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

test_loader = DataLoader(test_data_tensor, batch_size=batch_size)

criterion = nn.MSELoss()
losses = []
predictions = []
actuals = []

with torch.no_grad():
    for i, (src, tgt) in enumerate(test_loader):

        src = src.to(torch.float32).to(device)
        #src = src.permute(1, 0, 2)
        tgt = tgt.to(torch.float32).to(device)
        tgt_batch = model(src)
        
        actuals.extend(tgt.squeeze().tolist())
        predictions.extend(tgt_batch.squeeze().tolist())
        losses.append(criterion(tgt_batch, tgt))

inverse_y_train_data = scaler.inverse_transform(np.array(y_train_data).reshape(-1,1))
inverse_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
inverse_y_test_data = scaler.inverse_transform(np.array(y_test_data).reshape(-1,1))
#actuals = scaler.inverse_transform(np.array(actuals).reshape(-1,1))
inverse_y_train_data = inverse_y_train_data.flatten().tolist()
inverse_y_test_data = inverse_y_test_data.flatten().tolist()
inverse_predictions = inverse_predictions.flatten().tolist()
yrange1 = inverse_y_train_data + inverse_y_test_data
yrange2 = inverse_predictions
xrange = list(range(len(yrange1)))
shifted_xrange = range(len(xrange) - len(inverse_predictions), len(xrange))
plt.plot(xrange, yrange1, color="b", alpha=0.2)
plt.plot(shifted_xrange, yrange2, color="g")
plt.savefig("./output.png", dpi=300)

print("Predictions: ", scaler.inverse_transform(np.array(predictions).reshape(-1,1)))
print("Losses: ", losses)
rmse = np.sqrt(np.mean((scaler.inverse_transform(np.array(predictions).reshape(-1, 1)) - scaler.inverse_transform(y_test_data.numpy().reshape(-1, 1)))**2))
normalized_rmse = rmse / (torch.max(y_test_data) - torch.min(y_test_data))
print(f"Score (RMSE): {normalized_rmse:.4f}")
