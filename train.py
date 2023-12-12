from Dataset.data import train_data_tensor
from torch.utils.data import DataLoader
from timeseries_transformer import TimeSeriesTransformer
from hyperparameters import *
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

# Instantiate model with hyperparameters
model = TimeSeriesTransformer(
    n_input_feats=n_input_feats,
    d_model=d_model,
    n_heads=n_heads,
    n_predicted_feats=n_predicted_feats,
    n_layers_enc=n_layers_enc,
    d_feedforward_enc=d_feedforward_enc,
    dropout_pos_enc=dropout_pos_enc,
    dropout_enc=dropout_enc,
)

# Load training data into DataLoader
train_loader = DataLoader(train_data_tensor, batch_size=batch_size)

# Set up loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of epochs
epochs = 5

plot_losses = False
if plot_losses:
    losses = []

for epoch in range(epochs):

    # Move to device
    model.to(device)
    
    # Put into training mode
    model.train()

    # Running loss
    training_loss = 0

    for i, (src, tgt) in enumerate(train_loader):

        # Move to device and permute the src to be in form: [batch_size, sequence_length, n_features]
        src = src.to(torch.float32).to(device)
        #src = src.permute(1, 0, 2)
        tgt = tgt.to(torch.float32).to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Generate targets to run loss on
        tst_output = model(src)

        # Calculate loss, backpropagate, and add to running loss
        # Loss is calculated specifically on discharge, not on the other features
        loss = criterion(tst_output, tgt)
        loss.backward()
        training_loss += loss.item()
        # Step forward
        optimizer.step()

    if plot_losses:
        losses.append(training_loss / len(train_loader.dataset))

    print(f"Epoch: {epoch + 1} Loss: {training_loss / len(train_loader.dataset)}")

if plot_losses:
    plt.plot(range(1, len(losses) + 1), losses, color="b")
    plt.xlabel("Epoch")
    plt.ylabel("MSELoss")
    plt.savefig("loss_epochs.png", dpi=300)

# Save the model weights
torch.save(model, "model.pth")