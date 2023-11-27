from Dataset.data import train_data_tensor
from torch.utils.data import DataLoader
from timeseries_transformer import TimeSeriesTransformer
import torch.nn as nn
import torch.optim as optim
import torch

# Hyperparameters
batch_size        = 50
n_input_feats     = 4
d_model           = 512
n_heads           = 8
n_predicted_feats = 1
n_layers_enc      = 6
n_layers_dec      = 6
d_feedforward_enc = 2048
d_feedforward_dec = 2048
dropout_pos_enc   = 0.1
dropout_enc       = 0.1
dropout_dec       = 0.1

model = TimeSeriesTransformer(
    n_input_feats=n_input_feats,
    d_model=d_model,
    n_heads=n_heads,
    n_predicted_feats=n_predicted_feats,
    n_layers_enc=n_layers_enc,
    n_layers_dec=n_layers_dec,
    d_feedforward_enc=d_feedforward_enc,
    d_feedforward_dec=d_feedforward_dec,
    dropout_pos_enc=dropout_pos_enc,
    dropout_enc=dropout_enc,
    dropout_dec=dropout_dec
)

# Load training data into DataLoader
train_loader = DataLoader(train_data_tensor, batch_size=batch_size)

# Set up loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of epochs
epochs = 10

for epoch in range(epochs):

    # Move to device
    model.to(device)
    
    # Put into training mode
    model.train()

    # Running loss
    training_loss = 0

    for i, (src, tgt) in enumerate(train_loader):

        # Move to device
        src = src.to(torch.float32).to(device)
        tgt = tgt.to(torch.float32).to(device)

        # Generate masks
        src_mask = nn.Transformer.generate_square_subsequent_mask(sz=len(src))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz=len(tgt))

        # Reset gradients
        optimizer.zero_grad()

        # Generate targets to run loss on
        tst_output = model(src, tgt, src_mask, tgt_mask)

        # Calculate loss, backpropagate, and add to running loss
        loss = criterion(tst_output, tgt)
        loss.backward()
        training_loss += loss.item()

        # Step forward
        optimizer.step()

    print(f"Epoch: {epoch + 1} Loss: {training_loss / len(train_loader.dataset)}")

# Save the model weights
torch.save(model.state_dict(), "model_weights.pth")