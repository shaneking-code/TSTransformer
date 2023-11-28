from timeseries_transformer import TimeSeriesTransformer
from hyperparameters import *
import torch
from Dataset.data import test_data_tensor
from torch.utils.data import DataLoader
import torch.nn as nn

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
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

test_loader = DataLoader(test_data_tensor, batch_size=batch_size)

criterion = nn.MSELoss()
losses = []

for i, (src, tgt) in enumerate(test_loader):

    src = src.to(torch.float32).to(device)
    src = src.permute(1, 0, 2)
    tgt = tgt.to(torch.float32).to(device)
    tgt_batch = model(src, tgt)
    tgt = tgt.unsqueeze(0)
    
    print("OUTPUT VALUES: ", tgt_batch.shape)
    print("TARGET VALUES:", tgt.shape)
    losses.append(criterion(tgt_batch, tgt))

print("Losses: ", losses)