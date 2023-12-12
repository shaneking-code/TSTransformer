import torch
import torch.nn as nn
from positional_encoder import PositionalEncoder

# Following the architecture from https://arxiv.org/abs/2001.08317

class TimeSeriesTransformer(nn.Module):

    def __init__(self,
                 n_input_feats,
                 d_model,
                 n_heads,
                 n_predicted_feats,
                 n_layers_enc,
                 d_feedforward_enc,
                 dropout_pos_enc,
                 dropout_enc):
        
        super(TimeSeriesTransformer, self).__init__()

        self.d_model = d_model

        # Encoding pipeline
        self.encoder_input_layer       = nn.Linear(n_input_feats, d_model)
        self.positional_encoding_layer = PositionalEncoder(d_model, dropout_pos_enc)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward_enc,
            dropout=dropout_enc,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers_enc,
            enable_nested_tensor=False
        )

        # Decoding pipeline

        self.linear_mapping = nn.Linear(d_model, n_predicted_feats)

    def encode(self, src):

        # Encoding application

        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src)

        return src

    def decode(self, tgt):

        # Decoding application

        tgt = self.linear_mapping(tgt)

        return tgt

    def forward(self, src):
        
        src = self.encode(src) # [batch_size, 30, 1] -> [batch_size, 30, model_dimension]
        # Grab the last time step (the output) from each batch_size along the model_dimension
        src = src[:, -1, :] # [batch_size, 30, model dimension] -> [batch_size, model_dimension]
        src = self.decode(src) # [batch_size, model_dimension] -> [batch size, 1]

        return src
    

