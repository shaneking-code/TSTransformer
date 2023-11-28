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
                 n_layers_dec,
                 d_feedforward_enc,
                 d_feedforward_dec,
                 dropout_pos_enc,
                 dropout_enc,
                 dropout_dec):
        
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
            num_layers=n_layers_enc
        )

        # Decoding pipeline
        self.decoder_input_layer = nn.Linear(n_predicted_feats, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward_dec,
            dropout=dropout_dec,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layers_dec
        )

        self.linear_mapping = nn.Linear(d_model, n_predicted_feats)

    def encode(self, src):

        # Encoding application
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src)

        return src

    def decode(self, tgt, src, tgt_mask, src_mask):

        # Decoding application
        tgt = self.decoder_input_layer(tgt)
        tgt = tgt.unsqueeze(0)
        tgt = self.decoder(tgt=tgt,
                           memory=src,
                           tgt_mask=tgt_mask,
                           memory_mask=src_mask)
        
        tgt = self.linear_mapping(tgt)

        return tgt

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        
        src = self.encode(src)
        tgt = self.decode(tgt, src, tgt_mask, src_mask)

        return tgt
    

