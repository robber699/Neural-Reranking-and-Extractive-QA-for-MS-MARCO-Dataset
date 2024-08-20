import math

import torch
from torch import nn
from torch import Tensor

# https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch

# the encoder uses sine and cosine functions of different frequencies
# to generate the positional encoding
class PositionalEncoding(nn.Module):

    """
    @param d_model: dimension of the model's input
    @param max_len: maximum length of the sequence for which positional encodings are pre-computed
    @return: positional encodings for the input sequence
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 180):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # tensor containing the position indices for each position in the sequence
        position = torch.arange(max_len).unsqueeze(1)
        # term used to scale the position indices in a specific way
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        # sine function is applied to the even indices and the cosine function to the odd indices of pe
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # adds the positional encodings to the input x
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)