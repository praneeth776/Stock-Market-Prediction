# RNN cell

import torch
import torch.nn as nn

class RNNCell(nn.Module):
    """
    A single RNN cell.

    h_t = tanh(W_ih * x_t + b_ih + W_hh * h_(t-1) + b_hh)

    Args:
        input_size (int): The number of expected features in the input `x`.
        hidden_size (int): The number of features in the hidden state `h`.
    """
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # W_ih: Weight matrix for the input to hidden connection.
        # input `x` into a hidden .
        self.W_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        # b_ih: Bias vector for the input to hidden connection.
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))

        # W_hh: Weight matrix for the hidden to hidden connection.
        # `h_prev` to the new hidden state.
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        # b_hh: Bias vector for the hidden to hidden connection.
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input_data, h_prev):
        """
        Performs one forward pass of the RNN cell.

        Args:
            input_data (Tensor): The current input at time step t. Shape: (batch_size, input_size).
            h_prev (Tensor): The hidden state from the previous time step t-1. Shape: (batch_size, hidden_size).

        Returns:
            The new hidden state for the current time step.
        """
        # Input-to-hidden transformation: (x_t @ W_ih.T) + b_ih
        input_transform = torch.matmul(input_data, self.W_ih.T) + self.b_ih

        # Hidden-to-hidden transformation: (h_prev @ W_hh.T) + b_hh
        hidden_transform = torch.matmul(h_prev, self.W_hh.T) + self.b_hh

        # non-linear activation function (tanh).
        h_next = torch.tanh(input_transform + hidden_transform)
        
        return h_next