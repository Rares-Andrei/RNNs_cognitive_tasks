import torch.nn as nn
import torch


class BaseRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, output_size=1):
        """Initialize simple RNN model.

        Args:
            input_size (int): Number of elements in the input sequence
            hidden_size (int): Number of units in recurrent layer
            output_size (int): Number of output units
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        # Store hyperparameters of the network
        self.config = {"input_size": input_size,
                       "output_size": output_size,
                       "hidden_size": hidden_size}

    def recurrence(self, x: torch.Tensor, hidden_state: torch.Tensor):
        """
        Compute hidden state activations based on input plus
        previous hidden state. Each step processed sequentially.

        Args:
            x: Input of shape (batch_size, input_size)
            hidden_state: Shape (batch_size, self.hidden_size)

        Return:
            new_state: Shape (batch_size, self.hidden_size)

        """
        new_state = torch.tanh(self.i2h(x) + self.h2h(hidden_state))
        return new_state

    def init_hidden(self, batch_size: int):
        """
        Initialize hidden state for the first run.

        Args:
            batch_size: Size of batch.

        Return:
            Zero tensor of shape (batch_size, self.hidden_size)
        """
        return torch.zeros(batch_size, self.hidden_size)

    def forward(self, x: torch.Tensor, hidden_state=None):
        """
        Defines the forward pass of the model

        Args:
            x (torch.Tensor): Shape (seq_len, batch_size, input_size)
            hidden_state (torch.Tensor): Previous hidden state

        Returns:
            output (torch.Tensor): Shape (seq_len, batch_size, output_size)
            hidden_state (torch.Tensor): Last (batch_size, self.hidden_size)
        """
        output = []
        if hidden_state is None:
            hidden_state = self.init_hidden(x.shape[1]).to(x.device)
        steps = range(x.size(0))
        for step in steps:
            hidden_state = self.recurrence(x[step], hidden_state)
            output.append(hidden_state)
        output = torch.stack(
            output,
            dim=0
        )  # (seq_len, batch_size, hidden_size)
        output = self.h2o(output)
        return output, hidden_state
