import torch.nn as nn
import torch


class CTRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20,
                 output_size=1, tau=1000, dt=None,
                 sigma_rec=0.05, sigma_in=0.01,
                 act_fn='softplus'):
        """Initialize simple RNN model.

        Args:
            input_size (int): Number of elements in the input sequence
            hidden_size (int): Number of units in recurrent layer
            output_size (int): Number of output units
            tau (float): Time constant for memory function. Higher tau means longer memory.
            dt: Time step of simulation
            sigma_rec: std of internal noise
            sigma_in: std of input noise
            act_fn: Activation function to be used (tanh or softplus)
        """
        super().__init__()
        assert dt is None or dt < tau, "CTRNN discretization expects dt < tau (so alpha=dt/tau < 1)"
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.tau = tau  # Time constant for memory function
        self.sigma_rec = sigma_rec
        self.sigma_in = sigma_in
        if act_fn == 'softplus':
            self.act_fn = torch.nn.functional.softplus
        elif act_fn == 'tanh':
            self.act_fn = torch.nn.functional.tanh
        self.noise = torch.distributions.normal.Normal(0, 1)
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau  # In neurogym, durations are usually in 1000's for periods, so dt=100 and tau=1000 would give alpha=0.1, which means the network will have a longer memory than a normal RNN (which corresponds to alpha=1).
        self.alpha = alpha
        # Store hyperparameters of the network
        self.config = {"input_size": input_size,
                       "output_size": output_size,
                       "hidden_size": hidden_size,
                       "dt": dt,
                       "tau": tau,
                       "sigma_rec": sigma_rec,
                       "sigma_in": sigma_in,
                       "act_fn": act_fn}

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
        eps = self.noise.sample(hidden_state.size()).to(hidden_state.device)  # N(0,1)
        rec_noise = (2.0 / self.alpha) ** 0.5 * self.sigma_rec * eps
        preact = self.i2h(x) + self.h2h(hidden_state) + rec_noise
        h_tilde = self.act_fn(preact)
        new_state = (1 - self.alpha) * hidden_state + self.alpha * h_tilde
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
            hidden_state (torch.Tensor): Previous hidden state shape (batch_size, self.hidden_size)

        Returns:
            output (torch.Tensor): Shape (seq_len, batch_size, output_size)
            hidden_traj (torch.Tensor): Hidden state trajectory across time, shape (seq_len, batch_size, hidden_size)
        """
        output = []
        if hidden_state is None:
            hidden_state = self.init_hidden(x.shape[1]).to(x.device)
        steps = range(x.size(0))
        for step in steps:
            input_noisy = x[step] + \
                          ((2/self.alpha)**0.5*self.sigma_in) * \
                          self.noise.sample(x[step].size()).to(hidden_state.device)
            hidden_state = self.recurrence(input_noisy, hidden_state)
            output.append(hidden_state)
        hidden_traj = torch.stack(
            output,
            dim=0
        )  # (seq_len, batch_size, hidden_size), hidden state trajectory across time
        output = self.h2o(hidden_traj)
        return output, hidden_traj
