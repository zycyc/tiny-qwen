import torch
import torch.nn as nn


def _rolling_mask(
    input_size: int,
    num_unique: int,
    num_shared: int,
    *,
    device: str | torch.device = "cpu",
):
    assert isinstance(input_size, int) and input_size > 0, (
        "'input_size' must be a positive integer"
    )
    assert isinstance(num_unique, int), "'num_unique' must be an integer"
    assert isinstance(num_shared, int), "'num_shared' must be an integer"
    assert num_unique + num_shared > 0, (
        "'num_unique' + 'num_shared' must be greater than 0"
    )
    wmask = torch.kron(
        torch.eye(input_size, dtype=torch.bool, device=device),
        torch.ones((num_unique + num_shared, 1), dtype=torch.bool, device=device),
    )
    wmask |= wmask.roll(num_shared, 0)
    return wmask


def _random_band_tensor(
    m: int,
    band_reach: int,
    uniform_range: tuple[float, float] = (-1, 1.0),
    *,
    device: str | torch.device = "cpu",
):
    """Creates an m x m tensor with a band of random values, fully vectorized.

    Arguments:
    - m (int): Size of the tensor (m x m).
    - band_reach (int): Reach of the band at the top and left side of the tensor
    - device (str | torch.device): Which device to create the random band tensor on
    - uniform_range (float, float): Range for uniform distribution

    Returns:
    - torch.Tensor: An m x m tensor with the specified band of random values.
    """
    assert isinstance(m, int) and m >= 0, "'m' must be a non-negative integer"
    assert isinstance(band_reach, int) and 0 <= band_reach <= m, (
        "'band_reach' must be a non-negative integer bounded by 'm'"
    )
    tensor = torch.zeros(m, m, device=device)

    i = torch.arange(m, device=device).unsqueeze(0)
    j = torch.arange(m, device=device).unsqueeze(1)

    # Create the mask for the top band
    left_mask = (i <= j) & (j < i + band_reach)

    # Create the mask for the left band
    top_mask = (j <= i) & (i < j + band_reach)

    # Combine the masks
    mask = top_mask | left_mask

    # Fill the band with random values
    r1, r2 = uniform_range
    tensor_view = tensor[mask]
    tensor[mask] = tensor_view.uniform_(r1, r2)

    return tensor


class ReservoirLocalConnectivity(nn.Module):
    """Creates a reservoir network module with local connectivity.
    Reservoir computing: https://www.sciencedirect.com/science/article/abs/pii/S1574013709000173
    Reservoir computing on memory tasks: https://drive.google.com/file/d/1RZvrI-3yo3M7gkrmDQnV2Tmn4ijvfcAy/view

    Arguments:
    - input_size (int): Num features in input to network.
    - num_unique (int): Number of hidden elements activated exclusively by each input
        element.
    - num_shared (int): Number of hidden elements activated by exclusively by each pair
        of neighboring input elements.
    - reach (int): Num steps in off-diagonal in random band tensor for rnn weights.
    - input_conn_prob (float): Probability that input connections are not zeroed out.
    - global_conn_prob (float): Probability that global connections are not zeroed out.
    - device (str): Which device to run the reservoir on
    """

    def __init__(
        self,
        input_size: int = 279,
        num_unique: int = 20,
        num_shared: int = 10,
        reach: int = 10,
        input_conn_prob: float = 0.5,
        local_conn_prob: float = 0.5,
        global_conn_prob: float = 0.01,
        device: str | torch.device = "cpu",
        _random_band_uniform_range: tuple[float, float] = (-1.0, 1.0),
        _hidden_to_hidden_connections_uniform_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        wmask = _rolling_mask(input_size, num_unique, num_shared, device=device)
        self.hidden_size = wmask.shape[0]
        self.rnn = nn.RNNCell(input_size, self.hidden_size).to(device)

        with torch.no_grad():
            # Recurrent weights: Create a random band across the diagonal of the matrix
            self.rnn.weight_hh.data = _random_band_tensor(
                self.hidden_size, reach, _random_band_uniform_range, device=device
            )

            # Need to use bernoulli instead of dropout because dropout scales the non-zeroed out
            # outputs by 1/(1-p) to compensate for fewer neurons, which we don't want
            self.rnn.weight_hh.data *= torch.bernoulli(
                torch.ones_like(self.rnn.weight_hh.data) * local_conn_prob
            )
            r1, r2 = _hidden_to_hidden_connections_uniform_range
            new_cons_hh = torch.zeros_like(self.rnn.weight_hh.data).uniform_(r1, r2)
            new_cons_hh *= torch.bernoulli(
                torch.ones_like(self.rnn.weight_hh.data) * global_conn_prob
            )
            self.rnn.weight_hh += new_cons_hh

            # Input weights
            torch.nn.init.xavier_uniform_(self.rnn.weight_ih, gain=10)
            self.rnn.weight_ih *= torch.bernoulli(
                torch.ones_like(self.rnn.weight_ih) * input_conn_prob
            )
            self.rnn.bias_hh.zero_()
            self.rnn.bias_ih.zero_()
            self.rnn.weight_ih *= wmask
            principal_eigen_value = torch.max(
                torch.real(torch.linalg.eigvals(self.rnn.weight_hh))
            )
            self.rnn.weight_hh *= 1.0 / principal_eigen_value

        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x, mask: torch.Tensor | None = None):
        if not hasattr(self, "hidden"):
            self.register_buffer(
                "hidden", torch.zeros(x.shape[0], self.hidden_size, device=x.device)
            )

        if mask is None:
            self.hidden = self.rnn(x, self.hidden)
        else:
            self.hidden[mask] = self.rnn(x[mask], self.hidden[mask])
        return self.hidden


class ReservoirLocalConnectivityBatchSeq(nn.Module):
    def __init__(
        self,
        input_size,
        num_unique=10,
        num_shared=5,
        spectral_radius=1.0,
        reach=10,
        input_conn_prob=0.5,
        local_conn_prob=0.5,
        global_conn_prob=0.01,
    ):
        super().__init__()

        wmask = _rolling_mask(input_size, num_unique, num_shared)
        self.hidden_size = wmask.shape[0]
        self.register_buffer("hidden", torch.zeros(1, 1, self.hidden_size))
        wmask = wmask.to(self.hidden.device)
        self.rnn = nn.RNN(
            input_size, self.hidden_size, batch_first=True
        ).requires_grad_(False)
        # Recurrent weights
        self.rnn.weight_hh_l0 = nn.Parameter(
            _random_band_tensor(self.hidden_size, reach).to(self.hidden.device)
        ).requires_grad_(False)
        self.rnn.weight_hh_l0 *= torch.bernoulli(self.rnn.weight_hh_l0, local_conn_prob)
        newCons_hh = (2 * torch.rand_like(self.rnn.weight_hh_l0) - 1) * torch.bernoulli(
            self.rnn.weight_hh_l0, global_conn_prob
        )
        self.rnn.weight_hh_l0 += newCons_hh

        # Input weights
        torch.nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=10)
        self.rnn.weight_ih_l0 *= torch.bernoulli(self.rnn.weight_ih_l0, input_conn_prob)
        with torch.no_grad():
            self.rnn.bias_hh_l0.zero_()
            self.rnn.bias_ih_l0.zero_()  # = nn.Parameter(torch.randn_like(self.rnn.bias_ih))
            self.rnn.weight_ih_l0 *= wmask

            W = self.rnn.weight_hh_l0.detach()[-self.hidden_size :]
            eigs = torch.abs(torch.real(torch.linalg.eigvals(W)))
            maxEig = eigs.max()
            self.rnn.weight_hh_l0[-self.hidden_size :] *= spectral_radius / maxEig

    def forward(self, x, restart=False):
        output, self.hidden = self.rnn(x, (1 - restart) * self.hidden.detach())
        return output, self.hidden


class ReservoirBatchSeq(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        connection_probability: float = 0.4,
        input_connection_probability: float = 0.1,
        spectral_radius: float = 1.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.connection_probability = connection_probability
        self.input_connection_probability = input_connection_probability
        self.register_buffer("hidden", torch.zeros(1, 1, self.hidden_size))

        self.rnn = nn.RNN(
            input_size, self.hidden_size, batch_first=True
        ).requires_grad_(False)

        with torch.no_grad():
            # Recurrent weights
            self.rnn.weight_hh_l0 *= torch.bernoulli(
                self.rnn.weight_hh_l0, self.connection_probability
            )

            # Input weights
            torch.nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=10)
            self.rnn.weight_ih_l0 *= torch.bernoulli(
                self.rnn.weight_ih_l0, self.input_connection_probability
            )

            # Zeroed Biases
            self.rnn.bias_hh_l0.zero_()
            self.rnn.bias_ih_l0.zero_()

            # Spectral normalization
            W = self.rnn.weight_hh_l0.detach()[-self.hidden_size :]
            eigs = torch.abs(torch.real(torch.linalg.eigvals(W)))
            maxEig = eigs.max()
            self.rnn.weight_hh_l0[-self.hidden_size :] *= spectral_radius / maxEig

    def forward(self, x, restart=False):
        output, self.hidden = self.rnn(x, (1 - restart) * self.hidden.detach())
        return output, self.hidden


# import numpy as np
# from PIL import Image

# n = 1024
# d = 64
# k = 1024
# rnn = ReservoirBatchSeq(
#     d,
#     k,
#     connection_probability=0.1,
#     input_connection_probability=0.1,
#     spectral_radius=1.25,
#     reset=True
# )


# x = torch.randn(n, d)
# x[1:] *= 0

# z = rnn(x)[0]
# z = (z.transpose(1, 0) + 1) / 2
# z = torch.stack((z**2, z, 1 - z)).permute(1, 2, 0)
# z = z.numpy() * 255
# # z = np.clip(z, z*0, z**0)
# z = z.astype(np.uint8)


# im = Image.fromarray(z)
# im.save("reservoir.bmp")
