import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from reservoir_local_connectivity import ReservoirBatchSeq


# =============================================================================
# Hyperparameters
# =============================================================================
class Config:
    """All hyperparameters in one place for easy modification."""

    # Data generation
    D = 16  # Observation dimension
    T = 256  # Number of time steps
    N_TASK = 4  # Number of tasks
    SEED = 130  # Random seed
    STATE_DIM = 64  # State dimension for data generation
    DATA_NOISE_SD = 3  # Std. deviation of initial random state vector
    DATA_TEMP = 1e-8  # Temperature for data generation

    # Model architecture
    SQUARE_DIM = 64  # Dimension of the square grid for cortex
    HIDDEN_SIZE = SQUARE_DIM * SQUARE_DIM  # Hidden layer size (Dr)

    # Training
    TRAIN_DURATION = 200  # Number of training steps per task
    TRAIN_LR = 1e-4  # Learning rate for model training

    # Reconfiguration (mask optimization)
    RECONFIG_STEPS = 1  # Number of gradient steps for mask optimization
    RECONFIG_LR = 1  # Learning rate for mask optimization
    RECONFIG_SAMPLE_SIZE = 100  # Number of obs samples used to reconfigure

    # Lateral dynamics
    LATERAL_STEPS = 16  # Lateral steps with noise (train)
    LATERAL_NOISE_SD = 0.0  # Noise standard deviation for lateral dynamics

    # KWTA (k-winner-take-all)
    KWTA_K_RATIO = 8  # Ratio for k in KWTA (k = HIDDEN_SIZE // KWTA_K_RATIO)
    KWTA_BLEND_STEPS = 4  # Number of blending steps with KWTA
    KWTA_BLEND_RATIO = 0.25  # Blending ratio (0.75 old + 0.25 KWTA)

    # Meta training
    N_META_EPOCHS = 12  # Number of epochs over all tasks

    # Visualization
    GIF_SAVE_EPOCH = 4  # Save GIF at this epoch
    GIF_DURATION = 50  # Duration per frame in ms


# =============================================================================
# Utility Functions
# =============================================================================
def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a PyTorch module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())


def kwta(x, k=None):
    """
    K-winner-take-all function.
    Returns a binary mask with only the top k values set to 1.
    """
    if k is None:
        k = Config.HIDDEN_SIZE // Config.KWTA_K_RATIO
    idx = torch.topk(x, k).indices
    out = torch.zeros_like(x)
    out[idx] = 1
    return out


def normalize_mask(m):
    """Normalize mask to [0, 1] range."""
    m = m - m.min()
    m = m / m.max()
    return m


def create_frame(m, dim):
    """Create a PIL Image frame from a mask tensor."""
    frame_data = m.reshape(dim, dim) * 255
    frame = Image.fromarray(frame_data.detach().cpu().numpy().astype(np.uint8))
    return frame


# =============================================================================
# Data Generation
# =============================================================================
class DataGen:
    """Generates sequential data from multiple dynamical systems."""

    def __init__(self):
        self.state_dim = Config.STATE_DIM
        self.n_models = Config.N_TASK
        self.steps = Config.T
        self.obs_dim = Config.D

        # Create observation models (linear projections from state to observation)
        self.obs_model = [
            nn.Linear(self.state_dim, self.obs_dim, bias=False)
            for _ in range(self.n_models)
        ]

        # Create state evolution models (reservoir dynamics)
        self.state_model = [
            ReservoirBatchSeq(self.state_dim, self.state_dim)
            for _ in range(self.n_models)
        ]

    def __call__(self, noise_sd=None, temp=None, seed=None):
        """Generate data for all tasks."""
        if noise_sd is None:
            noise_sd = Config.DATA_NOISE_SD
        if temp is None:
            temp = Config.DATA_TEMP
        if seed:
            torch.manual_seed(seed)

        # Initial random conditions
        state_noise = noise_sd * torch.randn(self.n_models, self.steps, self.state_dim)
        state_noise[:, 1:] *= 0  # Only first step has noise

        # Generate states by evolving through reservoir
        states = [
            self.state_model[i](state_noise[i].unsqueeze(0))[0].squeeze()
            for i in range(self.n_models)
        ]

        # Project to observations and sample from categorical distribution
        probs = [
            torch.softmax(self.obs_model[i](states[i]) / temp, -1)
            for i in range(self.n_models)
        ]
        cats = [
            torch.distributions.OneHotCategorical(probs=probs[i]).sample()
            for i in range(self.n_models)
        ]
        return cats


# =============================================================================
# Model
# =============================================================================
class Cortex(nn.Module):
    """
    Cortex model with encoder-decoder architecture and lateral connections.
    The mask m modulates which hidden units are active.
    """

    def __init__(self, input_size, output_size, hidden_size, square_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.square_dim = square_dim

        # Encoder and decoder
        self.enc = nn.Linear(input_size, self.hidden_size)
        self.dec = nn.Linear(self.hidden_size, output_size)

        # Lateral connections (2D convolution for spatial processing)
        self.lateral = nn.Conv2d(
            2,
            2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=2,
            bias=False,
        ).requires_grad_(False)
        self.lateral.weight.data.fill_(1)

    def forward(self, x, m):
        """
        Forward pass through the cortex.
        Args:
            x: Input data
            m: Binary mask for hidden units
        Returns:
            y: Output predictions
            z0: Pre-mask hidden activations
        """
        z0 = self.enc(x)
        z = z0 * m  # Apply mask to hidden units
        y = self.dec(z)
        return y, z0

    def step_lateral(self, m, steps=1, noise_sd=0):
        """
        Apply lateral dynamics to the mask.
        Args:
            m: Current mask
            steps: Number of convolution steps
            noise_sd: Noise standard deviation
        Returns:
            Updated mask
        """

        # Reshape to 2D grid and split into on/off channels
        m = m.reshape(self.square_dim, self.square_dim)
        m = torch.stack((1 - m, m))

        # Apply lateral convolution with noise
        for _ in range(steps):
            m = self.lateral(m) + torch.randn_like(m) * noise_sd

        # Return the "on" channel, flattened
        return m[1].reshape(-1)


# =============================================================================
# Training and Evaluation Functions
# =============================================================================
def reconfigure_mask(model, y, m, frames, dim, is_test=False):
    """
    Reconfigure the mask for a new task.
    Args:
        model: Cortex model
        y: Task data
        m: Current mask
        frames: List to append visualization frames
        dim: Dimension for reshaping
        is_test: Whether this is test mode
    Returns:
        Updated mask
    """
    # Initial frame
    m = nn.Parameter(m.data)
    frames.append(create_frame(m.data, dim))

    # Optimize mask to fit the data
    m_opt = torch.optim.RMSprop([m], lr=Config.RECONFIG_LR)
    for i in range(Config.RECONFIG_STEPS):
        ypred, _z = model(
            y.unsqueeze(0)[: Config.RECONFIG_SAMPLE_SIZE], torch.sigmoid(m)
        )
        targ_idx = torch.where(y == 1)[1]
        loss = F.cross_entropy(
            ypred.squeeze()[:-1][: Config.RECONFIG_SAMPLE_SIZE],
            target=targ_idx[1:][: Config.RECONFIG_SAMPLE_SIZE],
        )
        entropy = torch.distributions.Categorical(logits=ypred).entropy().mean()

        m_opt.zero_grad()
        loss.backward()
        m_opt.step()

        if (i + 1) % 100 == 0:
            print(f"H={entropy},E={loss}")

        frames.append(create_frame(m, dim))

    # Apply lateral dynamics with noise
    for _ in range(Config.LATERAL_STEPS):
        m = model.step_lateral(m, noise_sd=Config.LATERAL_NOISE_SD)
        m = normalize_mask(m)
        frames.append(create_frame(m, dim))

    m = kwta(m.data)

    return m


def train_on_task(model, y, m, task_idx, opt=None, track_loss=True):
    """
    Train the model on a specific task.
    Args:
        model: Cortex model
        y: Task data
        m: Mask for this task
        task_idx: Task index
        opt: Optimizer (None for evaluation mode)
        track_loss: Whether to track and return loss values
    Returns:
        List of loss values (if track_loss is True)
    """
    is_training = opt is not None
    loss_history = []

    for i in range(Config.TRAIN_DURATION):
        ypred, _z = model(y.unsqueeze(0), m)
        targ_idx = torch.where(y == 1)[1]
        loss = F.cross_entropy(ypred.squeeze()[:-1], target=targ_idx[1:])
        entropy = torch.distributions.Categorical(logits=ypred).entropy().mean()

        if is_training:
            opt.zero_grad()
            loss.backward()
            opt.step()

        if (i + 1) % 100 == 0:
            print(f"Task:{task_idx},H={entropy},E={loss}")

        if track_loss:
            loss_history.append(loss.item())

    return loss_history


def save_plots(avg_acc_plot_0, avg_acc_plot_1, prefix=""):
    """
    Save training plots.
    Args:
        avg_acc_plot_0: Per-task loss history
        avg_acc_plot_1: Combined loss history
        prefix: Prefix for output filenames
    """
    # Per-task plot
    for task in range(Config.N_TASK):
        plt.plot(avg_acc_plot_0[task], linewidth=1)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.savefig(f"{prefix}test.png")
    plt.close()

    # Combined plot
    plt.plot(avg_acc_plot_1, linewidth=1)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.savefig(f"{prefix}test1.png")
    plt.close()


def run_experiment(model, data_gen, m, is_test=False):
    """
    Run a full experiment (train or test).
    Args:
        model: Cortex model
        data_gen: Data generator
        m: Initial mask
        is_test: Whether this is test mode
    Returns:
        frames: List of visualization frames
    """
    prefix = "test_" if is_test else ""
    gif_name = "self_organizing_nn_test.gif" if is_test else "self_organizing_nn.gif"

    avg_acc_plot_0 = [[] for _ in range(Config.N_TASK)]
    avg_acc_plot_1 = []
    frames = []

    # Generate data once for all epochs
    with torch.no_grad():
        Y = data_gen()

    # Run multiple epochs
    for epoch in range(Config.N_META_EPOCHS):
        for task in range(Config.N_TASK):
            y = Y[task]

            # Reconfigure mask for this task
            print(f"\nRECONFIGURE {task}")
            m = reconfigure_mask(model, y, m, frames, Config.SQUARE_DIM, is_test)

            # Save task-specific frame
            frame = create_frame(m, Config.SQUARE_DIM)
            for _ in range(10):
                frames.append(frame)
            frame.save(f"task{task}{('_test' if is_test else '')}.png")

            # Train on this task (or evaluate without training)
            print(f"\nTRAIN {task}")
            opt = (
                None
                if is_test
                else torch.optim.Adam(model.parameters(), lr=Config.TRAIN_LR)
            )
            loss_history = train_on_task(model, y, m, task, opt, track_loss=True)

            avg_acc_plot_0[task].extend(loss_history)
            avg_acc_plot_1.extend(loss_history)

            # Save plots
            save_plots(avg_acc_plot_0, avg_acc_plot_1, prefix)

        # Save GIF at specified epoch
        if epoch == Config.GIF_SAVE_EPOCH:
            print("Saving GIF...")
            frames[0].save(
                gif_name,
                save_all=True,
                append_images=frames[1:],
                duration=Config.GIF_DURATION,
                loop=0,
            )

    return frames


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # Set device and seed
    torch.set_default_device("cuda")
    torch.manual_seed(Config.SEED)

    # Initialize data generators
    data_gen = DataGen()
    data_gen_test = DataGen()

    # Initialize model
    model = Cortex(Config.D, Config.D, Config.HIDDEN_SIZE, Config.SQUARE_DIM)
    print(f"RNN parameters: {count_parameters(model)}")

    # Initialize mask
    m = torch.randn(Config.HIDDEN_SIZE)

    # Training phase
    print("\n" + "=" * 50)
    print("TRAINING PHASE")
    print("=" * 50)
    frames_train = run_experiment(model, data_gen, m, is_test=False)

    # Testing phase
    print("\n" + "=" * 50)
    print("TESTING GENERALIZATION")
    print("=" * 50)
    frames_test = run_experiment(model, data_gen_test, m, is_test=True)
