from typing import Iterable, Optional, Callable, List, Dict, Any, Union
import pickle
import os, sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import tqdm as tq
from tqdm.contrib.concurrent import (
    process_map,
)  # or use tqdm.contrib.concurrent.process_map if available

import itertools
from torch.utils.data import TensorDataset, Dataset
from torch.nn import functional as F

from devinterp.slt.llc import estimate_learning_coeff_with_summary
from devinterp.optim.sgld import SGLD

from abc import ABC


class ToyAutoencoder(nn.Module):
    """
    Basic Network class for linear transformation with non-linear activations
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tied: bool = True,
        final_bias: bool = False,
        hidden_bias: bool = False,
        nonlinearity: Callable = F.relu,
        unit_weights: bool = False,
        standard_magnitude: bool = False,
        initial_scale_factor: float = 1.0,
        initial_bias: Optional[torch.Tensor] = None,
        initial_embed: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Set the dimensions and parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity
        self.tied = tied
        self.final_bias = final_bias
        self.unit_weights = unit_weights
        self.standard_magnitude = standard_magnitude

        # Define the input layer (embedding)
        self.embedding = nn.Linear(self.input_dim, self.hidden_dim, bias=hidden_bias)

        # Set initial embeddings if provided
        if initial_embed is not None:
            self.embedding.weight.data = initial_embed

        # Define the output layer (unembedding)
        self.unembedding = nn.Linear(self.hidden_dim, self.input_dim, bias=final_bias)

        # Set initial bias if provided
        if initial_bias is not None:
            self.unembedding.bias.data = initial_bias

        # If standard magnitude is set, normalize weights and maintain average norm
        if self.standard_magnitude:
            avg_norm = torch.norm(self.embedding.weight.data, p=2, dim=0).mean()
            self.embedding.weight.data = (
                F.normalize(self.embedding.weight.data, p=2, dim=0) * avg_norm
            )

        # If unit weights is set, normalize weights
        if self.unit_weights:
            self.embedding.weight.data = F.normalize(
                self.embedding.weight.data, p=2, dim=0
            )

        # Tie the weights of embedding and unembedding layers
        if tied:
            self.unembedding.weight = torch.nn.Parameter(
                self.embedding.weight.transpose(0, 1)
            )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network
        """
        # Apply the same steps for weights as done during initialization
        if self.unit_weights:
            self.embedding.weight.data = F.normalize(
                self.embedding.weight.data, p=2, dim=0
            )

        if self.standard_magnitude:
            avg_norm = torch.norm(self.embedding.weight.data, p=2, dim=0).mean()
            self.embedding.weight.data = (
                F.normalize(self.embedding.weight.data, p=2, dim=0) * avg_norm
            )

        if self.tied:
            self.unembedding.weight.data = self.embedding.weight.data.transpose(0, 1)

        x = self.embedding(x)
        x = self.unembedding(x)
        x = self.nonlinearity(x)

        return x


"""
Adapted from [TMS-zoo](https://github.com/JakeMendel/TMS-zoo)
"""


class SyntheticDataset(Dataset, ABC):
    num_samples: int
    num_features: int
    sparsity: Union[float, int]
    # importance: Optional[float]

    def __init__(
        self,
        num_samples,
        num_features,
        sparsity,
        # importance=None
    ):
        """
        Initialize the  object.

        Args:
            num_samples: The number of samples to generate.
            num_features: The dimension of the feature vector.
            sparsity: (float) the probability that a given feature is zero or (int) the number of features that are set to one.
            importance: The importance of the features. If None, then the features are weighted uniformly.
                        Otherwise, the features are weighted by `importance ** (1 + i)`, where `i` is the index of the feature.
        """
        self.num_samples = num_samples  # The number of samples in the dataset
        self.num_features = (
            num_features  # The size of the feature vector for each sample
        )
        self.sparsity = sparsity
        # self.importance = importance
        self.data = self.generate_data()  # Generate the synthetic data

    def generate_values(self):
        raise NotImplementedError

    def generate_mask(self):
        """
        Generate a sparse mask for the given dataset.

        If ``sparsity`` is a float, then the mask is generated by sampling from a Bernoulli distribution with parameter ``1 - sparsity``.
        If ``sparsity`` is an integer, then the mask is generated by sampling exactly ``sparsity`` indices without replacement.

        Args:
            dataset: The dataset to generate the mask for.

        Returns:
            A sparse mask for the given dataset.
        """

        if isinstance(self.sparsity, float):
            return torch.bernoulli(
                torch.ones((self.num_samples, self.num_features)) * (1 - self.sparsity)
            )
        elif isinstance(self.sparsity, int):
            mask = torch.zeros((self.num_samples, self.num_features))
            for i in range(self.num_samples):
                indices = torch.randperm(self.num_features)[: self.sparsity]
                mask[i, indices] = 1

            return mask

        else:
            raise ValueError(
                f"Sparsity must be a float or an integer. Received {type(self.sparsity)}."
            )

    def generate_data(self):
        return self.generate_mask() * self.generate_values()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


class SyntheticUniformValued(SyntheticDataset):
    """
    This class creates a synthetic dataset where each sample is a vector which has indices which are zero with probability sparsity and uniform between 0 and 1 otherwise
    """

    def generate_values(self):
        return torch.rand((self.num_samples, self.num_features))


class SyntheticBinaryValued(SyntheticDataset):
    """
    This class creates a synthetic dataset where each sample is a vector which has indices which are zero with probability ``sparsity`` and 1 otherwise
    """

    def generate_values(self):
        return 1.0


torch.manual_seed(1)

DEVICE = os.environ.get(
    "DEVICE",
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu",
)
DEVICE = torch.device(DEVICE)
NUM_CORES = int(os.environ.get("NUM_CORES", 1))


def generate_2d_kgon_vertices(k, rot=0.0, pad_to=None, force_length=0.9):
    """Set the weights of a 2D k-gon to be the vertices of a regular k-gon."""
    # Angles for the vertices
    theta = np.linspace(0, 2 * np.pi, k, endpoint=False) + rot

    # Generate the vertices
    x = np.cos(theta)
    y = np.sin(theta)
    result = np.vstack((x, y))

    if pad_to is not None and k < pad_to:
        num_pad = pad_to - k
        result = np.hstack([result, np.zeros((2, num_pad))])

    return result * force_length


def generate_init_param(
    m,
    n,
    init_kgon,
    prior_std=1.0,
    no_bias=True,
    init_zerobias=True,
    seed=0,
    force_negb=False,
    noise=0.01,
):
    np.random.seed(seed)

    if init_kgon is None or m != 2:
        init_W = np.random.normal(size=(m, n)) * prior_std
    else:
        assert init_kgon <= n
        rand_angle = np.random.uniform(0, 2 * np.pi, size=(1,))
        noise = np.random.normal(size=(m, n)) * noise
        init_W = generate_2d_kgon_vertices(init_kgon, rot=rand_angle, pad_to=n) + noise

    if no_bias:
        param = {"W": init_W}
    else:
        init_b = np.random.normal(size=(n, 1)) * prior_std
        if force_negb:
            init_b = -np.abs(init_b)
        if init_zerobias:
            init_b = init_b * 0
        param = {"W": init_W, "b": init_b}
    return param


def create_and_train(
    m: int,
    n: int,
    num_samples: int,
    batch_size: Optional[int] = 1,
    num_epochs: int = 100,
    lr: float = 0.001,
    log_ivl: Iterable[int] = [],
    device=DEVICE,
    momentum=0.9,
    weight_decay=0.0,
    init_kgon=None,
    no_bias=False,
    init_zerobias=False,
    prior_std=10.0,
    force_negb=False,
    seed=0,
    opt_func=optim.SGD,
):
    model = ToyAutoencoder(m, n, final_bias=True)

    init_weights = generate_init_param(
        n,
        m,
        init_kgon,
        no_bias=no_bias,
        init_zerobias=init_zerobias,
        prior_std=prior_std,
        seed=seed,
        force_negb=force_negb,
    )
    model.embedding.weight.data = torch.from_numpy(init_weights["W"]).float()

    if "b" in init_weights:
        model.unembedding.bias.data = torch.from_numpy(
            init_weights["b"].flatten()
        ).float()

    dataset = SyntheticBinaryValued(num_samples, m, 1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if opt_func == optim.SGD:
        optimizer = opt_func(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif opt_func == optim.Adam:
        optimizer = opt_func(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_func}")
    criterion = nn.MSELoss()

    logs = pd.DataFrame([{"loss": None, "acc": None, "step": step} for step in log_ivl])

    model.to(device)
    weights = []

    def log(step):
        loss = 0.0
        acc = 0.0
        length = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                outputs = model(batch)
                loss += criterion(outputs, batch).item()
                acc += (outputs.round() == batch).float().sum().item()
                length += len(batch)

        acc /= length

        logs.loc[logs["step"] == step, ["loss", "acc"]] = [loss, acc]
        weights.append(
            {k: v.cpu().detach().clone().numpy() for k, v in model.state_dict().items()}
        )

    step = 0
    log(step)

    for epoch in tqdm(range(num_epochs), desc="Training"):
        for batch in dataloader:
            batch = batch.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, batch)

            # Backward pass and optimize
            loss.backward()

            ## Print gradients
            # for name, parameter in model.named_parameters():
            #    if parameter.requires_grad:
            #        print(f"{name} gradient: {parameter.grad}")

            optimizer.step()

            step += 1

            if step in log_ivl:
                log(step)

    return logs, weights


def get_versions():
    versions = {
        # first version that still works.
        "v1.4.0": {
            "num_runs": 10,
            "num_features": 8,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 20000,
            "init_kgon": 2,
            "num_observations": 100,
            "lr": 0.01,
            "prior_std": 10.0,
        },
        # Version with the most batch_sizes computed
        "v1.5.0": {
            "num_runs": 25,
            "batch_sizes": [2**n for n in range(10, -1, -1)],
            "num_features": 6,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 4500,
            "init_kgon": 4,
            "num_observations": 50,
            "lr": 0.01,  # Accidentally used this learning rate instead of learning_rate in the original code
            "prior_std": 10.0,
        },
        "v1.6.0": {
            "num_runs": 50,
            "batch_sizes": [2**n for n in range(10, 8, -1)],
            "num_features": 6,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 20000,  # wanted to see if there is a noticable increase in the number of
            "init_kgon": 4,
            "num_observations": 50,
            "lr": 0.005,
            "prior_std": 10.0,
        },
        "v1.7.0": {
            "num_runs": 100,
            "batch_sizes": [2**n for n in range(10, 7, -1)],
            "num_features": 6,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 4500,
            "init_kgon": 4,
            "num_observations": 50,
            "lr": 0.1,
            "prior_std": 10.0,
        },
        # c=20 and dimension 3
        "v1.8.0": {
            "num_runs": 100,
            "batch_sizes": [2**n for n in range(10, 8, -1)],
            "num_features": 20,
            "num_hidden_units": 3,
            "num_samples": 1024,
            "num_epochs": 4500,
            "init_kgon": 4,
            "num_observations": 50,
            "lr": 0.005,
            "prior_std": 10.0,
        },
        # c=20
        "v1.9.0": {
            "num_runs": 100,
            "batch_sizes": [2**n for n in range(10, 8, -1)],
            "num_features": 20,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 4500,
            "init_kgon": 4,
            "num_observations": 50,
            "lr": 0.005,
            "prior_std": 10.0,
        },
        # First version that uses hyperparameters that are roughly like in the paper
        "v1.10.0": {
            "num_runs": 100,
            "batch_sizes": [2**n for n in range(10, 5, -1)],
            "num_features": 6,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 4500,
            "init_kgon": 4,
            "num_observations": 100,
            "lr": 0.005,
            "prior_std": 10.0,
        },
        # unsure what is different?
        "v1.11.0": {
            "num_runs": 100,
            "batch_sizes": [2**n for n in range(10, 5, -1)],
            "num_features": 6,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 4500,
            "init_kgon": 4,
            "num_observations": 100,
            "lr": 0.005,
            "prior_std": 10.0,
        },
        # 20 features.
        "v1.12.0": {
            "num_runs": 100,
            "batch_sizes": [2**n for n in range(10, 5, -1)],
            "num_features": 20,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 4500,
            "init_kgon": 4,
            "num_observations": 100,
            "lr": 0.005,
            "prior_std": 10.0,
        },
        # 3 dimensions
        "v1.13.0": {
            "num_runs": 100,
            "batch_sizes": [2**n for n in range(10, 5, -1)],
            "num_features": 20,
            "num_hidden_units": 3,
            "num_samples": 1024,
            "num_epochs": 4500,
            "init_kgon": 4,
            "num_observations": 100,
            "lr": 0.005,
            "prior_std": 10.0,
        },
        # 20K episodes with original parameters:
        "v1.14.0": {
            "num_runs": 100,
            "batch_sizes": [2**n for n in range(10, 9, -1)],
            "num_features": 6,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 20000,
            "init_kgon": 4,
            "num_observations": 100,
            "lr": 0.005,
            "prior_std": 10.0,
        },
        # unfinished
        "v1.15.0": {
            "num_runs": 100,
            "batch_sizes": [2**n for n in range(10, 8, -1)],
            "num_features": 6,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 4500,
            "init_kgon": 4,
            "num_observations": 100,
            "lr": 0.005,
            "prior_std": 10.0,
        },
        # tried what happens if force_negative (didn't make a big different to phase transitions)
        "v1.16.0": {
            "num_runs": 10,
            "batch_sizes": [1024, 512, 16],
            "num_features": 6,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 4500,
            "init_kgon": 4,
            "num_observations": 100,
            "lr": 0.005,
            "prior_std": 1.0,
            "force_negb": True,
        },
        # unfinished
        "v1.17.0": {
            "num_runs": 10,
            "batch_sizes": [1024, 512, 32, 16],
            "num_features": 6,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 4500,
            "init_kgon": 4,
            "num_observations": 100,
            "lr": 0.005,
            "prior_std": 1.0,
            "force_negb": False,
        },
        "v1.18.0": {
            "num_runs": 10,
            "batch_sizes": [1024],
            "num_features": 6,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 4500,
            "init_kgon": 4,
            "num_observations": 100,
            "lr": 0.005,
            "prior_std": 0.1,
            "force_negb": False,
            "optimizer": optim.Adam,
        },
        # initialized with a smaller batch size (tried to see if this makes a difference)
        "v1.19.0": {
            "num_runs": 100,
            "batch_sizes": [1024, 16],
            "num_features": 6,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 4500,
            "init_kgon": 4,
            "num_observations": 100,
            "lr": 0.05,
            "prior_std": 0.1,
            "force_negb": False,
            "optimizer": optim.SGD,
        },
        "v1.20.0": {
            "num_runs": 100,
            "batch_sizes": [1024, 16],
            "num_features": 6,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 4500,
            "init_kgon": 4,
            "num_observations": 100,
            "lr": 0.005,
            "prior_std": 0.1,
            "force_negb": False,
            "optimizer": optim.SGD,
        },
        # unfinished
        "v1.21.0": {
            "num_runs": 100,
            "batch_sizes": [1024, 16],
            "num_features": 6,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 4500,
            "init_kgon": 4,
            "num_observations": 100,
            "lr": 0.5,
            "prior_std": 0.1,
            "force_negb": False,
            "optimizer": optim.SGD,
        },
        # overfitting (for debugging)
        "v1.22.0": {
            "num_runs": 1,
            "batch_sizes": [1024],
            "num_features": 4,
            "num_hidden_units": 2,
            "num_samples": 1024,
            "num_epochs": 20000,
            "init_kgon": 2,
            "num_observations": 50,
            "lr": 0.005,
            "prior_std": 10.0,
            "force_negb": False,
            "optimizer": optim.SGD,
        },
    }
    return versions


def save_individual_results(batch_size, run, run_logs, run_weights, version):
    os.makedirs("results", exist_ok=True)
    with open(f"results/batch_logs_{batch_size}_run_{run}_{version}.pkl", "wb") as f:
        pickle.dump(run_logs, f)
    with open(f"results/batch_weights_{batch_size}_run_{run}_{version}.pkl", "wb") as f:
        pickle.dump(run_weights, f)


def aggregate_and_save_results(batch_size, version, num_runs):
    all_logs = []
    all_weights = []
    for run in range(num_runs):
        with open(
            f"results/batch_logs_{batch_size}_run_{run}_{version}.pkl", "rb"
        ) as f:
            run_logs = pickle.load(f)
        with open(
            f"results/batch_weights_{batch_size}_run_{run}_{version}.pkl", "rb"
        ) as f:
            run_weights = pickle.load(f)
        all_logs.append(run_logs)
        all_weights.append(run_weights)

    # Save aggregated results in a single file
    with open(f"results/batch_logs_{batch_size}_{version}.pkl", "wb") as f:
        pickle.dump(all_logs, f)
    with open(f"results/batch_weights_{batch_size}_{version}.pkl", "wb") as f:
        pickle.dump(all_weights, f)


def main():
    (version,) = sys.argv[1:]
    versions = get_versions()

    num_runs = versions[version]["num_runs"]
    batch_sizes = versions[version]["batch_sizes"]
    NUM_FEATURES = versions[version]["num_features"]
    NUM_HIDDEN_UNITS = versions[version]["num_hidden_units"]
    NUM_SAMPLES = versions[version]["num_samples"]
    NUM_EPOCHS = versions[version]["num_epochs"]
    INIT_KGON = versions[version]["init_kgon"]
    NUM_OBSERVATIONS = versions[version]["num_observations"]
    lr = versions[version]["lr"]
    STEPS = sorted(
        list(set(np.logspace(0, np.log10(NUM_EPOCHS), NUM_OBSERVATIONS).astype(int)))
    )
    PLOT_STEPS = [
        min(STEPS, key=lambda s: abs(s - i))
        for i in [0, 200, 500, 1000, NUM_EPOCHS - 1]
    ]  # originally [0, 200, 2000, 10000, NUM_EPOCHS - 1]
    PLOT_INDICES = [STEPS.index(s) for s in PLOT_STEPS]
    prior_std = versions[version]["prior_std"]
    if "force_negb" in versions[version]:
        force_negb = versions[version]["force_negb"]
    else:
        force_negb = False
    if "optimizer" in versions[version]:
        optimizer = versions[version]["optimizer"]
    else:
        optimizer = optim.SGD

    # Dictionary to store aggregated results for all batch sizes
    aggregated_logs = {}
    aggregated_weights = {}

    for batch_size in batch_sizes:
        for run in range(num_runs):
            result_log_file = f"results/batch_logs_{batch_size}_run_{run}_{version}.pkl"
            if not os.path.exists(result_log_file):
                print(f"Running batch size {batch_size} for run {run}...")
                run_logs, run_weights = create_and_train(
                    NUM_FEATURES,
                    NUM_HIDDEN_UNITS,
                    num_samples=NUM_SAMPLES,
                    log_ivl=STEPS,
                    batch_size=batch_size,
                    lr=lr,
                    num_epochs=NUM_EPOCHS,
                    init_kgon=INIT_KGON,
                    init_zerobias=False,
                    seed=run,
                    prior_std=prior_std,
                    force_negb=force_negb,
                    opt_func=optimizer,
                )

                # Save the results for this run
                save_individual_results(batch_size, run, run_logs, run_weights, version)

        # Aggregate results after all runs are completed for this batch size
        aggregate_and_save_results(batch_size, version, num_runs)
        with open(f"results/batch_logs_{batch_size}_{version}.pkl", "rb") as f:
            aggregated_logs[batch_size] = pickle.load(f)
        with open(f"results/batch_weights_{batch_size}_{version}.pkl", "rb") as f:
            aggregated_weights[batch_size] = pickle.load(f)

    # Save the aggregated results for all batch sizes
    with open(f"results/batch_logs_{version}.pkl", "wb") as f:
        pickle.dump(aggregated_logs, f)
    with open(f"results/batch_weights_{version}.pkl", "wb") as f:
        pickle.dump(aggregated_weights, f)
    with open(f"results/batch_sizes_{version}.pkl", "wb") as f:
        pickle.dump(batch_sizes, f)


if __name__ == "__main__":
    main()
