"""
Discrete diffusion utilities for DiffMS.

Provides noise schedule computation, posterior distribution calculation,
and discrete feature sampling for the graph diffusion process.
"""

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PlaceHolder:
    """Container for graph features (nodes X, edges E, global y)."""
    X: torch.Tensor
    E: torch.Tensor
    y: torch.Tensor

    def mask(self, node_mask):
        x_mask = node_mask.unsqueeze(-1)
        e_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(1)
        self.X = self.X * x_mask
        self.E = self.E * e_mask
        return self


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """Cosine noise schedule for discrete diffusion."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze()


class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """Discrete noise schedule with precomputed betas and alpha_bars."""

    def __init__(self, noise_schedule: str = "cosine", timesteps: int = 500):
        super().__init__()
        self.timesteps = timesteps
        if noise_schedule == "cosine":
            betas = cosine_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer("betas", torch.from_numpy(betas).float())
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)
        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert (t_normalized is None) != (t_int is None)
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert (t_normalized is None) != (t_int is None)
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()]


class MarginalUniformTransition:
    """Transition matrices using marginal distributions as the limit."""

    def __init__(self, x_marginals, e_marginals, y_classes):
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes
        self.u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t, device):
        beta_t = beta_t.unsqueeze(1).to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)
        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)
        return PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t, device):
        alpha_bar_t = alpha_bar_t.unsqueeze(1).to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)
        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y
        return PlaceHolder(X=q_x, E=q_e, y=q_y)


class DiscreteUniformTransition:
    """Transition matrices with uniform limit distribution."""

    def __init__(self, x_classes, e_classes, y_classes):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        self.u_x = torch.ones(1, x_classes, x_classes) / x_classes if x_classes > 0 else torch.zeros(1, 1, 1)
        self.u_e = torch.ones(1, e_classes, e_classes) / e_classes if e_classes > 0 else torch.zeros(1, 1, 1)
        self.u_y = torch.ones(1, y_classes, y_classes) / y_classes if y_classes > 0 else torch.zeros(1, 1, 1)

    def get_Qt(self, beta_t, device):
        beta_t = beta_t.unsqueeze(1).to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)
        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)
        return PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t, device):
        alpha_bar_t = alpha_bar_t.unsqueeze(1).to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)
        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y
        return PlaceHolder(X=q_x, E=q_e, y=q_y)


def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """Compute p(x_{t-1} | x_t, x_0) for all possible x_0 values.

    Returns tensor where entry [b, n, x0, x_{t-1}] is the posterior probability.
    """
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)
    Qt_T = Qt.transpose(-1, -2)
    left_term = (X_t @ Qt_T).unsqueeze(dim=2)
    right_term = Qsb.unsqueeze(1)
    numerator = left_term * right_term
    X_t_transposed = X_t.transpose(-1, -2)
    prod = Qtb @ X_t_transposed
    prod = prod.transpose(-1, -2)
    denominator = prod.unsqueeze(-1)
    denominator[denominator == 0] = 1e-6
    return numerator / denominator


def sample_discrete_features(probX, probE, node_mask):
    """Sample discrete features from categorical distributions."""
    bs, n, _ = probX.shape
    probX[~node_mask] = 1 / probX.shape[-1]
    X_t = probX.reshape(bs * n, -1).multinomial(1).reshape(bs, n)

    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n, device=probE.device).unsqueeze(0).expand(bs, -1, -1).bool()
    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask] = 1 / probE.shape[-1]
    E_t = probE.reshape(bs * n * n, -1).multinomial(1).reshape(bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + E_t.transpose(1, 2)
    return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))


def sample_discrete_feature_noise(limit_dist, node_mask):
    """Sample from the limit distribution of the diffusion process."""
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)

    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)

    U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()

    upper_tri = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=n_max, col=n_max, offset=1)
    upper_tri[:, indices[0], indices[1], :] = 1
    U_E = U_E * upper_tri
    U_E = U_E + U_E.transpose(1, 2)

    return PlaceHolder(X=U_X, E=U_E, y=torch.zeros(bs, 0)).mask(node_mask)
