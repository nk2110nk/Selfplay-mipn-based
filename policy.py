"""Factorized categorical actor and joint-action distributional critics."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def action_mask(nvec, valid_nvec, can_accept=True, device=None):
    """Return a padded [head, value] mask; 0=accept and 1=reject."""
    nvec = list(map(int, nvec))
    valid_nvec = list(map(int, valid_nvec))
    if len(nvec) != len(valid_nvec) or any(v < 1 or v > n for n, v in zip(nvec, valid_nvec)):
        raise ValueError("Invalid action dimensions")
    width = max(nvec)
    mask = torch.zeros((len(nvec), width), dtype=torch.bool, device=device)
    for i, valid in enumerate(valid_nvec):
        mask[i, :valid] = True
    if not can_accept:
        mask[-1, 0] = False
    return mask


def domain_nvec(domain, padded_nvec):
    counts = [len(issue.values) for issue in domain]
    if len(counts) + 1 > len(padded_nvec):
        raise ValueError("Domain has more issues than checkpoint")
    valid = counts + [1] * (len(padded_nvec) - len(counts) - 1) + [2]
    if any(v > n for v, n in zip(valid, padded_nvec)):
        raise ValueError("Domain value count exceeds checkpoint action space")
    return valid


def padded_observation(raw, obs_dim):
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    if len(raw) > obs_dim:
        raise ValueError("Observation exceeds checkpoint dimension")
    padded = np.zeros(obs_dim, dtype=np.float32)
    padded[:len(raw) - 1] = raw[:-1]
    padded[-1] = raw[-1]
    return padded


class Actor(nn.Module):
    """MultiDiscrete adaptation of Alpha-Nego's shared encoder and action heads."""

    def __init__(self, obs_dim, nvec, hidden=256, dropout=0.1):
        super().__init__()
        self.nvec = tuple(map(int, nvec))
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList(nn.Linear(hidden, n) for n in self.nvec)
        self.apply(_init_linear)

    def logits(self, obs, mask):
        x = self.body(obs)
        return [head(x).masked_fill(~mask[..., i, :n], -1e9) for i, (head, n) in enumerate(zip(self.heads, self.nvec))]

    def distributions(self, obs, mask):
        return [torch.distributions.Categorical(logits=values) for values in self.logits(obs, mask)]

    def sample(self, obs, mask, deterministic=False, relaxed=False, temperature=1.0):
        logits = self.logits(obs, mask)
        actions, onehots, log_probs = [], [], []
        for values in logits:
            dist = torch.distributions.Categorical(logits=values)
            if relaxed:
                sample = F.gumbel_softmax(values, tau=temperature, hard=True)
                index = sample.argmax(-1)
            else:
                index = values.argmax(-1) if deterministic else dist.sample()
                sample = F.one_hot(index, values.shape[-1]).to(values.dtype)
            actions.append(index)
            onehots.append(sample)
            log_probs.append(dist.log_prob(index))
        return torch.stack(actions, -1), torch.cat(onehots, -1), torch.stack(log_probs, -1).sum(-1)

    def entropy(self, obs, mask, *, hierarchical=True):
        """Per-head entropy; bid-head entropy matters only when the policy rejects."""
        distributions = self.distributions(obs, mask)
        entropy = torch.stack([dist.entropy() for dist in distributions], dim=-1)
        if hierarchical and len(distributions) > 1:
            reject_probability = distributions[-1].probs[..., 1:2]
            entropy = torch.cat((entropy[..., :-1] * reject_probability, entropy[..., -1:]), dim=-1)
        return entropy


class QuantileCritic(nn.Module):
    def __init__(self, obs_dim, nvec, quantiles=64, hidden=256, dropout=0.1):
        super().__init__()
        self.nvec = tuple(map(int, nvec))
        self.net = nn.Sequential(
            nn.Linear(obs_dim + sum(self.nvec), hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, quantiles),
        )
        self.apply(_init_linear)

    def forward(self, obs, action_onehot):
        return self.net(torch.cat((obs, action_onehot), -1))


def onehot_action(actions, nvec):
    return torch.cat([F.one_hot(actions[..., i].long(), n).float() for i, n in enumerate(nvec)], -1)


def critic_action(action_onehot, nvec):
    """Remove meaningless issue values from accept actions before critic input."""
    parts = torch.split(action_onehot, tuple(map(int, nvec)), dim=-1)
    if len(parts) == 1:
        return action_onehot
    reject = parts[-1][..., 1:2]
    return torch.cat(tuple(part * reject for part in parts[:-1]) + (parts[-1],), dim=-1)


def quantile_huber_loss(prediction, target, kappa=1.0):
    n = prediction.shape[-1]
    delta = target.unsqueeze(-2) - prediction.unsqueeze(-1)
    abs_delta = delta.abs()
    huber = torch.where(abs_delta <= kappa, 0.5 * delta.square(), kappa * (abs_delta - 0.5 * kappa))
    tau = ((torch.arange(n, device=prediction.device, dtype=prediction.dtype) + 0.5) / n).view(1, n, 1)
    return ((tau - (delta.detach() < 0).to(prediction.dtype)).abs() * huber / kappa).mean()


def _init_linear(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def select_lower_distribution(q1, q2):
    """Element-wise clipped double-Q distribution used by Alpha-Nego targets."""
    return torch.minimum(q1, q2)


def style_value(quantiles, style="neutral", aggressive_quantile=0.8, conservative_quantile=0.2, risk_weight=0.1):
    ordered = quantiles.sort(dim=-1).values
    n = ordered.shape[-1]
    if style == "neutral":
        return ordered.mean(-1)
    if style == "aggressive":
        tail = ordered[..., min(n - 1, int(math.floor(aggressive_quantile * n))):]
        return tail.mean(-1) + risk_weight * tail.var(-1, unbiased=False)
    if style == "conservative":
        return ordered[..., :max(1, int(math.ceil(conservative_quantile * n)))].mean(-1)
    raise ValueError(f"Unknown style: {style}")


def twin_style_value(q1, q2, style="neutral", aggressive_quantile=0.8,
                     conservative_quantile=0.2, risk_weight=0.1):
    """Compute a style value per critic, then apply clipped double Q."""
    first = style_value(q1, style, aggressive_quantile, conservative_quantile, risk_weight)
    second = style_value(q2, style, aggressive_quantile, conservative_quantile, risk_weight)
    return torch.minimum(first, second)
