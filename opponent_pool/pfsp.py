"""Negotiation score and prioritized fictitious self-play probabilities."""

import math

import numpy as np


def negotiation_score(agreement_rate, utility, length, welfare, *, length_weight=-0.005,
                      welfare_weight=0.1, epsilon=0.01):
    """Learner score based only on observable outcomes.

    The welfare arguments remain in the signature so existing checkpoints and
    result writers stay compatible; they do not affect the score.
    """
    agreement = min(max(float(agreement_rate), 0.0), 1.0 - epsilon)
    return (1.0 - agreement) ** (-float(utility)) + length_weight * float(length)


def dominance_probability(entry):
    """Estimate opponent difficulty from learner-observable outcomes only."""
    metrics = (entry.agreement_rate, entry.agent_utility, entry.negotiation_length)
    if not np.all(np.isfinite(metrics)):
        return float("nan")
    if entry.matches == 0:
        return 0.5
    difficulty = (0.45 * (1.0 - np.clip(entry.agreement_rate, 0.0, 1.0)) +
                  0.40 * (1.0 - np.clip(entry.agent_utility, 0.0, 1.0)) +
                  0.15 * np.clip(entry.negotiation_length / 80.0, 0.0, 1.0))
    confidence = entry.matches / (entry.matches + 10.0)
    return float(np.clip(confidence * difficulty + (1.0 - confidence) * 0.5, 0.0, 1.0))


def pfsp_probabilities(entries, alpha=1.0, uniform_mix=0.1):
    """PFSP Eq. 10 using an estimated opponent-dominance probability."""
    if not entries:
        raise ValueError("Empty opponent pool")
    if alpha < 0 or not 0 <= uniform_mix <= 1:
        raise ValueError("Invalid PFSP parameters")
    values = np.asarray([dominance_probability(item) for item in entries], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.ptp(values) < 1e-12:
        return np.full(len(entries), 1 / len(entries))
    weights = np.exp(np.clip(alpha * (values - values.max()), -700, 0))
    total = weights.sum()
    if not math.isfinite(total) or total <= 0:
        return np.full(len(entries), 1 / len(entries))
    return (1 - uniform_mix) * weights / total + uniform_mix / len(entries)
