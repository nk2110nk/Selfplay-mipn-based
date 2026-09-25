"""Negotiation score and prioritized fictitious self-play probabilities."""

import math

import numpy as np


def negotiation_score(agreement_rate, utility, length, welfare, *, length_weight=-0.005,
                      welfare_weight=0.1, epsilon=0.01):
    agreement = min(max(float(agreement_rate), 0.0), 1.0 - epsilon)
    return (1.0 - agreement) ** (-float(utility)) + length_weight * float(length) + welfare_weight * float(welfare)


def dominance_probability(entry):
    """Estimate P[opponent dominates learner] from outcomes and negotiation metrics."""
    metrics = (entry.agreement_rate, entry.agent_utility, entry.opponent_utility,
               entry.social_welfare, entry.negotiation_length,
               entry.negotiation_score, entry.learner_negotiation_score)
    if not np.all(np.isfinite(metrics)):
        return float("nan")
    if entry.matches == 0:
        return 0.5
    empirical = (entry.wins + 0.5 * entry.draws + 0.5) / (entry.matches + 1.0)
    score_gap = float(entry.negotiation_score - entry.learner_negotiation_score)
    score_signal = 1.0 / (1.0 + math.exp(-np.clip(score_gap, -40.0, 40.0)))
    # Deal difficulty retains the multi-objective information required by the SAOP adaptation.
    difficulty = (1.0 - entry.agreement_rate + min(entry.negotiation_length / 80.0, 1.0) +
                  max(0.0, 3.0 - entry.social_welfare) / 3.0) / 3.0
    return float(np.clip(0.5 * empirical + 0.3 * score_signal + 0.2 * difficulty, 0.0, 1.0))


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
