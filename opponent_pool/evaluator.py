"""Evaluate the learner against every current pool entry."""

import random

import numpy as np
import torch

from environment import NegotiationEnv
from .pfsp import negotiation_score


def balanced_domain_schedule(domains, minimum_episodes):
    """Return complete domain cycles meeting the requested episode minimum."""
    unique_domains = list(dict.fromkeys(domains))
    if not unique_domains:
        raise ValueError("Pool evaluation needs at least one domain")
    if minimum_episodes < 1:
        raise ValueError("Pool evaluation episodes must be positive")
    cycles = (minimum_episodes + len(unique_domains) - 1) // len(unique_domains)
    return unique_domains * cycles


def evaluate_pool(model, pool, domains, episodes, model_dir, *, seed, device,
                  case="case1", length_weight=-0.005, welfare_weight=0.1):
    scripted = next((entry for entry in pool.entries if entry.kind == "scripted"), None)
    if scripted is None:
        raise ValueError("Pool needs a scripted anchor")
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    overall = []
    episode_rows = []
    schedule = balanced_domain_schedule(domains, episodes)
    try:
        for entry in list(pool.entries):
            rows = []
            for episode, domain in enumerate(schedule):
                episode_seed = seed + episode
                random.seed(episode_seed)
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                env = NegotiationEnv(domain, model_dir, model.obs_dim, model.nvec, case=case,
                                     device=device, test=True)
                observation = env.reset((entry, scripted))
                done = False
                while not done:
                    action = model.act(observation, env.current_mask(), deterministic=True)
                    observation, _, done, info = env.step(action)
                rows.append(info)
                episode_rows.append((entry, scripted, domain, episode_seed, info))
            metrics = {
                "agreement_rate": sum(row["agreement"] is not None for row in rows) / len(rows),
                "agent_utility": sum(row["my_util"] for row in rows) / len(rows),
                "opponent_utility": sum(row["opp_util1"] for row in rows) / len(rows),
                "opponent2_utility": sum(row["opp_util2"] for row in rows) / len(rows),
                "social_welfare": sum(row["social"] for row in rows) / len(rows),
                "negotiation_length": sum(row["step"] for row in rows) / len(rows),
            }
            metrics["negotiation_score"] = negotiation_score(
                metrics["agreement_rate"], metrics["agent_utility"], metrics["negotiation_length"],
                0.0, length_weight=length_weight, welfare_weight=0.0)
            overall.append(metrics)
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
    benchmark = {key: sum(row[key] for row in overall) / len(overall) for key in overall[0]}
    return dict(zip((entry.id for entry in pool.entries), overall)), benchmark, episode_rows
