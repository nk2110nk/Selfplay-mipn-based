"""Evaluate one checkpoint with scripted opponents in the shared TSV format."""

import argparse
import random
import shutil
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import torch

from compat import SCRIPTED, load_genius_domain
from dsac import DSAC
from environment import NegotiationEnv
from opponent_pool.pool import PoolEntry
from policy import domain_nvec
from results import result_path, row_from_info, write_results


def normalize_case(value):
    if value is None:
        return None
    text = str(value)
    return text if text.startswith("case") else f"case{text}"


def validate_domains(issues, model, config):
    """Fail before evaluation when a checkpoint cannot encode a domain."""
    for issue in issues:
        if config["model_type"] == "expert" and issue not in config["issues"]:
            raise ValueError(f"Expert checkpoint cannot evaluate domain {issue}")
        domain = load_genius_domain(issue)[0]
        try:
            domain_nvec(domain, model.nvec)
        except ValueError as error:
            counts = [len(item.values) for item in domain]
            raise ValueError(
                f"Checkpoint action space {list(model.nvec)} cannot represent {issue} "
                f"with per-issue values {counts}. Retrain a general model with "
                f"--compatible-domains including {issue}."
            ) from error
        required_obs_dim = sum(len(item.values) for item in domain) * 6 + 1
        if required_obs_dim > model.obs_dim:
            raise ValueError(
                f"Checkpoint observation size {model.obs_dim} cannot represent {issue} "
                f"(requires {required_obs_dim}). Retrain a general model with "
                f"--compatible-domains including {issue}."
            )


def evaluate(model_path, agents, issues, episodes=100, *, style="neutral", seed=0,
             device="auto", candidates=64, aggressive_quantile=0.8,
             conservative_quantile=0.2, risk_weight=0.1, noise=False, deterministic=False,
             case=None, export_root=None):
    model_path = Path(model_path)
    if model_path.is_dir():
        model_path = model_path / "checkpoint.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    configured_case = config.get("case", "case1")
    requested_case = normalize_case(case)
    if requested_case is not None and requested_case != configured_case:
        raise ValueError(f"Checkpoint is {configured_case}, not {requested_case}")
    state = checkpoint["dsac"]
    entropy = state["entropy_parameters"]
    coefficient = (float(entropy["log_alpha"].exp().mean()) if "log_alpha" in entropy
                   else entropy["coefficient"])
    model = DSAC(state["obs_dim"], state["action_nvec"], device=device,
                 quantiles=state["quantiles"], hidden=state["hidden"], buffer_size=1,
                 dropout=state.get("dropout", 0.1), entropy_coefficient=coefficient,
                 auto_entropy=entropy.get("auto", False),
                 target_entropy_ratio=state.get("target_entropy_ratio", 0.98),
                 target_entropy_final_ratio=state.get("target_entropy_final_ratio"),
                 entropy_anneal_steps=state.get("entropy_anneal_steps", 0),
                 canonical_accept_action=state.get("canonical_accept_action", False),
                 hierarchical_entropy=state.get("hierarchical_entropy", False),
                 gradient_clip=state.get("gradient_clip", 1.0),
                 policy_update_frequency=state.get("policy_update_frequency", 2),
                 gumbel_temperature=state.get("gumbel_temperature", 1.0),
                 training_style=state.get("training_style", "neutral"),
                 aggressive_quantile=state.get("aggressive_quantile", 0.8),
                 conservative_quantile=state.get("conservative_quantile", 0.2),
                 risk_weight=state.get("risk_weight", 0.1))
    model.load_state_dict(state, with_replay=False)
    model.actor.eval()
    for critic in model.critics:
        critic.eval()
    validate_domains(issues, model, config)
    model_dir = model_path.parent
    result_files = []
    pairs = [tuple(agents)] if len(agents) == 2 else list(combinations_with_replacement(agents, 2))
    for issue in issues:
        for pair in pairs:
            opponents = tuple(PoolEntry(f"scripted-{name}", "scripted", name) for name in pair)
            rows = []
            for episode in range(episodes):
                episode_seed = seed + episode
                random.seed(episode_seed)
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                env = NegotiationEnv(issue, model_dir, model.obs_dim, model.nvec,
                                     case=configured_case, device=device, test=True, noise=noise)
                observation = env.reset(opponents)
                done = False
                while not done:
                    action = model.act(observation, env.current_mask(), style=style, candidates=candidates,
                                       aggressive_quantile=aggressive_quantile,
                                       conservative_quantile=conservative_quantile,
                                       risk_weight=risk_weight)
                    observation, _, done, info = env.step(action)
                rows.append(row_from_info(info, style, opponents, episode_seed, domain=issue,
                                          case=configured_case,
                                          length_weight=config["score_length_weight"],
                                          welfare_weight=config["score_welfare_weight"]))
            path = result_path(model_dir, pair, issue, deterministic, noise)
            write_results(path, rows)
            result_files.append(path)
            if requested_case is not None:
                root = Path(export_root) if export_root else model_dir / "results_alpha-nego-based"
                target = root / config["model_type"] / f"{pair[0]}-{pair[1]}" / issue / requested_case / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    return result_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "--model", "-m", required=True)
    parser.add_argument("--agents", "-a", nargs="+", required=True, choices=SCRIPTED)
    parser.add_argument("--issues", "--issue", "-i", nargs="+", required=True)
    parser.add_argument("--episodes", "-e", type=int, default=100)
    parser.add_argument("--style", choices=("neutral", "aggressive", "conservative"), default="neutral")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--aggressive-quantile", type=float, default=0.8)
    parser.add_argument("--conservative-quantile", type=float, default=0.2)
    parser.add_argument("--risk-weight", type=float, default=0.1)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--case", choices=("1", "2", "3", "case1", "case2", "case3"))
    parser.add_argument("--export-root")
    args = parser.parse_args()
    if not 0 <= args.aggressive_quantile < 1 or not 0 < args.conservative_quantile <= 1:
        parser.error("Invalid style quantile")
    for path in evaluate(args.model_path, args.agents, args.issues, args.episodes,
                         style=args.style, seed=args.seed, device=args.device,
                         candidates=args.candidates, aggressive_quantile=args.aggressive_quantile,
                         conservative_quantile=args.conservative_quantile, risk_weight=args.risk_weight,
                         noise=args.noise, case=args.case, export_root=args.export_root):
        print(path)


if __name__ == "__main__":
    main()
