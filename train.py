"""Train Alpha-Nego adaptation with distributional SAC and PFSP."""

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from compat import SCRIPTED, load_genius_domain
from dsac import DSAC
from environment import NegotiationEnv
from opponent_pool.evaluator import evaluate_pool
from opponent_pool.pool import OpponentPool, is_dominant
from results import row_from_info, write_results


LOG_FIELDS = ("global_step", "episode", "actor_loss", "critic_loss", "entropy", "alpha", "alpha_loss",
              "target_entropy_ratio",
              "replay_buffer_size", "agreement_rate", "agent_utility", "opponent1_utility",
              "opponent2_utility", "social_welfare", "negotiation_length", "opponent1_id",
              "opponent2_id", "opponent1_probability", "opponent2_probability", "pool_size", "event")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def layout(issues, general_domain, model_type):
    target_name = general_domain if model_type == "general" else issues[0]
    target = load_genius_domain(target_name)[0]
    nvec = [len(issue.values) for issue in target] + [2]
    obs_dim = sum(len(issue.values) for issue in target) * 6 + 1
    for name in issues:
        domain = load_genius_domain(name)[0]
        if len(domain) > len(target):
            raise ValueError(f"{name} has more issues than padding domain {target_name}")
        if any(len(issue.values) > nvec[index] for index, issue in enumerate(domain)):
            raise ValueError(f"{name} has an action head larger than padding domain {target_name}")
        if sum(len(issue.values) for issue in domain) * 6 + 1 > obs_dim:
            raise ValueError(f"{name} observation exceeds padding domain {target_name}")
    return obs_dim, nvec


def build_model_dir(save_path, issues, agents, resume=None):
    if resume:
        path = Path(resume)
        return path if path.is_dir() else path.parent
    root = Path(save_path)
    if root.name == "AlphaNego_Negotiator":
        return root
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / f"{'-'.join(issues)}_{'-'.join(agents)}" / f"{stamp}-TA" / "AlphaNego_Negotiator"


def save_checkpoint(model_dir, model, pool, config, global_step, episode, rng):
    pool.save()
    state = {"dsac": model.state_dict(), "config": config, "global_step": global_step,
             "episode": episode, "obs_space_shape": [model.obs_dim],
             "action_nvec": list(model.nvec), "issues": config["issues"],
             "agents": config["agents"], "general_domain": config["general_domain"],
             "obs_layout": "padded_time_last", "pool_metadata": pool.to_dict(),
             "random_states": {"python": random.getstate(), "numpy": np.random.get_state(),
                               "torch": torch.get_rng_state(),
                               "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                               "sampler": rng.bit_generator.state}}
    path = model_dir / "checkpoint.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def snapshot(model_dir, model, config, step, metrics):
    path = model_dir / "pool" / "snapshots" / f"snapshot-{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": model.actor.state_dict(), "critics": [c.state_dict() for c in model.critics],
                "obs_dim": model.obs_dim, "action_nvec": list(model.nvec),
                "quantiles": model.quantiles, "hidden": model.hidden,
                "dropout": model.dropout, "architecture_version": 2,
                "obs_layout": "padded_time_last", "issues": config["issues"],
                "general_domain": config["general_domain"], "model_type": config["model_type"],
                "benchmark": metrics}, path)
    return path


def append_log(path, row):
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", "-a", nargs="+", choices=SCRIPTED, default=["Boulware", "Linear"])
    parser.add_argument("--issue", "--issues", "-i", nargs="+", default=["Laptop"])
    parser.add_argument("--save-path", "--save_path", "-sp", default="results")
    parser.add_argument("--resume")
    parser.add_argument("--total-timesteps", "--timesteps", "-t", type=int, default=100_000)
    parser.add_argument("--num-envs", "--n-envs", "--n_envs", "-n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-type", choices=("expert", "general"), default="expert")
    parser.add_argument("--general-domain", "--general_domain", default="EnergySmall_A")
    parser.add_argument("--style", choices=("neutral", "aggressive", "conservative"), default="neutral")
    parser.add_argument("--quantiles", type=int, default=64)
    parser.add_argument("--actor-lr", type=float, default=3e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-buffer-size", type=int, default=1_000_000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--auto-entropy", action="store_true")
    parser.add_argument("--target-entropy-ratio", type=float, default=0.98)
    parser.add_argument("--target-entropy-final-ratio", type=float)
    parser.add_argument("--entropy-anneal-steps", type=int, default=0)
    parser.add_argument("--canonical-accept-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hierarchical-entropy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--policy-update-frequency", type=int, default=2)
    parser.add_argument("--gumbel-temperature", type=float, default=1.0)
    parser.add_argument("--aggressive-quantile", type=float, default=0.8)
    parser.add_argument("--conservative-quantile", type=float, default=0.2)
    parser.add_argument("--risk-weight", type=float, default=0.1)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--pool-eval-freq", type=int, default=10_000)
    parser.add_argument("--pool-eval-episodes", type=int, default=4)
    parser.add_argument("--snapshot-freq", type=int, default=10_000)
    parser.add_argument("--max-pool-size", type=int, default=16)
    parser.add_argument("--dominance-tolerance", type=float, default=0.01)
    parser.add_argument("--min-agreement-rate", type=float, default=0.0)
    parser.add_argument("--allow-duplicate-opponents", action="store_true")
    parser.add_argument("--pfsp-alpha", type=float, default=1.0)
    parser.add_argument("--uniform-mix", type=float, default=0.1)
    parser.add_argument("--self-play-probability", type=float, default=0.2)
    parser.add_argument("--scripted-probability", type=float, default=0.3)
    parser.add_argument("--snapshot-probability", type=float, default=0.5)
    parser.add_argument("--score-length-weight", type=float, default=-0.005)
    parser.add_argument("--score-welfare-weight", type=float, default=0.0,
                        help=argparse.SUPPRESS)  # Old command/checkpoint compatibility only.
    args = parser.parse_args(argv)
    if args.model_type == "expert" and len(args.issue) != 1:
        parser.error("Expert mode needs exactly one issue")
    if args.total_timesteps < 1 or args.num_envs < 1 or args.batch_size < 1 or args.quantiles < 2 or args.pool_eval_episodes < 1:
        parser.error("Positive timesteps/envs/batch size and at least two quantiles required")
    if args.max_pool_size < 5 or not 0 <= args.uniform_mix <= 1:
        parser.error("Pool needs room for four scripted opponents, one snapshot and a valid uniform mix")
    final_entropy_ratio = (args.target_entropy_ratio if args.target_entropy_final_ratio is None
                           else args.target_entropy_final_ratio)
    if (args.entropy_coefficient <= 0 or not 0 < args.target_entropy_ratio <= 1 or
            not 0 < final_entropy_ratio <= 1 or args.entropy_anneal_steps < 0 or
            args.gradient_clip <= 0 or args.policy_update_frequency < 1 or
            args.gumbel_temperature <= 0 or not 0 <= args.dropout < 1):
        parser.error("Invalid DSAC entropy, update, clipping, temperature, or dropout setting")
    if (not 0 <= args.aggressive_quantile < 1 or not 0 < args.conservative_quantile <= 1 or
            min(args.self_play_probability, args.scripted_probability, args.snapshot_probability) < 0 or
            args.self_play_probability + args.scripted_probability + args.snapshot_probability <= 0):
        parser.error("Invalid style quantile or opponent source probability")
    return args


def train(args):
    model_dir = build_model_dir(args.save_path, args.issue, args.agents, args.resume)
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard = SummaryWriter(str(model_dir / "tensorboard")) if SummaryWriter is not None else None
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(model_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        for key in ("issues", "agents", "general_domain", "model_type"):
            current = args.issue if key == "issues" else args.agents if key == "agents" else getattr(args, key)
            if current != config[key]:
                raise ValueError(f"Resume configuration differs for {key}: {current} != {config[key]}")
    else:
        config = vars(args).copy()
        config["issues"] = config.pop("issue")
        config.pop("resume")
        (model_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    seed_everything(config["seed"])
    obs_dim, nvec = layout(config["issues"], config["general_domain"], config["model_type"])
    model = DSAC(obs_dim, nvec, device=args.device, quantiles=config["quantiles"],
                 actor_lr=config["actor_lr"], critic_lr=config["critic_lr"],
                 batch_size=config["batch_size"], buffer_size=config["replay_buffer_size"],
                 tau=config["tau"], gamma=config["gamma"],
                 entropy_coefficient=config["entropy_coefficient"], hidden=config["hidden"],
                 dropout=config.get("dropout", 0.1), auto_entropy=config.get("auto_entropy", False),
                 target_entropy_ratio=config.get("target_entropy_ratio", 0.98),
                 target_entropy_final_ratio=config.get("target_entropy_final_ratio"),
                 entropy_anneal_steps=config.get("entropy_anneal_steps", 0),
                 canonical_accept_action=config.get("canonical_accept_action", False),
                 hierarchical_entropy=config.get("hierarchical_entropy", False),
                 gradient_clip=config.get("gradient_clip", 1.0),
                 policy_update_frequency=config.get("policy_update_frequency", 2),
                 gumbel_temperature=config.get("gumbel_temperature", 1.0),
                 training_style=config["style"],
                 aggressive_quantile=config.get("aggressive_quantile", 0.8),
                 conservative_quantile=config.get("conservative_quantile", 0.2),
                 risk_weight=config.get("risk_weight", 0.1))
    pool_root = model_dir / "pool"
    if checkpoint:
        model.load_state_dict(checkpoint["dsac"])
        pool = OpponentPool.load(pool_root)
        if pool.to_dict() != checkpoint["pool_metadata"]:
            raise ValueError("Pool JSON and checkpoint metadata differ")
        random.setstate(checkpoint["random_states"]["python"])
        np.random.set_state(checkpoint["random_states"]["numpy"])
        torch.set_rng_state(checkpoint["random_states"]["torch"])
        if torch.cuda.is_available() and checkpoint["random_states"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["random_states"]["cuda"])
        global_step = checkpoint["global_step"]
        episode = checkpoint["episode"]
    else:
        pool = OpponentPool(pool_root, ("Boulware", "Linear", "Conceder", "Atlas3"),
                            max_size=config["max_pool_size"], pfsp_alpha=config["pfsp_alpha"],
                            uniform_mix=config["uniform_mix"],
                            allow_duplicates=config["allow_duplicate_opponents"],
                            length_weight=config["score_length_weight"],
                            welfare_weight=config["score_welfare_weight"],
                            self_play_probability=config["self_play_probability"],
                            scripted_probability=config["scripted_probability"],
                            snapshot_probability=config["snapshot_probability"])
        pool.save()
        global_step = 0
        episode = 0
    rng = np.random.default_rng(config["seed"] + global_step)
    if checkpoint:
        rng.bit_generator.state = checkpoint["random_states"]["sampler"]
    if not (model_dir / "training_log.csv").exists():
        with (model_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=LOG_FIELDS).writeheader()

    def new_episode():
        domain = config["issues"][int(rng.integers(len(config["issues"])))]
        first, second, probabilities = pool.sample_pair(rng, current_model_available=True)
        env = NegotiationEnv(domain, model_dir, obs_dim, nvec, device=args.device)
        return {"env": env, "obs": env.reset((first, second), current_model=model), "opponents": (first, second),
                "probabilities": probabilities}

    workers = [new_episode() for _ in range(args.num_envs)]
    last_loss = {"actor_loss": "", "critic_loss": "", "entropy": "", "alpha": model.alpha,
                 "alpha_loss": "", "target_entropy_ratio": model.current_target_entropy_ratio}
    while global_step < args.total_timesteps:
        for index, worker in enumerate(workers):
            if global_step >= args.total_timesteps:
                break
            env = worker["env"]
            mask = env.current_mask()
            action = model.act(worker["obs"], mask)
            next_obs, reward, done, info = env.step(action)
            next_mask = env.current_mask()
            model.replay.add(worker["obs"], action, reward, next_obs, done, mask, next_mask)
            worker["obs"] = next_obs
            global_step += 1
            if global_step >= config["learning_starts"]:
                for _ in range(config["gradient_steps"]):
                    updated = model.update()
                    if updated:
                        last_loss = updated
            if done:
                episode += 1
                first, second = worker["opponents"]
                for entry in (first, second):
                    entry.record({"agreement_rate": float(info["agreement"] is not None),
                                  "agent_utility": info["my_util"],
                                  "negotiation_length": info["step"]}, global_step,
                                 length_weight=pool.length_weight, welfare_weight=pool.welfare_weight)
                append_log(model_dir / "training_log.csv", {
                    "global_step": global_step, "episode": episode, **last_loss,
                    "replay_buffer_size": len(model.replay),
                    "agreement_rate": float(info["agreement"] is not None),
                    "agent_utility": info["my_util"], "opponent1_utility": info["opp_util1"],
                    "opponent2_utility": info["opp_util2"], "social_welfare": info["social"],
                    "negotiation_length": info["step"], "opponent1_id": first.id,
                    "opponent2_id": second.id,
                    "opponent1_probability": worker["probabilities"][0],
                    "opponent2_probability": worker["probabilities"][1],
                    "pool_size": len(pool.entries), "event": "episode"})
                if tensorboard is not None:
                    tensorboard.add_scalar("train/agent_utility", info["my_util"], global_step)
                    tensorboard.add_scalar("train/agreement", float(info["agreement"] is not None), global_step)
                    tensorboard.add_scalar("train/actor_loss", last_loss["actor_loss"] or 0.0, global_step)
                    tensorboard.add_scalar("train/critic_loss", last_loss["critic_loss"] or 0.0, global_step)
                    tensorboard.add_scalar("pool/size", len(pool.entries), global_step)
                workers[index] = new_episode()
            if (config["pool_eval_freq"] > 0 and global_step % config["pool_eval_freq"] == 0 or
                    config["snapshot_freq"] > 0 and global_step % config["snapshot_freq"] == 0):
                measurements, benchmark, evaluation_rows = evaluate_pool(
                    model, pool, config["issues"], config["pool_eval_episodes"], model_dir,
                    seed=config["seed"] + global_step, device=args.device,
                    length_weight=pool.length_weight, welfare_weight=pool.welfare_weight)
                for entry in pool.entries:
                    entry.record(measurements[entry.id], global_step,
                                 length_weight=pool.length_weight, welfare_weight=pool.welfare_weight)
                write_results(model_dir / "evaluation" / f"step-{global_step}.tsv", [
                    row_from_info(info, "neutral", (entry, anchor), episode_seed,
                                  length_weight=pool.length_weight,
                                  welfare_weight=pool.welfare_weight)
                    for entry, anchor, _, episode_seed, info in evaluation_rows])
                event = "pool_evaluation"
                if config["snapshot_freq"] > 0 and global_step % config["snapshot_freq"] == 0:
                    incumbents = [e for e in pool.entries if e.kind == "snapshot"]
                    incumbent = max(incumbents, key=lambda e: (e.snapshot_agent_utility, e.snapshot_score)) if incumbents else None
                    if is_dominant(benchmark, incumbent, config["dominance_tolerance"],
                                   config["min_agreement_rate"]):
                        path = snapshot(model_dir, model, config, global_step, benchmark)
                        removed = pool.add_snapshot(global_step, path, benchmark)
                        event = f"snapshot_added:{global_step}"
                        if removed:
                            event += ";pruned:" + ",".join(removed)
                append_log(model_dir / "training_log.csv", {"global_step": global_step,
                           "episode": episode, **last_loss, "replay_buffer_size": len(model.replay),
                           "agreement_rate": benchmark["agreement_rate"],
                           "agent_utility": benchmark["agent_utility"],
                           "opponent1_utility": benchmark["opponent_utility"],
                           "opponent2_utility": benchmark["opponent2_utility"],
                           "social_welfare": benchmark["social_welfare"],
                           "negotiation_length": benchmark["negotiation_length"],
                           "pool_size": len(pool.entries), "event": event})
                save_checkpoint(model_dir, model, pool, config, global_step, episode, rng)
    save_checkpoint(model_dir, model, pool, config, global_step, episode, rng)
    if tensorboard is not None:
        tensorboard.close()
    return model_dir


def main():
    args = parse_args()
    print(train(args))


if __name__ == "__main__":
    main()
