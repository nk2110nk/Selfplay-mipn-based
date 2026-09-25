import csv
from pathlib import Path

import numpy as np
import torch

from compat import load_genius_domain
from environment import NegotiationEnv
from opponent_pool.pool import OpponentPool
from results import FIELDS
from test_negotiator import evaluate
from train import layout, parse_args, train


def test_domains_and_observation():
    for name in ("Laptop", "ItexvsCypress", "IS_BT_Acquisition", "Grocery",
                 "thompson", "Car", "EnergySmall_A"):
        domain, utilities = load_genius_domain(name)
        assert domain and len(utilities) == 3
    obs_dim, nvec = layout(["Laptop", "Car"], "EnergySmall_A", "general")
    env = NegotiationEnv("Laptop", ".", obs_dim, nvec)
    assert env.action_space.nvec.tolist() == [len(issue.values) for issue in env.domain] + [2]
    assert env.observation_space.shape == (obs_dim,)
    assert env.current_mask()[-1, 0] == 0


def test_training_snapshot_resume_and_tsv(tmp_path):
    model_dir = tmp_path / "AlphaNego_Negotiator"
    args = parse_args(["-a", "Boulware", "Linear", "-i", "Laptop", "-sp", str(model_dir),
                       "--total-timesteps", "3", "--batch-size", "2", "--learning-starts", "1",
                       "--hidden", "16", "--quantiles", "8", "--pool-eval-freq", "2",
                       "--snapshot-freq", "2", "--pool-eval-episodes", "1", "--max-pool-size", "5"])
    assert train(args) == model_dir
    checkpoint = torch.load(model_dir / "checkpoint.pt", weights_only=False)
    assert checkpoint["global_step"] == 3
    pool = OpponentPool.load(model_dir / "pool")
    assert any(entry.kind == "snapshot" for entry in pool.entries)
    assert (model_dir / "evaluation" / "step-2.tsv").is_file()
    snapshot = next(entry for entry in pool.entries if entry.kind == "snapshot")
    env = NegotiationEnv("Laptop", model_dir, checkpoint["dsac"]["obs_dim"],
                         checkpoint["dsac"]["action_nvec"])
    env.reset((snapshot, pool.entries[0]))
    action = np.zeros(len(checkpoint["dsac"]["action_nvec"]), dtype=np.int64)
    action[-1] = 1
    env.step(action)
    args = parse_args(["--resume", str(model_dir), "-a", "Boulware", "Linear", "-i", "Laptop",
                       "--total-timesteps", "4"])
    train(args)
    assert torch.load(model_dir / "checkpoint.pt", weights_only=False)["global_step"] == 4
    for style in ("neutral", "aggressive", "conservative"):
        files = evaluate(model_dir, ["Boulware", "Linear"], ["Laptop"], episodes=1,
                         style=style, candidates=3, case=1)
        assert len(files) == 1
        with files[0].open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            assert tuple(reader.fieldnames[:7]) == FIELDS[:7]
            rows = list(reader)
            assert len(rows) == 1 and rows[0]["style"] == style
        case_path = model_dir / "results_alpha-nego-based" / "expert" / "Boulware-Linear" / "Laptop" / "case1" / files[0].name
        assert case_path.is_file()
