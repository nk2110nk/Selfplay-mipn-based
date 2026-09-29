import json

import numpy as np

from opponent_pool.pfsp import dominance_probability, negotiation_score, pfsp_probabilities
from opponent_pool.pool import OpponentPool, PoolEntry, is_dominant


def metrics(value):
    return {"agreement_rate": 0.5, "agent_utility": value,
            "opponent_utility": 0.8, "social_welfare": 2.0,
            "negotiation_length": 20, "negotiation_score": 1.5 + value}


def test_pfsp_and_fallback():
    easy = PoolEntry("easy", "scripted", "Linear")
    hard = PoolEntry("hard", "scripted", "Boulware")
    assert np.allclose(pfsp_probabilities([easy, hard]), [0.5, 0.5])
    easy.record(metrics(0.9), 1)
    hard.record(metrics(0.1), 1)
    probs = pfsp_probabilities([easy, hard], alpha=3, uniform_mix=0.1)
    assert probs[1] > probs[0] and np.isclose(probs.sum(), 1)
    assert dominance_probability(hard) > dominance_probability(easy)
    hard.opponent_utility = 0.0
    hard.social_welfare = 0.0
    unchanged = dominance_probability(hard)
    hard.opponent_utility = 1.0
    hard.social_welfare = 3.0
    assert dominance_probability(hard) == unchanged
    hard.agent_utility = float("nan")
    assert np.allclose(pfsp_probabilities([easy, hard]), [0.5, 0.5])
    assert negotiation_score(1, 0.5, 10, 2) > 0


def test_pool_persistence_prune_and_dominance(tmp_path):
    root = tmp_path / "pool"
    pool = OpponentPool(root, ["Boulware", "Linear"], max_size=3)
    first, second, probabilities = pool.sample_pair(np.random.default_rng(1))
    assert first.id != second.id and all(0 < p <= 1 for p in probabilities)
    first_path = root / "snapshots" / "snapshot-1.pt"
    first_path.parent.mkdir(parents=True)
    first_path.write_bytes(b"first")
    pool.add_snapshot(1, first_path, metrics(0.4))
    assert is_dominant(metrics(0.5), pool.entries[-1])
    assert not is_dominant(metrics(0.39), pool.entries[-1])
    pool.save()
    loaded = OpponentPool.load(root)
    assert loaded.to_dict() == pool.to_dict()
    second_path = root / "snapshots" / "snapshot-2.pt"
    second_path.write_bytes(b"second")
    removed = pool.add_snapshot(2, second_path, metrics(0.9))
    assert removed == ["snapshot-1"] and not first_path.exists()
    assert second_path.exists()
    assert len(pool.entries) == 3


def test_three_source_current_self_play(tmp_path):
    pool = OpponentPool(tmp_path / "pool", ["Boulware", "Linear"], max_size=3,
                        self_play_probability=1.0, scripted_probability=0.0,
                        snapshot_probability=0.0, allow_duplicates=True)
    first, second, probabilities = pool.sample_pair(
        np.random.default_rng(3), current_model_available=True)
    assert first.kind == second.kind == "self_play"
    assert probabilities == (1.0, 1.0)
