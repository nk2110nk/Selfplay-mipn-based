import json

import numpy as np

from opponent_pool.pfsp import dominance_probability, negotiation_score, pfsp_probabilities
from opponent_pool.pool import OpponentPool, PoolEntry, is_dominant, snapshot_pruning_key


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
    pool = OpponentPool(root, ["Boulware", "Linear"], max_size=3, allow_duplicates=False)
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


def test_pruning_keeps_hard_snapshot_and_has_deterministic_tiebreaks():
    hard = PoolEntry("hard", "snapshot", "Hard", added_step=20, selected=8)
    easy = PoolEntry("easy", "snapshot", "Easy", added_step=10, selected=1)
    hard.record({"agreement_rate": 0.2, "agent_utility": 0.1,
                 "negotiation_length": 70}, 1)
    easy.record({"agreement_rate": 0.9, "agent_utility": 0.9,
                 "negotiation_length": 10}, 1)
    assert snapshot_pruning_key(easy) < snapshot_pruning_key(hard)

    tied_new = PoolEntry("new", "snapshot", "New", added_step=20, selected=1)
    tied_old = PoolEntry("old", "snapshot", "Old", added_step=10, selected=1)
    for entry in (tied_new, tied_old):
        entry.record({"agreement_rate": 0.5, "agent_utility": 0.5,
                      "negotiation_length": 40}, 1)
    assert snapshot_pruning_key(tied_old) < snapshot_pruning_key(tied_new)


def test_add_snapshot_prunes_easy_evaluated_opponent_and_protects_new(tmp_path):
    root = tmp_path / "pool"
    snapshots = root / "snapshots"
    snapshots.mkdir(parents=True)
    pool = OpponentPool(root, ["Boulware"], max_size=3)

    hard_path = snapshots / "snapshot-1.pt"
    easy_path = snapshots / "snapshot-2.pt"
    new_path = snapshots / "snapshot-3.pt"
    for path in (hard_path, easy_path, new_path):
        path.write_bytes(path.name.encode())

    pool.add_snapshot(1, hard_path, metrics(0.4))
    hard = next(entry for entry in pool.entries if entry.id == "snapshot-1")
    hard.record({"agreement_rate": 0.2, "agent_utility": 0.1,
                 "negotiation_length": 70}, 10)
    pool.add_snapshot(2, easy_path, metrics(0.8))
    easy = next(entry for entry in pool.entries if entry.id == "snapshot-2")
    easy.record({"agreement_rate": 0.9, "agent_utility": 0.9,
                 "negotiation_length": 10}, 20)

    removed = pool.add_snapshot(3, new_path, metrics(0.9))
    assert removed == ["snapshot-2"]
    assert hard_path.exists() and new_path.exists() and not easy_path.exists()
    assert {entry.id for entry in pool.entries if entry.kind == "snapshot"} == {
        "snapshot-1", "snapshot-3"
    }
    newest = next(entry for entry in pool.entries if entry.id == "snapshot-3")
    assert newest.matches == 0


def test_scripted_pair_space_includes_duplicates(tmp_path):
    pool = OpponentPool(tmp_path / "pool", ["Boulware", "Conceder", "Linear"], max_size=4)
    pairs = [(first.name, second.name) for first, second in pool.scripted_pairs()]
    assert pairs == [
        ("Boulware", "Boulware"),
        ("Boulware", "Conceder"),
        ("Boulware", "Linear"),
        ("Conceder", "Conceder"),
        ("Conceder", "Linear"),
        ("Linear", "Linear"),
    ]
    sampled = pool.sample_pair(np.random.default_rng(3), current_model_available=True)
    assert sampled[0].kind == sampled[1].kind == "scripted"
    counts = {tuple(sorted((first.name, second.name))): 0 for first, second in pool.scripted_pairs()}
    rng = np.random.default_rng(4)
    for _ in range(6000):
        first, second, _ = pool.sample_pair(rng)
        counts[tuple(sorted((first.name, second.name)))] += 1
    assert max(counts.values()) / min(counts.values()) < 1.2


def test_three_source_current_self_play(tmp_path):
    pool = OpponentPool(tmp_path / "pool", ["Boulware", "Linear"], max_size=3,
                        self_play_probability=1.0, scripted_probability=0.0,
                        snapshot_probability=0.0, allow_duplicates=True)
    first, second, probabilities = pool.sample_pair(
        np.random.default_rng(3), current_model_available=True)
    assert first.kind == second.kind == "self_play"
    assert probabilities == (1.0, 1.0)
