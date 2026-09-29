"""Persistent scripted and policy-snapshot opponent pool."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from itertools import combinations, combinations_with_replacement
from pathlib import Path

import numpy as np

from .pfsp import negotiation_score, pfsp_probabilities


@dataclass
class PoolEntry:
    id: str
    kind: str
    name: str
    checkpoint_path: str | None = None
    added_step: int = 0
    matches: int = 0
    agreement_rate: float = 0.0
    agent_utility: float = 0.0
    opponent_utility: float = 0.0
    social_welfare: float = 0.0
    negotiation_length: float = 0.0
    negotiation_score: float = 0.0
    snapshot_agent_utility: float = 0.0
    snapshot_score: float = 0.0
    selected: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    learner_negotiation_score: float = 0.0
    evaluation_history: list = field(default_factory=list)

    def record(self, metrics, step, *, length_weight=-0.005, welfare_weight=0.1,
               dominance_tolerance=0.01):
        previous_matches = self.matches
        self.matches += 1
        for field_name, value in metrics.items():
            if field_name in ("agreement_rate", "agent_utility", "negotiation_length"):
                # Official Alpha-Nego uses alpha=0.1 moving opponent statistics.
                updated = float(value) if previous_matches == 0 else 0.9 * getattr(self, field_name) + 0.1 * float(value)
                setattr(self, field_name, updated)
        self.negotiation_score = negotiation_score(
            self.agreement_rate, self.agent_utility, self.negotiation_length,
            0.0, length_weight=length_weight, welfare_weight=0.0)
        self.learner_negotiation_score = self.negotiation_score
        public_metrics = {key: float(metrics[key]) for key in
                          ("agreement_rate", "agent_utility", "negotiation_length")}
        self.evaluation_history.append({"step": int(step), **public_metrics})
        self.evaluation_history = self.evaluation_history[-50:]


def is_dominant(current, incumbent, tolerance=0.01, min_agreement=0.0):
    if current["agreement_rate"] < min_agreement:
        return False
    if incumbent is None:
        return True
    difference = current["agent_utility"] - incumbent.snapshot_agent_utility
    if abs(difference) > tolerance:
        return difference > 0
    return current["negotiation_score"] > incumbent.snapshot_score


class OpponentPool:
    def __init__(self, root, names, *, max_size=16, pfsp_alpha=1.0, uniform_mix=0.1,
                 allow_duplicates=True, length_weight=-0.005, welfare_weight=0.1,
                 self_play_probability=0.0, scripted_probability=0.5,
                 snapshot_probability=0.5):
        self.root = Path(root)
        self.max_size = max_size
        self.pfsp_alpha = pfsp_alpha
        self.uniform_mix = uniform_mix
        self.allow_duplicates = allow_duplicates
        self.length_weight = length_weight
        self.welfare_weight = welfare_weight
        source_total = self_play_probability + scripted_probability + snapshot_probability
        if source_total <= 0:
            raise ValueError("At least one opponent source must have positive probability")
        self.self_play_probability = self_play_probability / source_total
        self.scripted_probability = scripted_probability / source_total
        self.snapshot_probability = snapshot_probability / source_total
        self.entries = [PoolEntry(f"scripted-{name}", "scripted", name) for name in dict.fromkeys(names)]
        if not self.entries:
            raise ValueError("At least one scripted opponent is required")

    def probabilities(self):
        return pfsp_probabilities(self.entries, self.pfsp_alpha, self.uniform_mix)

    def scripted_pairs(self):
        """Return the unordered scripted matchup space used at initialization."""
        scripted = [entry for entry in self.entries if entry.kind == "scripted"]
        factory = combinations_with_replacement if self.allow_duplicates else combinations
        return list(factory(scripted, 2))

    def sample_pair(self, rng, *, current_model_available=False):
        """Sample both SAOP slots from Alpha-Nego's source mixture and PFSP."""
        groups = []
        if current_model_available and self.self_play_probability > 0:
            groups.append((self.self_play_probability,
                           [PoolEntry("current-self-play", "self_play", "CurrentAlphaNego")],
                           np.ones(1)))
        scripted = [entry for entry in self.entries if entry.kind == "scripted"]
        snapshots = [entry for entry in self.entries if entry.kind == "snapshot"]
        if scripted and self.scripted_probability > 0:
            groups.append((self.scripted_probability, scripted,
                           pfsp_probabilities(scripted, self.pfsp_alpha, self.uniform_mix)))
        if snapshots and self.snapshot_probability > 0:
            groups.append((self.snapshot_probability, snapshots,
                           pfsp_probabilities(snapshots, self.pfsp_alpha, self.uniform_mix)))
        source_total = sum(weight for weight, _, _ in groups)
        choices, weights = [], []
        for source_weight, entries, within_source in groups:
            for entry, probability in zip(entries, within_source):
                choices.append(entry)
                weights.append(source_weight / source_total * float(probability))
        probabilities = np.asarray(weights, dtype=np.float64)
        probabilities /= probabilities.sum()
        if len(choices) < 2 and not self.allow_duplicates:
            raise ValueError("Two distinct opponents are required when duplicates are disabled")
        factory = combinations_with_replacement if self.allow_duplicates else combinations
        pairs = list(factory(range(len(choices)), 2))
        pair_probabilities = np.asarray([
            probabilities[first] * probabilities[second]
            for first, second in pairs
        ], dtype=np.float64)
        pair_probabilities /= pair_probabilities.sum()
        first, second = pairs[int(rng.choice(len(pairs), p=pair_probabilities))]
        if first != second and rng.random() < 0.5:
            first, second = second, first
        indices = (first, second)
        selected = [choices[index] for index in indices]
        for entry in selected:
            entry.selected += 1
        return selected[0], selected[1], tuple(float(probabilities[index]) for index in indices)

    def add_snapshot(self, step, actor_path, metrics):
        entry = PoolEntry(f"snapshot-{step}", "snapshot", f"Snapshot{step}",
                          checkpoint_path=str(Path(actor_path).relative_to(self.root.parent)), added_step=step)
        entry.record(metrics, step, length_weight=self.length_weight, welfare_weight=self.welfare_weight)
        entry.snapshot_agent_utility = metrics["agent_utility"]
        entry.snapshot_score = metrics["negotiation_score"]
        self.entries.append(entry)
        return self.prune()

    def prune(self):
        removed = []
        while len(self.entries) > self.max_size:
            snapshots = [e for e in self.entries if e.kind == "snapshot"]
            if not snapshots:
                raise ValueError("max_pool_size is smaller than scripted pool")
            victim = min(snapshots, key=lambda e: (e.negotiation_score, e.selected, e.added_step, e.id))
            self.entries.remove(victim)
            path = self.root.parent / victim.checkpoint_path
            if path.is_file():
                path.unlink()
            removed.append(victim.id)
        return removed

    def to_dict(self):
        return {"max_size": self.max_size, "pfsp_alpha": self.pfsp_alpha,
                "uniform_mix": self.uniform_mix, "allow_duplicates": self.allow_duplicates,
                "length_weight": self.length_weight, "welfare_weight": self.welfare_weight,
                "self_play_probability": self.self_play_probability,
                "scripted_probability": self.scripted_probability,
                "snapshot_probability": self.snapshot_probability,
                "entries": [asdict(entry) for entry in self.entries]}

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / "pool.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(destination)

    @classmethod
    def load(cls, root):
        root = Path(root)
        data = json.loads((root / "pool.json").read_text(encoding="utf-8"))
        obj = cls(root, [e["name"] for e in data["entries"] if e["kind"] == "scripted"],
                  max_size=data["max_size"], pfsp_alpha=data["pfsp_alpha"],
                  uniform_mix=data["uniform_mix"], allow_duplicates=data["allow_duplicates"],
                  length_weight=data["length_weight"], welfare_weight=data["welfare_weight"],
                  self_play_probability=data.get("self_play_probability", 0.0),
                  scripted_probability=data.get("scripted_probability", 0.5),
                  snapshot_probability=data.get("snapshot_probability", 0.5))
        obj.entries = [PoolEntry(**entry) for entry in data["entries"]]
        for entry in obj.entries:
            if entry.kind == "snapshot" and not (root.parent / entry.checkpoint_path).is_file():
                raise FileNotFoundError(root.parent / entry.checkpoint_path)
        return obj
