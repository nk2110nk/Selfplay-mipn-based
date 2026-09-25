"""Three-party SAOP environment using MiPN's domains and mechanism."""

from pathlib import Path

import numpy as np
import torch
import gym
from negmas.outcomes import ResponseType
from negmas.sao import SAONegotiator

from compat import MySAOMechanism, load_genius_domain, scripted_opponent
from dsac import DSAC
from policy import action_mask, domain_nvec, padded_observation


class BidObserver:
    """MiPN OnehotObserve2nT layout: two newest offers per role, then time."""

    def __init__(self, domain, own_name, opponent_names):
        self.domain = domain
        self.names = (own_name, *opponent_names)
        self.bits = [(issue.name, value) for issue in domain for value in issue.values]
        self.history = np.zeros((3, 2, len(self.bits)), dtype=np.float32)
        self.time = 0.0
        self.seen_step = None

    def reset(self):
        self.history.fill(0)
        self.time = 0.0
        self.seen_step = None

    def feed(self, state):
        if state is None:
            return
        self.time = float(state.get("relative_time", self.time))
        offer = state.get("current_offer")
        if offer is None:
            return
        step = state.get("step")
        if step == self.seen_step:
            return
        self.seen_step = step
        if "new_offers" in state and not state["new_offers"]:
            return
        proposer = str(state.get("current_proposer", ""))
        matches = [i for i, name in enumerate(self.names) if name == proposer or proposer.startswith(name + "-")]
        if not matches:
            raise ValueError(f"Unknown proposer: {proposer}; expected {self.names}")
        role = matches[0]
        self.history[role, 1] = self.history[role, 0].copy()
        self.history[role, 0] = [float(offer.get(issue) == value) for issue, value in self.bits]

    def vector(self):
        return np.concatenate((self.history.reshape(-1), np.asarray([self.time], dtype=np.float32)))


class ActionNegotiator(SAONegotiator):
    def __init__(self, name):
        super().__init__(name=name)
        self.action = None
        self.domain = None

    def set_action(self, action, domain):
        self.action = np.asarray(action, dtype=np.int64)
        self.domain = domain

    def respond(self, state, offer):
        if offer is not None and self.action is not None and self.action[-1] == 0:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state):
        if self.action is None:
            raise RuntimeError("No action has been assigned")
        return {issue.name: issue.values[int(value)] for issue, value in zip(self.domain, self.action)}


class PolicyNegotiator(ActionNegotiator):
    def __init__(self, name, model, obs_dim, nvec, *, style="neutral", candidates=16,
                 aggressive_quantile=0.8, conservative_quantile=0.2, risk_weight=0.1):
        super().__init__(name)
        self.model = model
        self.obs_dim = obs_dim
        self.nvec = tuple(nvec)
        self.style = style
        self.candidates = candidates
        self.aggressive_quantile = aggressive_quantile
        self.conservative_quantile = conservative_quantile
        self.risk_weight = risk_weight
        self.observer = None

    def prepare(self, domain, opponent_names):
        self.domain = domain
        self.valid_nvec = domain_nvec(domain, self.nvec)
        self.observer = BidObserver(domain, self.name, opponent_names)
        self.action = None

    def _choose(self, state):
        state_dict = None if state is None else state.__dict__
        self.observer.feed(state_dict)
        obs = padded_observation(self.observer.vector(), self.obs_dim)
        can_accept = state_dict is not None and state_dict.get("current_offer") is not None
        mask = action_mask(self.nvec, self.valid_nvec, can_accept).numpy()
        self.action = self.model.act(obs, mask, style=self.style, candidates=self.candidates,
                                     aggressive_quantile=self.aggressive_quantile,
                                     conservative_quantile=self.conservative_quantile,
                                     risk_weight=self.risk_weight)

    def respond(self, state, offer):
        self._choose(state)
        return super().respond(state, offer)

    def propose(self, state):
        if self.action is None:
            self._choose(state)
        return super().propose(state)


def load_snapshot(entry, model_dir, expected_obs_dim, expected_nvec, domain_name, device="cpu"):
    path = Path(model_dir) / entry.checkpoint_path
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["obs_layout"] != "padded_time_last" or state["obs_dim"] != expected_obs_dim or tuple(state["action_nvec"]) != tuple(expected_nvec):
        raise ValueError(f"Incompatible snapshot: {path}")
    if state["model_type"] == "expert" and domain_name not in state["issues"]:
        raise ValueError(f"Snapshot {path} was trained for {state['issues']}, not {domain_name}")
    model = DSAC(state["obs_dim"], state["action_nvec"], device=device, quantiles=state["quantiles"],
                 hidden=state["hidden"], dropout=state.get("dropout", 0.1), buffer_size=1)
    model.actor.load_state_dict(state["actor"])
    for critic, weights in zip(model.critics, state["critics"]):
        critic.load_state_dict(weights)
    model.actor.eval()
    for critic in model.critics:
        critic.eval()
    return model


class NegotiationEnv:
    def __init__(self, domain_name, model_dir, obs_dim, nvec, *, device="cpu", test=False, noise=False):
        self.domain_name = domain_name
        self.model_dir = Path(model_dir)
        self.obs_dim = obs_dim
        self.nvec = tuple(nvec)
        self.device = device
        self.test = test
        self.noise = noise
        self.domain, self.utilities = load_genius_domain(domain_name)
        self.valid_nvec = domain_nvec(self.domain, self.nvec)
        self.observation_space_shape = (obs_dim,)
        self.action_space_nvec = tuple(len(issue.values) for issue in self.domain) + (2,)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete(self.action_space_nvec)
        self.session = None
        self.state = None

    def reset(self, opponents, *, current_model=None):
        self.domain, self.utilities = load_genius_domain(self.domain_name)
        self.session = MySAOMechanism(issues=self.domain, n_steps=80, avoid_ultimatum=False)
        self.learner = ActionNegotiator("RLAgent")
        agents = []
        for slot, entry in enumerate(opponents):
            if entry.kind == "scripted":
                agents.append(scripted_opponent(entry.name, slot, noise=self.noise))
            elif entry.kind == "self_play":
                if current_model is None:
                    raise ValueError("A current model is required for self-play opponents")
                agents.append(PolicyNegotiator(f"CurrentAlphaNego-{slot}", current_model,
                                               self.obs_dim, self.nvec, style=None))
            else:
                model = load_snapshot(entry, self.model_dir, self.obs_dim, self.nvec, self.domain_name, self.device)
                agents.append(PolicyNegotiator(f"Snapshot{entry.added_step}-{slot}", model,
                                               self.obs_dim, self.nvec, style=None))
        names = [agent.name for agent in agents]
        for index, agent in enumerate(agents):
            if isinstance(agent, PolicyNegotiator):
                agent.prepare(self.domain, ["RLAgent", names[1-index]])
        self.session.add(self.learner, ufun=self.utilities[0])
        self.session.add(agents[0], ufun=self.utilities[1])
        self.session.add(agents[1], ufun=self.utilities[2])
        self.observer = BidObserver(self.domain, self.learner.name, names)
        self.state = None
        self.opponent_names = names
        return padded_observation(self.observer.vector(), self.obs_dim)

    def current_mask(self):
        return action_mask(self.nvec, self.valid_nvec,
                           self.state is not None and self.state.get("current_offer") is not None).numpy()

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (len(self.nvec),):
            raise ValueError("Action head count mismatch")
        for index, valid in enumerate(self.valid_nvec):
            if action[index] < 0 or action[index] >= valid:
                raise ValueError("Invalid action for domain")
        if self.state is None and action[-1] == 0:
            raise ValueError("Cannot accept before the first offer")
        compact = np.concatenate((action[:len(self.domain)], action[-1:]))
        self.learner.set_action(compact, self.domain)
        for _ in range(3):
            self.state = self.session.step().__dict__
            self.observer.feed(self.state)
            if self.state.get("agreement") is not None or self.state.get("timedout") or self.state.get("broken"):
                break
        done = self.state.get("agreement") is not None or self.state.get("timedout") or self.state.get("broken")
        agreement = self.state.get("agreement")
        utilities = [float(utility(agreement)) for utility in self.utilities] if agreement is not None else [0.0] * 3
        reward = utilities[0] if agreement is not None else (-1.0 if done and not self.test else 0.0)
        info = {"my_util": utilities[0], "opp_util1": utilities[1], "opp_util2": utilities[2],
                "social": sum(utilities), "nash": float(np.prod(utilities)),
                "agreement": agreement, "step": self.state.get("step", 0),
                "opponent_names": self.opponent_names}
        return padded_observation(self.observer.vector(), self.obs_dim), reward, bool(done), info
