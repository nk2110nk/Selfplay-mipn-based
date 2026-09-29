"""Alpha-Nego-style distributional SAC adapted to MultiDiscrete bids."""

import copy
import itertools

import numpy as np
import torch

from policy import (Actor, QuantileCritic, critic_action, onehot_action, quantile_huber_loss,
                    select_lower_distribution, twin_style_value)
from replay_buffer import ReplayBuffer


class DSAC:
    def __init__(self, obs_dim, nvec, *, device="cpu", quantiles=64, actor_lr=3e-5,
                 critic_lr=1e-4, batch_size=128, buffer_size=1_000_000, tau=0.005,
                 gamma=0.99, entropy_coefficient=0.01, hidden=256, dropout=0.1,
                 auto_entropy=False, target_entropy_ratio=0.98, gradient_clip=1.0,
                 target_entropy_final_ratio=None, entropy_anneal_steps=0,
                 canonical_accept_action=True, hierarchical_entropy=True,
                 policy_update_frequency=2, gumbel_temperature=1.0,
                 training_style="neutral", aggressive_quantile=0.8,
                 conservative_quantile=0.2, risk_weight=0.1):
        self.device = self._resolve_device(device)
        self.nvec = tuple(map(int, nvec))
        self.obs_dim = int(obs_dim)
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.auto_entropy = bool(auto_entropy)
        self.target_entropy_ratio = float(target_entropy_ratio)
        self.target_entropy_final_ratio = float(target_entropy_final_ratio if target_entropy_final_ratio is not None
                                                else target_entropy_ratio)
        self.entropy_anneal_steps = int(entropy_anneal_steps)
        self.canonical_accept_action = bool(canonical_accept_action)
        self.hierarchical_entropy = bool(hierarchical_entropy)
        self.gradient_clip = float(gradient_clip)
        self.policy_update_frequency = int(policy_update_frequency)
        self.gumbel_temperature = float(gumbel_temperature)
        self.training_style = training_style
        self.aggressive_quantile = float(aggressive_quantile)
        self.conservative_quantile = float(conservative_quantile)
        self.risk_weight = float(risk_weight)
        self.quantiles = quantiles
        self.hidden = hidden
        self.dropout = float(dropout)
        self.actor = Actor(obs_dim, nvec, hidden, dropout).to(self.device)
        self.critics = [QuantileCritic(obs_dim, nvec, quantiles, hidden, dropout).to(self.device) for _ in range(2)]
        self.targets = [copy.deepcopy(c).to(self.device) for c in self.critics]
        for target in self.targets:
            target.eval()
            target.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizers = [torch.optim.Adam(c.parameters(), lr=critic_lr) for c in self.critics]
        initial_log_alpha = np.log(max(float(entropy_coefficient), 1e-8))
        self.log_alpha = torch.nn.Parameter(
            torch.full((len(self.nvec),), initial_log_alpha, device=self.device),
            requires_grad=self.auto_entropy,
        )
        self.alpha_optimizer = (torch.optim.Adam([self.log_alpha], lr=actor_lr)
                                if self.auto_entropy else None)
        self.replay = ReplayBuffer(buffer_size)
        self.update_steps = 0

    @property
    def alpha(self):
        values = self.log_alpha.detach().exp()
        return float(values.mean().item())

    @property
    def current_target_entropy_ratio(self):
        if self.entropy_anneal_steps <= 0:
            return self.target_entropy_final_ratio
        progress = min(self.update_steps / self.entropy_anneal_steps, 1.0)
        return self.target_entropy_ratio + progress * (
            self.target_entropy_final_ratio - self.target_entropy_ratio)

    def _critic_action(self, action):
        return critic_action(action, self.nvec) if self.canonical_accept_action else action

    @staticmethod
    def _resolve_device(requested):
        if requested not in ("auto", "cpu", "cuda"):
            raise ValueError(f"Unsupported device: {requested}")
        if requested == "cpu":
            return torch.device("cpu")
        cuda_usable = torch.cuda.is_available()
        if cuda_usable:
            major, minor = torch.cuda.get_device_capability()
            cuda_usable = f"sm_{major}{minor}" in torch.cuda.get_arch_list()
        if requested == "cuda" and not cuda_usable:
            raise RuntimeError("CUDA is present, but this PyTorch build cannot execute on the installed GPU")
        return torch.device("cuda" if cuda_usable else "cpu")

    @torch.no_grad()
    def act(self, obs, mask, *, deterministic=False, style=None, candidates=16,
            aggressive_quantile=0.8, conservative_quantile=0.2, risk_weight=0.1):
        observation = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        valid = torch.as_tensor(np.asarray(mask, dtype=np.bool_), device=self.device).unsqueeze(0)
        actor_training = self.actor.training
        critic_training = [critic.training for critic in self.critics]
        self.actor.eval()
        for critic in self.critics:
            critic.eval()
        if style is None:
            action, _, _ = self.actor.sample(observation, valid, deterministic=deterministic)
            self.actor.train(actor_training)
            for critic, mode in zip(self.critics, critic_training):
                critic.train(mode)
            return action.squeeze(0).cpu().numpy()
        # Enumerate compact domains exactly; sample candidates only for large joint spaces.
        logits = self.actor.logits(observation, valid)
        seen = set()
        options = []
        valid_indices = [torch.nonzero(head[:size], as_tuple=False).squeeze(-1).tolist()
                         for head, size in zip(valid.squeeze(0), self.nvec)]
        joint_size = int(np.prod([len(indices) for indices in valid_indices]))
        if joint_size <= max(1, candidates):
            for key in itertools.product(*valid_indices):
                tensor = torch.as_tensor(key, device=self.device).unsqueeze(0)
                options.append((key, self._critic_action(onehot_action(tensor, self.nvec))))
                seen.add(key)
        else:
            for _ in range(max(1, candidates)):
                action, onehot, _ = self.actor.sample(observation, valid)
                key = tuple(action.squeeze(0).tolist())
                if key not in seen:
                    seen.add(key)
                    options.append((key, self._critic_action(onehot)))
        greedy = tuple(int(x.argmax(-1).item()) for x in logits)
        if greedy not in seen:
            onehot = self._critic_action(
                onehot_action(torch.as_tensor(greedy, device=self.device).unsqueeze(0), self.nvec))
            options.append((greedy, onehot))
        best = None
        for key, onehot in options:
            q1, q2 = (critic(observation, onehot) for critic in self.critics)
            value = twin_style_value(q1, q2, style, aggressive_quantile,
                                     conservative_quantile, risk_weight).item()
            if best is None or value > best[0]:
                best = (value, key)
        self.actor.train(actor_training)
        for critic, mode in zip(self.critics, critic_training):
            critic.train(mode)
        return np.asarray(best[1], dtype=np.int64)

    def update(self):
        if len(self.replay) < self.batch_size:
            return None
        self.update_steps += 1
        obs, actions, rewards, next_obs, dones, masks, next_masks = self.replay.sample(self.batch_size, self.device)
        with torch.no_grad():
            _, next_onehot, _ = self.actor.sample(next_obs, next_masks)
            next_onehot = self._critic_action(next_onehot)
            q_next = select_lower_distribution(self.targets[0](next_obs, next_onehot), self.targets[1](next_obs, next_onehot))
            # Alpha-Nego Algorithm 2 uses an entropy-free distributional Bellman target.
            target = rewards[:, None] + self.gamma * (1 - dones[:, None]) * q_next
        onehot = self._critic_action(onehot_action(actions, self.nvec))
        critic_losses = []
        for critic, optimizer in zip(self.critics, self.critic_optimizers):
            loss = quantile_huber_loss(critic(obs, onehot), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), self.gradient_clip)
            optimizer.step()
            critic_losses.append(loss.item())
        actor_loss = torch.zeros((), device=self.device)
        entropy = self.actor.entropy(obs, masks, hierarchical=self.hierarchical_entropy)
        alpha_loss = torch.zeros((), device=self.device)
        if self.update_steps % self.policy_update_frequency == 0:
            _, policy_action, _ = self.actor.sample(
                obs, masks, relaxed=True, temperature=self.gumbel_temperature)
            policy_action = self._critic_action(policy_action)
            for critic in self.critics:
                critic.requires_grad_(False)
            q1, q2 = (critic(obs, policy_action) for critic in self.critics)
            q = twin_style_value(q1, q2, self.training_style, self.aggressive_quantile,
                                 self.conservative_quantile, self.risk_weight)
            alpha = self.log_alpha.detach().exp().view(1, -1)
            actor_loss = (-q - (alpha * entropy).sum(-1)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
            self.actor_optimizer.step()
            for critic in self.critics:
                critic.requires_grad_(True)

            if self.auto_entropy:
                valid_counts = masks.sum(-1).clamp_min(1).to(entropy.dtype)
                target_entropy = self.current_target_entropy_ratio * valid_counts.log()
                distributions = self.actor.distributions(obs, masks)
                if self.hierarchical_entropy and len(distributions) > 1:
                    reject_probability = distributions[-1].probs[..., 1:2].detach()
                    target_entropy = torch.cat((target_entropy[..., :-1] * reject_probability,
                                                target_entropy[..., -1:]), dim=-1)
                alpha_loss = (self.log_alpha.exp().view(1, -1) *
                              (entropy.detach() - target_entropy)).sum(-1).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
        with torch.no_grad():
            for critic, target_critic in zip(self.critics, self.targets):
                for source, target_param in zip(critic.parameters(), target_critic.parameters()):
                    target_param.lerp_(source, self.tau)
        return {"actor_loss": actor_loss.item(), "critic_loss": float(np.mean(critic_losses)),
                "entropy": float(entropy.detach().sum(-1).mean().item()), "alpha": self.alpha,
                "alpha_loss": alpha_loss.item(),
                "target_entropy_ratio": self.current_target_entropy_ratio}

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "critics": [c.state_dict() for c in self.critics],
                "target_critics": [c.state_dict() for c in self.targets],
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizers": [o.state_dict() for o in self.critic_optimizers],
                "entropy_parameters": {"log_alpha": self.log_alpha.detach().cpu(),
                                       "auto": self.auto_entropy,
                                       "optimizer": self.alpha_optimizer.state_dict() if self.alpha_optimizer else None},
                "replay_buffer": self.replay.state_dict(), "obs_dim": self.obs_dim,
                "action_nvec": self.nvec, "quantiles": self.quantiles, "hidden": self.hidden,
                "dropout": self.dropout, "batch_size": self.batch_size, "tau": self.tau,
                "gamma": self.gamma, "target_entropy_ratio": self.target_entropy_ratio,
                "target_entropy_final_ratio": self.target_entropy_final_ratio,
                "entropy_anneal_steps": self.entropy_anneal_steps,
                "canonical_accept_action": self.canonical_accept_action,
                "hierarchical_entropy": self.hierarchical_entropy,
                "gradient_clip": self.gradient_clip,
                "policy_update_frequency": self.policy_update_frequency,
                "gumbel_temperature": self.gumbel_temperature,
                "training_style": self.training_style,
                "aggressive_quantile": self.aggressive_quantile,
                "conservative_quantile": self.conservative_quantile,
                "risk_weight": self.risk_weight, "update_steps": self.update_steps}

    def load_state_dict(self, state, *, with_replay=True):
        if state["obs_dim"] != self.obs_dim or tuple(state["action_nvec"]) != self.nvec:
            raise ValueError("Checkpoint observation/action space mismatch")
        self.actor.load_state_dict(state["actor"])
        for model, weights in zip(self.critics, state["critics"]):
            model.load_state_dict(weights)
        for model, weights in zip(self.targets, state["target_critics"]):
            model.load_state_dict(weights)
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        for optimizer, weights in zip(self.critic_optimizers, state["critic_optimizers"]):
            optimizer.load_state_dict(weights)
        entropy_state = state["entropy_parameters"]
        with torch.no_grad():
            if "log_alpha" in entropy_state:
                self.log_alpha.copy_(entropy_state["log_alpha"].to(self.device))
            else:
                self.log_alpha.fill_(np.log(entropy_state["coefficient"]))
        if self.alpha_optimizer and entropy_state.get("optimizer"):
            self.alpha_optimizer.load_state_dict(entropy_state["optimizer"])
        self.update_steps = state.get("update_steps", 0)
        if with_replay:
            self.replay.load_state_dict(state["replay_buffer"])
