"""Replay buffer with checkpointable contents and per-transition action masks."""

import random

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.data = []
        self.position = 0

    def __len__(self):
        return len(self.data)

    def add(self, obs, action, reward, next_obs, done, mask, next_mask):
        row = (np.asarray(obs, dtype=np.float32).copy(), np.asarray(action, dtype=np.int64).copy(),
               float(reward), np.asarray(next_obs, dtype=np.float32).copy(), float(done),
               np.asarray(mask, dtype=np.bool_).copy(), np.asarray(next_mask, dtype=np.bool_).copy())
        if len(self.data) < self.capacity:
            self.data.append(row)
        else:
            self.data[self.position] = row
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device):
        rows = random.sample(self.data, batch_size)
        cols = zip(*rows)
        obs, action, reward, next_obs, done, mask, next_mask = [np.stack(col) for col in cols]
        return (torch.as_tensor(obs, device=device), torch.as_tensor(action, device=device),
                torch.as_tensor(reward, device=device), torch.as_tensor(next_obs, device=device),
                torch.as_tensor(done, device=device), torch.as_tensor(mask, device=device),
                torch.as_tensor(next_mask, device=device))

    def state_dict(self):
        return {"capacity": self.capacity, "data": self.data, "position": self.position}

    def load_state_dict(self, state):
        self.capacity = state["capacity"]
        self.data = state["data"]
        self.position = state["position"]
