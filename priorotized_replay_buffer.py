import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor
import copy
import torch
import random
from buffers_modified import BaseBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from gymnasium import spaces
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

class TrajectorySample():
    def __init__(self):
        self.samples=[]
        self.new_obs=None
        self.dones=None
        self.distributions=None
    def add(self, sample):
        self.samples = self.samples + [sample]


class Sample:
    def __init__(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],):
        
        self.information = [obs,action,next_obs,done,reward,infos]
            

class CustomReplayBuffer:
    def __init__(self, device, buffer_size=10):
        self.rollout_buffer = None
        self.buffer_size = buffer_size
        self.buffer = []
        self.device = device
    
    def get(self, batch_size):
        return [self.buffer[x] for x in np.random.randint(low=0,high=len(self.buffer),size=(batch_size))]
    
    def insert(self, sample):
        # each sample is like a run of simulation plus final values and more
        self.buffer = self.buffer + [sample]
        if len(self.buffer)>self.buffer_size: self.buffer=self.buffer[1:]

    def process(self, trajectory_sample, policy):
        self.rollout_buffer.reset()
        for sample in trajectory_sample.samples:
            self.rollout_buffer.add(sample._last_obs, sample.actions, sample.rewards, sample._last_episode_starts, sample.values, sample.log_probs)
        values = policy.predict_values(obs_as_tensor(trajectory_sample.new_obs, torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        self.rollout_buffer.compute_returns_and_advantage(last_values=values.detach(), dones=trajectory_sample.dones)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            return copy.deepcopy(rollout_data), trajectory_sample.distributions


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
    
class PriorityReplayBuffer(BaseBuffer):  # stored as ( s, a, r, s_ ) in SumTree

    # method is compatible
    def __init__(
            self, 
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
            e=1e-2, 
            a=6e-1, 
            beta=4e-1, 
            beta_increment_per_sampling=1e-3, 
            priority_type='PS_Seq', 
            ann_beta=False,):

        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.tree = SumTree(self.buffer_size)
        self.epsilon = e
        self.alpha = a
        self.beta = beta
        self._beta = beta
        self.b_inc = beta_increment_per_sampling
        self.priority_type = priority_type
        self.ann_beta = ann_beta
        self.pos = 0
        self.gamma = 0.99

    def empty(self):
        self.tree = SumTree(self.capacity)
        if self.ann_beta : self._beta = 0.99 * self._beta
        self.beta = self._beta

    def scale_gamme(self): self.gamma *= self.gamma
    def reset_gamma(self): self.gamma  = 0.99
    def set_gamma(self, value): self.gamma = value

    def _get_priority(self, error):
        if self.priority_type == 'TD_Err': return (np.abs(error) + self.epsilon) ** self.alpha
        elif self.priority_type == 'KL_Div': return 1/(error + self.epsilon)
        elif self.priority_type == 'PS_Seq': return 1/(error + self.epsilon)

    # method is compatible
    def add(
            self, 
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
            error: float=None):
        
        if error is not None:
            p = self._get_priority(error)
        else: p = self._get_priority(150.0*self.gamma)

        self.tree.add(p, Sample(obs=obs,next_obs=next_obs,action=action,reward=reward,done=done,infos=infos))

        self.pos = min(self.tree.write, self.tree.capacity)

    # method is compatible
    def _get_samples(self, batch_inds: np.ndarray, env = None, reward_training = False):
        batch = []
        idxs = []
        batch_size = len(batch_inds)
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.b_inc])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        if reward_training : 
            expert_idx = np.argwhere(1/np.array(priorities) <= 142.0)
            agent_idx  = np.array([x for x in range(batch_size) if x not in expert_idx])
            states = self._normalize_obs(np.array([e.information[0].reshape(-1) for e in batch], dtype=np.float32), env)
            expert_states = states[expert_idx]
            agent_states  = states[agent_idx]
            actions = np.array([e.information[1].reshape(-1) for e in batch], dtype=np.float32)
            expert_actions = actions[expert_idx]
            agent_actions  = actions[agent_idx]
            return [expert_states, expert_actions], [agent_states, agent_actions]

        # standardize output
        data = (
            self._normalize_obs(np.array([e.information[0].reshape(-1) for e in batch], dtype=np.float32), env),
            np.array([e.information[1].reshape(-1) for e in batch], dtype=np.float32),
            np.array([e.information[2].reshape(-1) for e in batch], dtype=np.float32),
            np.array([e.information[3] for e in batch], dtype=np.float32).reshape(-1,1),
            self._normalize_reward(np.array([e.information[4].reshape(-1) for e in batch], dtype=np.float32).reshape(-1, 1), env),
        )

        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def sample(self, batch_size: int, env = None, reward_training = False):
        return self._get_samples(np.arange(batch_size), env, reward_training)