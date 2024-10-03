from __future__ import annotations
from typing import Any

import gymnasium as gym
from gymnasium.vector import VectorEnv
import numpy as np
import os
import torch
from datetime import datetime

"""
Vectorized environment wrapper.
"""

class GymIsaacWrapperForROSPlanner(VectorEnv):

    def __init__(self, num_envs, single_observation_space, single_action_space):
        # Collect common information
        self.num_envs = num_envs
        self.single_observation_space = single_observation_space
        self.single_action_space = single_action_space

        self.observation = None
        self.rewards = np.zeros(self.num_envs)
        self.terminated = np.zeros(self.num_envs, dtype=bool)
        self.truncated = np.zeros(self.num_envs, dtype=bool)

        if isinstance(action_space, gym.spaces.Box) and not action_space.is_bounded("both"):
            action_space = gym.spaces.Box(low=-100, high=100, shape=action_space.shape)

        # Initialize vec-env
        VectorEnv.__init__(self, self.num_envs, self.single_observation_space, self.single_action_space)

    #def seed(self, seed: int | None = None) -> list[int | None]:  # noqa: D102
    #    return [self.unwrapped.seed(seed)] * self.unwrapped.num_envs

    def set_dummy_observation(self):
        self.observation = np.zeros((self.num_envs, self.single_observation_space.shape[0]))

    def update_observations(self, observation):
        self.observation = observation

    def reset(self, seed: int | None = None, options: dict = None):  # noqa: D102
        observations = self.observation
        info = dict()
        return observations, info
    
    def step_async(self, actions):  # noqa: D102
        # Convert input to numpy array
        pass

    def step_wait(self):  # noqa: D102
        observations = self.observation
        rewards = self.rewards
        terminated = self.terminated
        truncated = self.truncated
        info = dict()
        return observations, rewards, terminated, truncated, info

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__
