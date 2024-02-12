from itertools import product

import numpy as np
from gym import Env
from gym.envs.registration import EnvSpec
from gym.spaces import Box, Discrete
from yaaf.policies import sample_action



class LightEnvironment(Env):

    def __init__(self, id, reset_fn, transition_fn, reward_fn, teammate_policy_fn,
                 low, high, num_actions, min_reward, max_reward):

        # OpenAI Gym Metadata
        self.spec = EnvSpec(id=id)
        self.observation_space = Box(low, high, reset_fn().shape, np.float)
        self.state_space = Box(low, high, reset_fn().shape, np.float)
        self.action_space = Discrete(num_actions)
        self.reward_range = (min_reward, max_reward)
        self.metadata = {}

        self.reset_fn = reset_fn

        self.transition_fn = transition_fn
        self.reward_fn = reward_fn

        self.teammates_fn = lambda state: [sample_action(policy) for policy in teammate_policy_fn(state)]
        self.teammate_policy_fn = teammate_policy_fn
        self.num_teammates = len(self.teammates_fn(reset_fn()))
        self.joint_action_space = list(product(range(num_actions), repeat=self.num_teammates+1))

        self.reset()

    def reset(self):
        self.state = self.reset_fn()
        return self.state

    def step(self, action: int):

        actions = tuple([action] + self.teammates_fn(self.state))
        joint_action = self.joint_action_space.index(actions)

        next_state = self.transition_fn(self.state, joint_action)
        reward = self.reward_fn(self.state, joint_action)

        self.state = next_state

        terminal = reward == 100.00
        info = {"actions": actions, "joint action": joint_action}

        return next_state, reward, terminal, info

    def render(self, mode="human"):
        from backend.environments.cmu import print_state
        print_state(self.state, self.layout)
        print()
