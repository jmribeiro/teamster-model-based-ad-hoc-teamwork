from numpy import ndarray
from yaaf import Timestep
from yaaf.agents import Agent
from yaaf.policies import greedy_policy
from yaaf.search.MCTSNode import MCTSNode
import numpy as np

from backend.environments import LightEnvironment
from backend.search.UCTNode import UCTNode


class MCTSAgent(Agent):

    def __init__(self, light_env: LightEnvironment, hyperparams):

        super().__init__("MCTS")

        self.transition_fn = light_env.transition_fn
        self.reward_fn = light_env.reward_fn
        self.teammates_fn = light_env.teammates_fn
        self.num_actions = light_env.action_space.n
        self.joint_action_space = light_env.joint_action_space

        self.iterations = hyperparams["MCTS"]["iterations"]
        self.max_rollout_depth = hyperparams["MCTS"]["maximum rollout depth"]
        self.exploration = hyperparams["MCTS"]["Cp"]
        self.discount_factor = hyperparams["MCTS"]["discount factor"]

    def policy(self, state: ndarray):

        old = True

        if old:
            node = MMDPMCTSNode(state, self.transition_fn, self.reward_fn, self.teammates_fn, self.num_actions, self.joint_action_space)
            node.uct_search(self.iterations, self.max_rollout_depth, self.exploration, self.discount_factor)
        else:
            node = UCTNode(state, self.num_actions, self.dynamics_fn)
            node.uct_search(self.iterations, self.exploration, self.max_rollout_depth, self.discount_factor)

        q_values = node.Q
        return greedy_policy(q_values)

    def dynamics_fn(self, state, action):
        teammates_action = self.teammates_fn(state)
        actions = tuple([action] + teammates_action)
        joint_action = self.joint_action_space.index(actions)
        next_state = self.transition_fn(state, joint_action)
        reward = self.reward_fn(state, joint_action)
        if reward == -1: reward = 0
        else: reward = 1
        return next_state, reward

    def _reinforce(self, timestep: Timestep):
        return {}


class MMDPMCTSNode(MCTSNode):

    def __init__(self, state, transition_fn, reward_fn, teammates_fn,
                 num_actions, joint_action_space, step_reward=-1, win_reward=100):
        super().__init__(num_actions=num_actions)
        self.state = state
        self.transition_fn = transition_fn
        self.reward_fn = reward_fn
        self.teammates_fn = teammates_fn
        self.joint_action_space = joint_action_space
        self.step_reward, self.win_reward = step_reward, win_reward

    def __eq__(self, other):
        return isinstance(other, MMDPMCTSNode) and np.array_equal(self.state, other.state)

    def simulate_action(self, action):
        teammates_action = self.teammates_fn(self.state)
        actions = tuple([action] + teammates_action)
        joint_action = self.joint_action_space.index(actions)
        next_state = self.transition_fn(self.state, joint_action)
        reward = self.reward_fn(self.state, joint_action)
        terminal = reward == self.win_reward
        normalized_reward = self._normalize_reward(reward)
        next_state_node = MMDPMCTSNode(
            next_state, self.transition_fn, self.reward_fn, self.teammates_fn,
            self.num_actions, self.joint_action_space, self.step_reward, self.win_reward)
        return next_state_node, normalized_reward, terminal

    def _normalize_reward(self, reward):
        if reward == self.step_reward: reward = 0
        else: reward = 1
        return reward
