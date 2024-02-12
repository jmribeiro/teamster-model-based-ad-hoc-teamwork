import math
from abc import ABC
import numpy as np
import yaaf.policies


class UCTNode(ABC):

    def __init__(self, features, num_actions, dynamics_fn, parent=None, action=None, reward=None, feature_decimals=0):

        self.features = features

        self.num_actions = num_actions

        self.dynamics_fn = dynamics_fn

        self.Q = np.zeros((self.num_actions,))
        self.N = np.ones((self.num_actions,))

        self.parent = parent
        self.action = action
        self.reward = reward

        self.feature_decimals = feature_decimals

    def uct_search(self, max_iterations, exploration, rollout_depth, discount_factor):

        nodes = [self]

        for _ in range(max_iterations):

            # 1 - Select
            node, action = self.selection(nodes, exploration)

            # 2 - Expand
            next_node, reward = node.expansion(action)
            if next_node not in nodes: nodes.append(next_node)

            # 3 - Rollout
            next_value = next_node.rollout(rollout_depth, discount_factor)
            q_value = reward + discount_factor * next_value

            # 4 - Backup
            node.backup(action, q_value, discount_factor)

        return self.Q.argmax()

    def selection(self, nodes, exploration):
        selected_node = None
        selected_action = None
        max = -math.inf
        for node in nodes:
            node_action_ucb = node.upper_confidence_bound(exploration)
            if node_action_ucb.max() > max:
                argmaxes = np.where(node_action_ucb == node_action_ucb.max())[0]
                argmax = np.random.choice(argmaxes)
                max = node_action_ucb[argmax]
                selected_action = argmax
                selected_node = node
        return selected_node, selected_action

    def expansion(self, action):
        next_features, reward = self.dynamics_fn(self.features, action)
        next_node = UCTNode(next_features, self.num_actions, self.dynamics_fn, self, action, reward)
        return next_node, reward

    def rollout(self, max_depth, discount_factor):

        rewards = []
        node = self
        for depth in range(max_depth):
            action = yaaf.policies.random_action(self.num_actions)
            node, reward = node.expansion(action)
            rewards.append(reward)
            if reward == 1: break

        value = rewards[-1]
        for r, reward in enumerate(reversed(rewards)):
            if r != 0:
                value += (reward + discount_factor * value)

        return value

    def backup(self, action, q_value, discount_factor):

        self.N[action] += 1
        self.Q[action] += (q_value - self.Q[action]) / self.N[action]

        node = self.parent
        next_node = self
        while node is not None:
            reward = next_node.reward
            action = next_node.action
            next_value = next_node.Q.max()
            q_value = reward + discount_factor * next_value
            #node.N[action] += 1
            node.Q[action] += (q_value - node.Q[action]) / node.N[action]

            next_node = node
            node = node.parent

    def upper_confidence_bound(self, Cp):
        """Vectorized Implementation of the UCB Formula! (NumPy)"""
        return self.Q + Cp * np.sqrt(2 * np.log(self.N.sum()) / self.N)

    def __eq__(self, other):
        is_node = isinstance(other, UCTNode)
        same_features = np.array_equal(other.features, self.features)
        close_features = np.array_equal(np.round(other.features, decimals=self.feature_decimals), np.round(self.features, decimals=self.feature_decimals))
        same_parent_action = self.action == other.action
        same_reward = self.reward == other.reward
        return is_node and (same_features or close_features) and same_parent_action and same_reward
