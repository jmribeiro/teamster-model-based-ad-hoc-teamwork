from numpy import ndarray
from yaaf.policies import greedy_policy

from backend.agents.MCTSAgent import MMDPMCTSNode
from backend.agents.adhoc import PLASTICAgent
from backend.search.UCTNode import UCTNode


class PLASTICModelAgent(PLASTICAgent):

    def __init__(self, envs, hyperparams):

        super().__init__("PLASTIC-Model", envs, hyperparams)

        self.transition_fns = [env.transition_fn for env in envs]
        self.reward_fns = [env.reward_fn for env in envs]

        self.iterations = hyperparams["MCTS"]["iterations"]
        self.max_rollout_depth = hyperparams["MCTS"]["maximum rollout depth"]
        self.exploration = hyperparams["MCTS"]["Cp"]
        self.discount_factor = hyperparams["MCTS"]["discount factor"]

    def dynamics_fn(self, state, action, t):
        teammates_action = self.teammates_fns[t](state)
        actions = tuple([action] + teammates_action)
        joint_action = self.joint_action_spaces[t].index(actions)
        next_state = self.transition_fns[t](state, joint_action)
        reward = self.reward_fns[t](state, joint_action)
        if reward == -1: reward = 0
        else: reward = 1
        return next_state, reward

    def policy(self, state: ndarray):

        k = self.most_likely_team()

        old = True

        if old:
            transition_fn = self.transition_fns[k]
            reward_fn = self.reward_fns[k]
            teammates_fn = self.teammates_fns[k]
            joint_action_space = self.joint_action_spaces[k]
            node = MMDPMCTSNode(state, transition_fn, reward_fn, teammates_fn, self.num_actions, joint_action_space)
            node.uct_search(self.iterations, self.max_rollout_depth, self.exploration, self.discount_factor)
        else:
            node = UCTNode(state, self.num_actions, lambda state, action: self.dynamics_fn(state, action, k))
            node.uct_search(self.iterations, self.exploration, self.max_rollout_depth, self.discount_factor)

        q_values = node.Q
        return greedy_policy(q_values)
