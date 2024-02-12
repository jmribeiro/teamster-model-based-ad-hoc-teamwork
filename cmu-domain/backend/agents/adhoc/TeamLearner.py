import torch
from numpy import ndarray
from yaaf.policies import greedy_policy

from backend.agents.MCTSAgent import MMDPMCTSNode
from backend.agents.adhoc import PLASTICAgent
from backend.models import TeammatesModel
from backend.search.UCTNode import UCTNode


class TeamLearner(PLASTICAgent):

    def __init__(self, envs, checkpoint_directories, hyperparams):

        super().__init__("TeamLearner", envs, hyperparams)

        # Models
        self.teammates_models = []

        # Lambdas
        self.teammates_fns = []
        self.teammates_policy_fn = []
        self.transition_fns = [env.transition_fn for env in envs]
        self.reward_fns = [env.reward_fn for env in envs]

        for checkpoint_directory in checkpoint_directories:

            # Auxiliary
            team = checkpoint_directory.split("/")[-1]
            t = self.teams.index(team)
            env = envs[t]
            state_features = env.state_space.shape[0]
            num_agents = len(env.joint_action_space[0])
            num_teammates = num_agents - 1

            # Models
            teammates_model = TeammatesModel(state_features, num_teammates, self.num_actions, hyperparams)
            teammates_model.load(checkpoint_directory)
            self.teammates_models.append(teammates_model)

            # Lambdas
            teammates_fn = lambda state: self.teammates_models[t].predict_actions(state)
            teammate_policy_fn = lambda state: self.teammates_models[t].predict_policies(state)
            self.teammates_fns.append(teammates_fn)
            self.teammates_policy_fn.append(teammate_policy_fn)

        # MCTS
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
        with torch.no_grad():
            k = self.most_likely_team()
            old = True
            if old:
                teammates_fn = self.teammates_fns[k]
                transition_fn = self.transition_fns[k]
                reward_fn = self.reward_fns[k]
                joint_action_space = self.joint_action_spaces[k]
                root = MMDPMCTSNode(state, transition_fn, reward_fn, teammates_fn, self.num_actions, joint_action_space)
                root.uct_search(self.iterations, self.max_rollout_depth, self.exploration, self.discount_factor)
            else:
                node = UCTNode(state, self.num_actions, lambda state, action: self.dynamics_fn(state, action, k))
                node.uct_search(self.iterations, self.exploration, self.max_rollout_depth, self.discount_factor)
            q_values = root.Q
            return greedy_policy(q_values)
