import torch
from numpy import ndarray

from backend.agents.adhoc import PLASTICAgent
from backend.models import TeammatesModel


class PLASTICPolicyAgent(PLASTICAgent):

    def __init__(self, envs, checkpoint_directories, hyperparams):

        super().__init__("PLASTIC-Policy", envs, hyperparams)

        # Auxiliary
        from backend.agents import load_pretrained_dqn_agent

        # Models & Lambdas
        self.dqns = []
        self.teammates_models = []
        self.teammates_fns = []
        self.teammates_policy_fn = []

        for checkpoint_directory in checkpoint_directories:

            team = checkpoint_directory.split("/")[-1]
            t = self.teams.index(team)
            env = envs[t]

            # Auxiliary
            state_features = env.state_space.shape[0]
            num_agents = len(env.joint_action_space[0])
            num_teammates = num_agents - 1

            # Policy
            dqn = load_pretrained_dqn_agent(env, checkpoint_directory, hyperparams)
            self.dqns.append(dqn)

            # Models
            teammates_model = TeammatesModel(state_features, num_teammates, self.num_actions, hyperparams)
            teammates_model.load(checkpoint_directory)
            self.teammates_models.append(teammates_model)

            # Lambdas
            teammate_fn = lambda state: self.teammates_models[t].predict_actions(state)
            teammate_policy_fn = lambda state: self.teammates_models[t].predict_policies(state)
            self.teammates_fns.append(teammate_fn)
            self.teammates_policy_fn.append(teammate_policy_fn)

    def policy(self, state: ndarray):
        with torch.no_grad():
            t = self.most_likely_team()
            dqn = self.dqns[t]
            return dqn.policy(state)
