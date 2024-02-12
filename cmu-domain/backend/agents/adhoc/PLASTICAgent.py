import random

import numpy as np
from abc import ABC
from yaaf.agents import Agent

from backend.configs import TEAMS


class PLASTICAgent(Agent, ABC):

    def __init__(self, name, envs, hyperparams):

        super().__init__(name)

        # For indexing auxiliary, doesn't mean agent has them in the library
        self.teams = TEAMS

        self.K = len(envs)

        # Overwrite if needed
        self.teammates_fns = [env.teammates_fn for env in envs]
        self.teammates_policy_fn = [env.teammate_policy_fn for env in envs]

        self.num_actions = envs[0].action_space.n
        for k in range(self.K):
            assert envs[k].action_space.n == self.num_actions, f"{name} requires same action space on all models"
        self.joint_action_spaces = [env.joint_action_space for env in envs]

        # Uniform distribution
        self.beliefs = np.ones(self.K) / self.K
        self.eta = hyperparams["PLASTIC"]["eta"]

        self.shared_environment_model = True    # For teamster and env learner

    def _model_probability(self, k, state, observed_actions):

        policies = self.teammates_policy_fn[k](state)
        num_teammates = len(policies)
        probabilities = np.zeros(num_teammates)
        for i, observed_action in enumerate(observed_actions):
            if i != 0:  # Ignore ad hoc agent
                teammate_policy = policies[i - 1]
                probabilities[i - 1] = teammate_policy[observed_action]

        model_probability = np.multiply.reduce(probabilities)

        return model_probability

    def reinforcement(self, timestep):
        return self._reinforce(timestep)

    def _reinforce(self, timestep):

        state = timestep.observation
        actions = timestep.info["actions"]

        # P(a|m, s) for each m in beliefs
        model_probs = np.array([self._model_probability(k, state, actions) for k in range(self.K)])

        new_beliefs = np.zeros_like(self.beliefs)

        # Compute beliefs given new probabilities
        for i in range(self.K):
            loss = 1 - model_probs[i]
            previous_belief = self.beliefs[i]
            new_belief = previous_belief * (1 - self.eta * loss)
            new_beliefs[i] = new_belief

        # Normalize
        self.beliefs = new_beliefs / new_beliefs.sum()

        return {}

    def most_likely_team(self):
        argmaxes = np.argwhere(self.beliefs == np.max(self.beliefs)).reshape(-1)
        t = random.choice(argmaxes)
        #print("Most likely: ", self.teams[t])
        return t
