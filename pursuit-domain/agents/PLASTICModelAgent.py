import numpy as np
import yaml

from agents.plastic.model.LearningPLASTICModel import LearningPLASTICModel
from agents.plastic.model.LearntPLASTICModel import LearntPLASTICModel
from environment.Pursuit import Pursuit
from agents.plastic.PLASTICAgent import PLASTICAgent
from agents.plastic.model.HandcodedPLASTICModel import HandcodedPLASTICModel
from environment.PursuitState import PursuitState
from search.PursuitMCTSNode import PursuitMCTSNode
from search.UCTNode import UCTNode


class PLASTICModelAgent(PLASTICAgent):

    def __init__(self, num_teammates, world_size, preload=True):
        super().__init__("Plastic model", num_teammates, world_size, learn_team=True)
        with open("config.yaml", 'r') as stream: config = yaml.load(stream, Loader=yaml.FullLoader)["mcts"]

        self.world_size = world_size

        self.mcts_iterations = config["iterations"]
        self.mcts_Cp = config["Cp"]
        self.mcts_max_rollout_depth = config["maximum rollout depth"]
        self.mcts_discount_factor = config["discount factor"]

        if preload:
            self.load_handcoded_priors()

    def load_handcoded_prior(self, team_name):
        del self.learning_prior
        self.learning_prior = self.setup_learning_prior()
        self.priors[-1] = self.learning_prior
        self.priors = [
            HandcodedPLASTICModel(team_name, self.num_teammates, self.world_size),
        ] + self.priors
        self.belief_distribution = np.zeros((len(self.priors),)) + 1 / len(self.priors)
        assert np.isclose(np.sum(self.belief_distribution), 1.0), self.belief_distribution.sum()

    def load_handcoded_priors(self):
        self.priors = [
            HandcodedPLASTICModel("greedy", self.num_teammates, self.world_size),
            HandcodedPLASTICModel("teammate aware", self.num_teammates, self.world_size)
        ] + self.priors
        self.belief_distribution = np.zeros((len(self.priors),)) + 1 / len(self.priors)
        assert np.sum(self.belief_distribution) == 1.0

    def dynamics_fn(self, state, action, most_likely_model):
        pursuit_state = PursuitState.from_features(state, self.world_size)
        teammate_actions = most_likely_model.simulate_teammates_actions(pursuit_state)
        actions = [action] + teammate_actions
        next_pursuit_state, reward = Pursuit.transition(pursuit_state, actions)
        if reward == -1: reward = 0
        elif reward == 100: reward = 1
        else: raise ValueError("")
        return next_pursuit_state.features(), reward

    #################
    # PLASTIC Agent #
    #################

    def select_action_according_to_model(self, state, most_likely_model):
        old = True
        if old:
            node = PursuitMCTSNode(state, Pursuit.transition, most_likely_model.simulate_teammates_actions)
            action = node.uct_search(self.mcts_iterations, self.world_size[0], self.mcts_Cp, self.mcts_discount_factor)
        else:
            node = UCTNode(state.features(), num_actions=4, dynamics_fn=lambda state, action: self.dynamics_fn(state, action, most_likely_model))
            action = node.uct_search(self.mcts_iterations, self.mcts_Cp, self.mcts_max_rollout_depth, self.mcts_discount_factor)
        return action

    def setup_learning_prior(self):
        return LearningPLASTICModel(self.num_teammates)

    def _load_prior_team(self, directory, name):
        return LearntPLASTICModel(directory, name, self.num_teammates)
