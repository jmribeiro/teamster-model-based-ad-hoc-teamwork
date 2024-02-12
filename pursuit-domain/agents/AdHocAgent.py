import numpy as np
import yaml

from environment.PursuitState import PursuitState
from models.IndependentStochasticEnvironmentModel import IndependentStochasticEnvironmentModel
from models.ReplayMemoryModel import ReplayMemoryModel
from agents.plastic.PLASTICAgent import PLASTICAgent
from agents.plastic.model.LearningPLASTICModel import LearningPLASTICModel
from agents.plastic.model.LearntPLASTICModel import LearntPLASTICModel
from environment.utils import pursuit_datapoint

from yaaf.policies import random_action, linear_annealing

from search.PursuitMCTSNode import PursuitMCTSNode
from search.UCTNode import UCTNode


class AdHocAgent(PLASTICAgent):

    def __init__(self, num_teammates, world_size):

        super().__init__("Adhoc", num_teammates, world_size)

        with open("config.yaml", 'r') as stream: config = yaml.load(stream, Loader=yaml.FullLoader)

        self.environment_model = IndependentStochasticEnvironmentModel(num_agents=self.num_agents, trainable=True,
                                                                       learning_rate=config["learning rate"],
                                                                       layers=ReplayMemoryModel.parse_layer_blueprints(config["environment model"]["layers"]),
                                                                       replay_batch_size=config["replay min batch"],
                                                                       replay_memory_size=config["replay memory size"],
                                                                       learn_reward=config["environment model"]["learn reward"])

        self._start_exploration_rate = config["start exploration rate"]
        self._end_exploration_rate = config["end exploration rate"]
        self._final_timesteps = config["final exploration timestep"]

        self._total_team_timesteps = 0

        self.mcts_iterations = config["mcts"]["iterations"]
        self.mcts_Cp = config["mcts"]["Cp"]
        self.mcts_max_rollout_depth = config["mcts"]["maximum rollout depth"]
        self.mcts_discount_factor = config["mcts"]["discount factor"]

    #########
    # Agent #
    #########

    def _reinforce(self, timestep):
        info = super()._reinforce(timestep)
        self._total_team_timesteps += 1
        info["environment model"] = self.environment_model.replay_fit(pursuit_datapoint(timestep, self.world_size))
        info["exploration rate"] = self.exploration_rate
        return info

    def save(self, directory):
        super().save(directory)
        self.environment_model.save(directory)

    def load(self, directory):
        super().load(directory)
        self.environment_model.load(directory)

    #################
    # PLASTIC Agent #
    #################

    def select_action_according_to_model(self, state, most_likely_model):
        current_exploration_rate = linear_annealing(
            current_timestep=self.total_training_timesteps,
            final_timestep=self._final_timesteps,
            start_value=self._start_exploration_rate,
            final_value=self._end_exploration_rate
        )
        if self.trainable:
            roll = np.random.uniform(0, 1)
            if roll < current_exploration_rate:
                return random_action(4)

        old = True
        if old:
            root = PursuitMCTSNode(state, self.environment_model.simulate_transition, most_likely_model.simulate_teammates_actions)
            action = root.uct_search(self.mcts_iterations, self.world_size[0], self.mcts_Cp, self.mcts_discount_factor)
        else:
            node = UCTNode(state.features(), num_actions=4, dynamics_fn=lambda state, action: self.dynamics_fn(state, action, most_likely_model))
            action = node.uct_search(self.mcts_iterations, self.mcts_Cp, self.mcts_max_rollout_depth, self.mcts_discount_factor)

        return action

    def dynamics_fn(self, state, action, most_likely_model):
        pursuit_state = PursuitState.from_features(state, self.world_size)
        teammate_actions = most_likely_model.simulate_teammates_actions(pursuit_state)
        actions = [action] + teammate_actions
        next_pursuit_state, reward = self.environment_model.simulate_transition(pursuit_state, actions)
        if reward == -1: reward = 0
        elif reward == 100: reward = 1
        else: raise ValueError("")
        return next_pursuit_state.features(), reward

    @property
    def exploration_rate(self):
        if not self.trainable: return 0.0
        else: return linear_annealing(self._total_team_timesteps, self._final_timesteps, self._start_exploration_rate, self._end_exploration_rate)

    def setup_learning_prior(self):
        return LearningPLASTICModel(self.num_teammates)

    def _load_prior_team(self, directory, name):
        return LearntPLASTICModel(directory, name, self.num_teammates)
