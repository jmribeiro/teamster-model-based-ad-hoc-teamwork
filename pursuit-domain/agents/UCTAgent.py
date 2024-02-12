import yaml

from environment.PursuitState import PursuitState
from models.PerfectEnvironmentModel import PerfectEnvironmentModel
from agents.plastic.model.HandcodedPLASTICModel import HandcodedPLASTICModel

from yaaf.agents import Agent
from yaaf.policies import greedy_policy

from search.PursuitMCTSNode import PursuitMCTSNode
from search.UCTNode import UCTNode


class UCTAgent(Agent):

    def __init__(self, num_teammates, world_size, team_name="greedy", config=None):

        super().__init__("UCT")

        if config is None:
            with open("config.yaml", 'r') as stream:
                config = yaml.load(stream, Loader=yaml.FullLoader)

        self.world_size = world_size
        self.mcts_iterations = config["mcts"]["iterations"]
        self.mcts_Cp = config["mcts"]["Cp"]
        self.mcts_max_rollout_depth = config["mcts"]["maximum rollout depth"]
        self.mcts_discount_factor = config["mcts"]["discount factor"]
        self.teammates_model = HandcodedPLASTICModel(team_name, num_teammates, world_size)
        self.environment_model = PerfectEnvironmentModel()

    def dynamics_fn(self, state, action):
        pursuit_state = PursuitState.from_features(state, self.world_size)
        teammate_actions = self.teammates_model.simulate_teammates_actions(pursuit_state)
        actions = [action] + teammate_actions
        next_pursuit_state, reward = self.environment_model.simulate_transition(pursuit_state, actions)
        if reward == -1: reward = 0
        elif reward == 100: reward = 1
        else: raise ValueError("")
        return next_pursuit_state.features(), reward

    def policy(self, observation):
        Q = self.mcts_q_values(observation)
        return greedy_policy(Q)

    def mcts_q_values(self, observation):
        old = True
        if old:
            state = PursuitState.from_features(observation, self.world_size)
            node = PursuitMCTSNode(state, self.environment_model.simulate_transition, self.teammates_model.simulate_teammates_actions)
            node.uct_search(self.mcts_iterations, self.mcts_max_rollout_depth, self.mcts_Cp, self.mcts_discount_factor)
        else:
            node = UCTNode(observation, num_actions=4, dynamics_fn=lambda state, action: self.dynamics_fn(state, action))
            node.uct_search(self.mcts_iterations, self.mcts_Cp, self.mcts_max_rollout_depth, self.mcts_discount_factor)
        return node.Q

    def _reinforce(self, timestep):
        return {}
