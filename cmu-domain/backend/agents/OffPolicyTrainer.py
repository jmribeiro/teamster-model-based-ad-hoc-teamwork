from collections import deque

import numpy as np
import torch
from numpy import ndarray
from torch.nn.functional import one_hot
from yaaf import Timestep
from yaaf.agents import Agent
from yaaf.policies import random_policy

from backend.environments import LightEnvironment
from backend.environments.cmu import parse_positions
from backend.models import TeammatesModel, EnvironmentModel


class OffPolicyTrainer(Agent):

    def __init__(self, env: LightEnvironment, hyperparams):

        super().__init__("OffPolicyTrainer")

        # Auxiliary
        self.num_actions = env.action_space.n
        self.joint_action_space = env.joint_action_space
        state_features = env.state_space.shape[0]
        num_agents = len(env.joint_action_space[0])
        self.num_teammates = num_agents - 1

        # Models
        self.environment_model = EnvironmentModel(state_features, env.world_size, self.num_actions, self.joint_action_space, hyperparams)
        self.teammates_model = TeammatesModel(state_features, self.num_teammates, self.num_actions, hyperparams)

        # Hyperparams
        self.buffer_size = hyperparams["MBMCTS"]["buffer"]["max size"]

        # Datasets

        self.X_transition = deque(maxlen=(self.buffer_size*num_agents))
        self.Y_transition = deque(maxlen=(self.buffer_size*num_agents))

        self.terminal_states = deque(maxlen=self.buffer_size)
        self.non_terminal_states = deque(maxlen=self.buffer_size)

        self.X_teammates = deque(maxlen=self.buffer_size)
        self.Y_teammates = [deque(maxlen=self.buffer_size) for t in range(self.num_teammates)]

    def policy(self, state: ndarray):
        return random_policy(self.num_actions)

    def _reinforce(self, timestep: Timestep):

        # Transition Model Sample
        state = timestep.observation
        next_state = timestep.next_observation

        joint_action = timestep.info["joint action"]
        actions = self.joint_action_space[joint_action]

        agent_positions = parse_positions(state)
        next_agent_positions = parse_positions(next_state)
        for agent_id, agent_position in enumerate(agent_positions):
            action = actions[agent_id]
            action_one_hot = one_hot(torch.tensor(action), self.num_actions).numpy()
            x_transition = np.concatenate([np.array(agent_position), action_one_hot])
            y_transition = next_agent_positions[agent_id]
            self.X_transition.append(x_transition)
            self.Y_transition.append(y_transition)

        # Teammates Model Sample
        self.X_teammates.append(state)
        for t in range(self.num_teammates):
            self.Y_teammates[t].append(actions[t+1])

        # Rewards Model Sample
        reward = timestep.reward
        terminal = reward != -1.0
        if terminal: self.terminal_states.append(state)
        else: self.non_terminal_states.append(state)

    def train_models(self, epochs, verbose=False):

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        info = {}

        print(f"\tEnvironment Model", flush=True)
        transition_loss, reward_loss = self.environment_model.fit_and_validate(
            self.X_transition, self.Y_transition,
            self.terminal_states, self.non_terminal_states, epochs, verbose
        )

        info["transition model loss"] = transition_loss
        if reward_loss is not None: info["rewards model loss"] = reward_loss

        print(f"\tTeam Model", flush=True)
        teammate_model_losses = self.teammates_model.fit_and_validate(self.X_teammates, self.Y_teammates, epochs, verbose)
        for k, v in teammate_model_losses.items(): info[k] = v
        return info

    def train_environment_model(self, epochs, verbose=False):

        self.environment_model.fit_and_validate(
            self.X_transition, self.Y_transition,
            self.terminal_states, self.non_terminal_states, epochs, verbose
        )

    def train_teammates_model(self, epochs, verbose=False):
        self.teammates_model.fit_and_validate(self.X_teammates, self.Y_teammates, epochs, verbose)

    def save(self, directory):
        self.environment_model.save(directory)
        self.teammates_model.save(directory)

    def load(self, directory):
        self.environment_model.load(directory)
        self.teammates_model.load(directory)
