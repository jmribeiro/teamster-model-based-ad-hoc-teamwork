import numpy as np
import torch
import random

from torch.nn.functional import one_hot
from yaaf.models.FeedForwardNetwork import FeedForwardNetwork

from backend.environments.cmu import parse_positions


class EnvironmentModel:

    def __init__(self, state_features, world_size, num_actions, joint_action_space, hyperparams, step_reward=-1.0, win_reward=100):

        self.step_reward = step_reward
        self.win_reward = win_reward
        self.num_rows, self.num_columns = world_size

        self.batch_size = hyperparams["MBMCTS"]["buffer"]["batch size"]
        self.joint_action_space = joint_action_space
        self.num_actions = num_actions
        grid_dimensions = 2 # x, y

        self.transition_model = FeedForwardNetwork(
            num_inputs=grid_dimensions + self.num_actions,
            num_outputs=grid_dimensions,
            layers=[
                (hyperparams["MBMCTS"]["transition model"]["hidden sizes"], "relu")
                for _ in range(hyperparams["MBMCTS"]["transition model"]["num layers"])
            ],
            learning_rate=hyperparams["MBMCTS"]["transition model"]["learning rate"],
            loss="mae",
            cuda=torch.cuda.is_available()
        )

        def transition_accuracy(X, y):
            y_pred = self.transition_model.predict(X).round()
            total = 0
            correct = 0
            for i in range(y_pred.shape[0]):
                correct += (y_pred[i] == y[i]).sum()
                total += y[i].shape[0]
            accuracy = correct / total
            return accuracy.item()

        self.transition_model.accuracy = transition_accuracy

        self.rewards_model = FeedForwardNetwork(
            num_inputs=state_features,
            num_outputs=2,
            layers=((64, "relu"),),
            learning_rate=0.01, cuda=torch.cuda.is_available()
        )

    def predict_transition(self, state, joint_action):

        actions = self.joint_action_space[joint_action]
        agent_positions = parse_positions(state)
        next_agent_positions = []

        for agent_id, agent_position in enumerate(agent_positions):

            action = actions[agent_id]
            action_one_hot = one_hot(torch.tensor(action), self.num_actions).numpy()
            features = np.concatenate([np.array(agent_position), action_one_hot])
            next_agent_position = self.transition_model.predict(features).round()
            next_agent_position = list(next_agent_position.int().numpy())

            # x = column
            if next_agent_position[0] < 0:
                next_agent_position[0] = 0
            elif next_agent_position[0] > (self.num_columns - 1):
                next_agent_position[0] = self.num_columns - 1

            # y = row
            if next_agent_position[1] < 0:
                next_agent_position[1] = 0
            elif next_agent_position[1] > (self.num_rows - 1):
                next_agent_position[1] = self.num_rows - 1

            next_agent_position = tuple(next_agent_position)

            collision = False
            for other_agent_id in range(len(agent_positions)):

                if other_agent_id != agent_id:

                    if agent_id < other_agent_id:
                        # Other hasn't moved yet
                        other_agent_position = agent_positions[other_agent_id]
                    else:
                        # Other has moved
                        other_agent_position = next_agent_positions[other_agent_id]

                    if next_agent_position == other_agent_position:
                        collision = True
                        break

            next_agent_positions.append(agent_position if collision else tuple(next_agent_position))

        next_state = np.array(next_agent_positions).reshape(-1)

        return next_state

    def predict_reward(self, state, joint_action):
        terminal = self.rewards_model.predict(state).argmax()
        return 100 if terminal else -1.0

    def fit_and_validate(self, X_transition, Y_transition, terminal_states, non_terminal_states, epochs=1, verbose=False):
        print(f"\t\tTransition Model", flush=True)
        transition_loss, _, _ = self.transition_model.fit_and_validate(list(X_transition), list(Y_transition), epochs=epochs, batch_size=self.batch_size, verbose=verbose)
        X_rewards, Y_rewards = self.balance_rewards_dataset(list(terminal_states), list(non_terminal_states))
        if len(X_rewards) > 0:
            print(f"\t\tRewards Model", flush=True)
            reward_loss, _, _ = self.rewards_model.fit_and_validate(X_rewards, Y_rewards, epochs=epochs, batch_size=self.batch_size, verbose=verbose)
        else:
            reward_loss = None
        return transition_loss, reward_loss

    def load(self, directory):
        self.transition_model.load(directory + "/Transition Model")
        self.rewards_model.load(directory + "/Rewards Model")

    def save(self, directory):
        self.transition_model.save(directory + "/Transition Model")
        self.rewards_model.save(directory + "/Rewards Model")

    @staticmethod
    def balance_rewards_dataset(terminal_states, non_terminal_states):

        num_terminal_states = len(terminal_states)
        num_non_terminal_states = len(non_terminal_states)

        if num_terminal_states > num_non_terminal_states:
            terminal_states = random.sample(terminal_states, num_non_terminal_states)
        elif num_non_terminal_states > num_terminal_states:
            non_terminal_states = random.sample(non_terminal_states, num_terminal_states)

        X = terminal_states + non_terminal_states
        y = [1 for _ in range(len(terminal_states))] + [0 for _ in range(len(non_terminal_states))]

        return X, y
