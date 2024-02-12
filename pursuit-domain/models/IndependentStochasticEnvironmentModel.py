import numpy as np

from models.PursuitEnvironmentModel import PursuitEnvironmentModel
from models.utils import transition_network_features, compute_direction
from environment.utils import prey_directions
from models.FeedForwardNetwork import FeedForwardNetwork


class IndependentStochasticEnvironmentModel(PursuitEnvironmentModel):

    def __init__(self, num_agents, trainable, learning_rate, layers, replay_batch_size, replay_memory_size, fix_collisions=True, learn_reward=False):

        super().__init__(num_agents, "independent stochastic environment model", trainable, learning_rate, layers, replay_batch_size, replay_memory_size, fix_collisions, learn_reward)

        self.total_agents = num_agents
        self.prediction_cache = dict()

        axis = 2
        self.directions = prey_directions()
        directions = len(self.directions)
        num_entities = num_agents + 1
        coordinates = (num_entities - 1) * axis
        actions_one_hot = num_agents * 4

        self.num_features = num_entities + coordinates + actions_one_hot
        self.num_outputs = directions
        self.learning_rate = learning_rate

        self.model = FeedForwardNetwork(self.num_features, self.num_outputs, layers, learning_rate, cuda=True)

    # ########## #
    # PREDICTION #
    # ########## #

    def _predict_next_state(self, state, actions):
        next_possible_state = state + self.predict_offsets(state, actions)
        return next_possible_state

    def predict_offsets(self, state, actions):
        directions = []
        for entity in range(self.num_agents+1):
            features = transition_network_features(state, actions, entity)
            movement_scores = self.get_from_cache(features) if self.in_cache(features) else self.uncached_prediction(features)
            most_likely_direction_index = movement_scores.argmax()
            direction = self.directions[most_likely_direction_index]
            directions.append(direction)
        offsets = np.array(directions).reshape((((self.total_agents + 1) * 2),))
        return offsets

    # ######## #
    # Training #
    # ######## #

    def prepare_batch(self, batch):

        X = []
        Y = []

        for i, (state, actions, reward, next_state, terminal) in enumerate(batch):
            for entity in range(self.num_agents+1):
                X.append(transition_network_features(state, actions, entity))
                Y.append(self.directions.index(compute_direction(state, next_state, entity)))

        X = np.array(X)
        y = np.array(Y)

        return X, y

    def update_state_predictors(self, batch, verbose):
        if verbose: print("***Fitting Movements***", flush=True)
        X, y = self.prepare_batch(batch)
        losses, _, accuracies = self.model.update(X, y, epochs=1, batch_size=self._replay_batch_size, verbose=verbose)
        info = {"training loss": losses[-1], "training accuracy": accuracies[-1]}
        return info

    # ################# #
    # Caching Mechanism #
    # ################# #

    def in_cache(self, x):
        cache_key = tuple(x)
        return cache_key in self.prediction_cache

    def get_from_cache(self, x):
        cache_key = tuple(x)
        movement_probabilities = self.prediction_cache[cache_key]
        return movement_probabilities

    def uncached_prediction(self, x):
        movement_scores = self.predict(x)
        cache_key = tuple(x)
        self.prediction_cache[cache_key] = movement_scores
        return movement_scores

    def predict(self, x):
        x = x.reshape(1, -1)
        movement_scores = self.model.predict(x)
        return movement_scores

    def _save(self, directory):
        self.model.save(f"{directory}/movements")

    def _load(self, directory):
        super(IndependentStochasticEnvironmentModel, self)._load(directory)
        self.model.load(f"{directory}/movements")
