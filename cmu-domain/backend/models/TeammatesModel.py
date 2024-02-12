import torch
from yaaf.models.FeedForwardNetwork import FeedForwardNetwork
from torch.nn.functional import softmax
from yaaf.policies import sample_action


class TeammatesModel:

    def __init__(self, state_features, num_teammates, num_actions, hyperparams):

        self.hyperparams = hyperparams
        self.num_teammates = num_teammates
        self.num_actions = num_actions
        self.batch_size = hyperparams["MBMCTS"]["buffer"]["batch size"]

        self.models = [
            FeedForwardNetwork(
                num_inputs=state_features, num_outputs=self.num_actions,
                layers=[
                    (hyperparams["MBMCTS"]["teammates model"]["hidden sizes"], "relu")
                    for _ in range(hyperparams["MBMCTS"]["teammates model"]["num layers"])
                ],
                learning_rate=hyperparams["MBMCTS"]["teammates model"]["learning rate"], cuda=torch.cuda.is_available()
            )
            for _ in range(self.num_teammates)
        ]

    def fit_and_validate(self, X, Y, epochs=1, verbose=False):
        losses = {}
        for t in range(self.num_teammates):
            print(f"\t\tTeammate {t+1}", flush=True)
            teammates_loss, _, _ = self.models[t].fit_and_validate(
                list(X), list(Y[t]), epochs=epochs, batch_size=self.batch_size, verbose=verbose)
            losses[f"teammate {t + 1} model loss"] = teammates_loss
        return losses

    def predict_policies(self, state):
        policies = []
        for t in range(self.num_teammates):
            scores = self.models[t].predict(state)
            policy = softmax(scores, dim=0).numpy()
            policy /= policy.sum()
            policies.append(policy)
        return policies

    def predict_actions(self, state):
        policies = self.predict_policies(state)
        actions = [sample_action(policy) for policy in policies]
        return actions

    def save(self, directory):
        [self.models[t].save(directory+f"/Teammate Model {t+1}") for t in range(self.num_teammates)]

    def load(self, directory):
        [self.models[t].load(directory+f"/Teammate Model {t+1}") for t in range(self.num_teammates)]
