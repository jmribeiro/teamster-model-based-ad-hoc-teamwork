from backend.agents.DQNAgent import MLPDQNAgent as DQNAgent
from backend.agents.adhoc.PLASTICModelAgent import PLASTICModelAgent
from backend.agents.adhoc.PLASTICPolicyAgent import PLASTICPolicyAgent
from backend.agents.adhoc.EnvironmentLearner import EnvironmentLearner
from backend.agents.adhoc.TeamLearner import TeamLearner
from backend.agents.MCTSAgent import MCTSAgent
from backend.agents.adhoc.TEAMSTER import TEAMSTER
from backend.agents.OffPolicyTrainer import OffPolicyTrainer


def load_pretrained_dqn_agent(env, dqn_directory, hyperparams):

    hyperparams = hyperparams['DQN']

    hidden_sizes = hyperparams['hidden sizes']
    num_layers = hyperparams['num layers']
    learning_rate = hyperparams['learning rate']
    initial_exploration_steps = hyperparams['initial exploration steps']
    initial_exploration_rate = hyperparams['initial exploration rate']
    final_exploration_rate = hyperparams['final exploration rate']
    final_exploration_step = hyperparams['final exploration step']
    buffer_size = hyperparams['buffer size']
    batch_size = hyperparams['batch size']
    target_upd = hyperparams['target update frequency']

    dqn_agent = DQNAgent(env.observation_space.shape[0], env.action_space.n,
                         layers=[(hidden_sizes, "relu") for _ in range(num_layers)],
                         learning_rate=learning_rate,
                         initial_exploration_rate=initial_exploration_rate,
                         final_exploration_rate=final_exploration_rate,
                         initial_exploration_steps=initial_exploration_steps,
                         final_exploration_step=final_exploration_step,
                         replay_buffer_size=buffer_size,
                         replay_batch_size=batch_size,
                         target_network_update_frequency=target_upd)
    dqn_agent._name = "DQN"
    dqn_agent.load(dqn_directory)
    dqn_agent.eval()
    return dqn_agent


DQNAgent.load_pretrained = load_pretrained_dqn_agent
