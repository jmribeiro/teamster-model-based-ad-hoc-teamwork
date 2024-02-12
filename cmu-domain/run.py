import yaaf
import argparse
import numpy as np
from os import path

import yaml
from gym import Env
from yaaf.agents import Agent, RandomAgent

from backend.environments import cmu
from backend.agents.adhoc import PLASTICAgent
from backend.agents import DQNAgent, PLASTICModelAgent, MCTSAgent, TeamLearner, EnvironmentLearner, PLASTICPolicyAgent, TEAMSTER, OffPolicyTrainer
from backend.configs import TEAMS


# ###### #
# Ad Hoc #
# ###### #

def instantiate_checkpoint_agent(agent_name, domain, checkpoint_directories, hyperparams):

    teams_in_library = [directory.split("/")[-1] for directory in checkpoint_directories]
    envs = [setup_environment(domain, team)[0] for team in teams_in_library]

    # PLASTIC
    if agent_name == "PLASTIC-Policy":
        agent = PLASTICPolicyAgent(envs, checkpoint_directories, hyperparams)
    elif agent_name == "PLASTIC-Model":
        agent = PLASTICModelAgent(envs, hyperparams)

    # Ours
    elif agent_name == "TEAMSTER":
        agent = TEAMSTER(envs, checkpoint_directories, hyperparams)

    # Auxiliary
    elif agent_name == "Learns Team":
        agent = TeamLearner(envs, checkpoint_directories, hyperparams)
    elif agent_name == "Learns Environment":
        agent = EnvironmentLearner(envs, checkpoint_directories, hyperparams)

    else:
        raise ValueError(f"Invalid agent {agent_name}")

    return agent


def offline_train_eval(agent_name, domain, current_team, known_teams, num_evals, num_trials, resources):
    hyperparams = load_yaml(f"hyperparams.yaml", domain)
    env, timesteps, max_eval_steps = setup_environment(domain, current_team)
    steps_per_eval = int(timesteps / num_evals)
    evals = []
    for eval_phase in range(num_evals):
        print(f"\tTraining Phase {eval_phase+1}/{num_evals} ({steps_per_eval} steps)", flush=True)
        known_teams_checkpoint = steps_per_eval * num_evals
        known_teams_checkpoint_directories = [f"{resources}/{domain}/DQN/Checkpoint_{known_teams_checkpoint}/{known_team}" for known_team in known_teams]
        current_team_checkpoint = steps_per_eval * (eval_phase+1)
        current_team_checkpoint_directory = f"{resources}/{domain}/DQN/Checkpoint_{current_team_checkpoint}/{current_team}"
        checkpoint_directories = known_teams_checkpoint_directories + [current_team_checkpoint_directory]
        train_eval_checkpoints_directory = f"{resources}/{domain}/{agent_name}/offline_train_eval_checkpoints"
        yaaf.mkdir(train_eval_checkpoints_directory)
        checkpoint_result_filename = f"{train_eval_checkpoints_directory}/{len(known_teams)}_{current_team}_phase_{eval_phase}.npy"
        try:
            result = np.load(checkpoint_result_filename)
            if isinstance(result, tuple): result = result[0]
            if result.shape[0] < num_trials:
                raise FileNotFoundError()
        except Exception:
            agent = instantiate_checkpoint_agent(agent_name, domain, checkpoint_directories, hyperparams)
            print(f"\tEvaluating Phase {eval_phase+1}/{num_evals}", flush=True)
            result = checkpoint_eval(agent, env, num_trials, max_eval_steps, allow_fail=True)
            if isinstance(result, tuple): result = result[0]
            np.save(checkpoint_result_filename, result)
        evals.append(result)
        print(f"\tDone: {evals[-1].mean()}", flush=True)

    return evals


def checkpoint_eval(agent: Agent, environment: Env, episodes: int, max_steps: int, render=False, allow_fail=False):

    if isinstance(agent, PLASTICAgent):
        beliefs = []
        agents_training_beliefs = agent.beliefs.copy()

    agent.eval()
    results = np.zeros(episodes)

    for episode in range(episodes):

        if allow_fail:
            result = run_episode(agent, environment, max_steps)
            episode_reward = result[0] if isinstance(agent, PLASTICAgent) else result
            if episode_reward == -1:
                episode_reward = -1 * max_steps
                if isinstance(agent, PLASTICAgent): result = episode_reward, result[1]
                else: result = episode_reward
        else:
            result = -1
            while result == -1:
                result = run_episode(agent, environment, max_steps)

        if isinstance(agent, PLASTICAgent):
            episode_reward, episode_beliefs = result
            results[episode] = episode_reward
            beliefs.append(episode_beliefs)
        else:
            episode_reward = result
            results[episode] = episode_reward

        print(f"\t\tEpisode {episode + 1}/{episodes}: {episode_reward}")

    agent.train()

    if isinstance(agent, PLASTICAgent):
        agent.beliefs = agents_training_beliefs
        return results, beliefs
    else:
        return results


def evaluate_on_team_given_library(agent_name, domain, eval_team, known_teams, num_evals, num_trials, resources):

    env, train_steps, max_eval_steps = setup_environment(domain, eval_team)
    eval_phase = num_evals - 1
    steps_per_eval = int(train_steps / num_evals)

    # Assumes training has ended and takes last checkpoint
    known_teams_checkpoint = steps_per_eval * num_evals
    known_teams_checkpoint_directories = [f"{resources}/{domain}/DQN/Checkpoint_{known_teams_checkpoint}/{known_team}" for known_team in known_teams]
    agent = instantiate_checkpoint_agent(agent_name, domain, known_teams_checkpoint_directories, load_yaml(f"hyperparams.yaml", domain))

    print(f"\tEvaluating Phase {eval_phase + 1}/{num_evals}", flush=True)

    result = checkpoint_eval(agent, env, num_trials, max_eval_steps, allow_fail=False)

    if isinstance(result, tuple): print(f"\tDone: {result[0].mean()}", flush=True)
    else: print(f"\tDone: {result.mean()}", flush=True)

    return result


def run_with_beliefs(agent_name: str, domain: str, resources: str, num_evals: int, trials: int):

    print(f"\t{agent_name} on {domain}", flush=True)
    root = f"{resources}/{domain}/{agent_name}"
    yaaf.mkdir(root)

    known_teams = []
    for t1, train_team in enumerate(TEAMS):

        print(flush=True)

        # Training phase
        if agent_name in ["TEAMSTER", "PLASTIC-Policy", "Learns Team", "Learns Environment"]:
            print(f"\tLearning {train_team} (Tau{t1 + 1}/{len(TEAMS)})", flush=True)

            try:
                curve = np.load(f"{root}/{t1}_train_{train_team}.npy")
            except Exception:
                curve = offline_train_eval(agent_name, domain, train_team, known_teams, num_evals, trials, resources)
                np.save(f"{root}/{t1}_train_{train_team}.npy", curve)

        known_teams.append(train_team)

        # Eval Phase
        for t2, eval_team in enumerate(TEAMS):

            print(f"\tEvaluating {eval_team} team (agent knows {known_teams})", flush=True)
            try:
                np.load(f"{root}/{t1}_eval_{eval_team}.npy")
                np.load(f"{root}/{t1}_eval_{eval_team}_beliefs.npy", allow_pickle=True)
            except Exception:
                eval, beliefs = evaluate_on_team_given_library(agent_name, domain, eval_team, known_teams, num_evals, trials, resources)
                np.save(f"{root}/{t1}_eval_{eval_team}.npy", eval)
                np.save(f"{root}/{t1}_eval_{eval_team}_beliefs.npy", beliefs)
            print("\tDone!", flush=True)


# ########## #
# Non-Ad Hoc #
# ########## #

def run_baseline(agent_name, domain, resources, num_trials):

    print(f"\t{agent_name} on {domain}", flush=True)
    root = f"{resources}/{domain}/{agent_name}"
    yaaf.mkdir(root)

    hyperparams = load_yaml(f"hyperparams.yaml", domain)

    for eval_team in TEAMS:

        print(f"\tEvaluating {eval_team} team", flush=True)

        try:

            for eval_team_id in range(len(TEAMS)):
                result = np.load(f"{root}/{eval_team_id}_eval_{eval_team}.npy")
                if result.size < num_trials:
                    raise FileNotFoundError()

        except FileNotFoundError:

            env, _, max_steps = setup_environment(domain, eval_team)

            factory = {
                "UCT": MCTSAgent(env, hyperparams),
                "Random Policy": RandomAgent(num_actions=env.action_space.n),
                "Original Teammate": {
                    "greedy": env.make_greedy(),
                    "teammate aware": env.make_teammate_aware(),
                    "probabilistic destinations": env.make_probabilistic_destinations()
                }[eval_team]
            }

            agent = factory[agent_name]

            aux = final_eval(agent, env, episodes=num_trials, max_steps=max_steps)

            # Needed for UCT which is PLASTIC-Model with known beliefs
            evals = aux[0] if isinstance(aux, tuple) else aux
            yaaf.mkdir(root)

            # Just to make plotting easier
            [np.save(f"{root}/{eval_team_id}_eval_{eval_team}.npy", evals) for eval_team_id in range(len(TEAMS))]

        print(f"\tDone!", flush=True)


def run_episode(agent, environment, max_steps):

    if isinstance(agent, PLASTICAgent):
        agent.beliefs = np.zeros((len(agent.beliefs),)) + 1 / len(agent.beliefs)
        episode_beliefs = [agent.beliefs]

    terminal = False
    step = 0
    state = environment.reset()

    episode_reward = 0.0

    while not terminal:

        action = agent.action(state)
        next_state, reward, terminal, info = environment.step(action)
        timestep = yaaf.Timestep(state, action, reward, next_state, terminal, info)
        agent.reinforcement(timestep)
        state = next_state
        episode_reward += reward
        step += 1
        if isinstance(agent, PLASTICAgent):
            episode_beliefs.append(agent.beliefs)

        if step == max_steps:
            #print("FAIL")
            episode_reward = -1
            break

    if isinstance(agent, PLASTICAgent):
        return episode_reward, episode_beliefs
    else:
        return episode_reward


def final_eval(agent: Agent, environment: Env, episodes: int, max_steps: int):

    agent.eval()
    results = np.zeros(episodes)

    if isinstance(agent, PLASTICAgent): beliefs = []

    for episode in range(episodes):

        result = -1
        while result == -1:
            result = run_episode(agent, environment, max_steps)

        if isinstance(agent, PLASTICAgent):
            episode_reward, episode_beliefs = result
            results[episode] = episode_reward
            beliefs.append(episode_beliefs)
        else:
            episode_reward = result
            results[episode] = episode_reward

        print(f"\tEpisode {episode + 1}/{episodes}: {episode_reward}")

    print(f"\tMean: {results.mean()}")
    print(f"\tStd: {results.std()}")

    if isinstance(agent, PLASTICAgent):
        return results, beliefs
    else:
        return results


# ############ #
# Pre-Training #
# ############ #

def check_checkpoint_cache(domain, num_evals, num_trials, resources):

    hyperparams = load_yaml(f"hyperparams.yaml", domain)

    for team in TEAMS:

        _, train_steps, max_eval_steps = setup_environment(domain, team)
        steps_per_eval = int(train_steps / num_evals)
        needs_caching = False

        for eval_phase in range(num_evals):
            checkpoint_directory = f"{resources}/{domain}/DQN/Checkpoint_{steps_per_eval*(eval_phase+1)}/{team}"
            checkpoint_done = path.exists(f"{checkpoint_directory}/done.txt")
            if not checkpoint_done:
                needs_caching = True
                break

        if needs_caching:
            build_checkpoint_cache(domain, team, num_evals, num_trials, hyperparams, resources)
        else:
            #print(f"{domain} on team {team} cached")
            pass


def build_checkpoint_cache(domain, team, num_evals, num_trials, hyperparams, resources):

    dqn_hyperparams = hyperparams["DQN"]
    hidden_sizes = dqn_hyperparams['hidden sizes']
    num_layers = dqn_hyperparams['num layers']
    learning_rate = dqn_hyperparams['learning rate']
    initial_exploration_steps = dqn_hyperparams['initial exploration steps']
    initial_exploration_rate = dqn_hyperparams['initial exploration rate']
    final_exploration_rate = dqn_hyperparams['final exploration rate']
    final_exploration_step = dqn_hyperparams['final exploration step']
    buffer_size = dqn_hyperparams['buffer size']
    batch_size = dqn_hyperparams['batch size']
    target_upd = dqn_hyperparams['target update frequency']

    env, train_steps, max_eval_steps = setup_environment(domain, team)

    steps_per_phase = int(train_steps / num_evals)

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
    offpolicytrainer = OffPolicyTrainer(env, hyperparams)

    curve = []
    for eval_phase in range(num_evals):

        print(f"Checkpoint Cache {domain}-{team}, Phase {eval_phase+1}/{num_evals}", flush=True)
        checkpoint = steps_per_phase * (eval_phase + 1)
        checkpoint_directory = f"{resources}/{domain}/DQN/Checkpoint_{checkpoint}/{team}"

        # Check if exists already
        try:

            with open(f"{checkpoint_directory}/done.txt", "w"):
                pass

            dqn_agent.network.load(checkpoint_directory)
            offpolicytrainer.load(checkpoint_directory)

            for timestep in np.load(f"{checkpoint_directory}/buffer.npy", allow_pickle=True):
                timestep = yaaf.Timestep(timestep[0], timestep[1], timestep[2], timestep[3], timestep[4], timestep[5])
                dqn_agent.remember(timestep)
                offpolicytrainer.reinforcement(timestep)

        except FileNotFoundError:

            dqn_agent = train_with_observers(dqn_agent, env, steps_per_phase, log_interval=int(steps_per_phase/10), observers=[offpolicytrainer])

            yaaf.mkdir(checkpoint_directory)

            print(f"Saving DQN", flush=True)
            dqn_agent.network.save(checkpoint_directory)
            np.save(f"{checkpoint_directory}/buffer.npy", dqn_agent._replay_buffer.all, allow_pickle=True)

            # Saving All
            print(f"Training models", flush=True)

            offpolicytrainer.train_models(epochs=hyperparams["MBMCTS"]['epochs'], verbose=False)
            offpolicytrainer.save(checkpoint_directory)

            with open(f"{checkpoint_directory}/done.txt", "w"):
                pass

        try:
            result = np.load(f"{checkpoint_directory}/result.npy")
        except FileNotFoundError:
            result = checkpoint_eval(dqn_agent, env, num_trials, max_eval_steps, allow_fail=True)
            np.save(f"{checkpoint_directory}/result.npy", result)

        if isinstance(result, tuple): result = result[0]
        print(f"Eval Done: {result.mean()}(+-{result.std().round()})", flush=True)
        curve.append(result)

    np.save(f"{resources}/{domain}/DQN/{team}_train.npy", curve)


def train_with_observers(agent, env, timesteps, log_interval, observers=None):

    if observers is None: observers = []

    agent.train()
    state = env.reset()
    episode_steps = 0
    max_training_episode_steps = 999

    for t in range(timesteps):
        action = agent.action(state)
        next_state, reward, terminal, info = env.step(action)
        yaaf_step = yaaf.Timestep(state, action, reward, next_state, terminal, info)
        agent.reinforcement(yaaf_step)
        for observer in observers: observer.reinforcement(yaaf_step)

        episode_steps += 1

        if terminal or episode_steps == max_training_episode_steps:
            state = env.reset()
            episode_steps = 0
        else:
            state = next_state

        if t % log_interval == 0:
            print(f"\tStep {t + 1}/{timesteps}", flush=True)

    return agent


# ######### #
# Auxiliary #
# ######### #

def setup_environment(domain: str, team: str):
    env = cmu.create(team, domain)
    if domain == "isr": train_steps = 100000
    elif domain == "ntu": train_steps = 100000
    elif domain == "pentagon": train_steps = 125000
    else:
        raise ValueError()
    max_eval_steps = 100
    return env, train_steps, max_eval_steps


def load_yaml(filename, environment):
    with open(filename, "r") as file:
        hyperparams = yaml.load(file, Loader=yaml.Loader)
    try:
        hyperparams = hyperparams[environment]
    except KeyError:
        hyperparams = hyperparams["default"]
    return hyperparams


# #### #
# MAIN #
# #### #


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('agent')
    parser.add_argument('domain')

    parser.add_argument('--evals', "-e", type=int, default=10)
    parser.add_argument('--trials', "-t", type=int, default=8)

    parser.add_argument('--resources', "-r", type=str, default="../ResultsCMU")

    opt = parser.parse_args()
    opt.agent = opt.agent.replace("_", " ")

    if opt.agent in ["TEAMSTER", "PLASTIC-Policy", "Learns Team", "Learns Environment"]:
        check_checkpoint_cache(opt.domain, opt.evals, opt.trials, opt.resources)
        run_with_beliefs(opt.agent, opt.domain, opt.resources, opt.evals, opt.trials)
    elif opt.agent == "PLASTIC-Model":
        run_with_beliefs(opt.agent, opt.domain, opt.resources, opt.evals, opt.trials)
    elif opt.agent in ["UCT", "Random Policy", "Original Teammate"]:
        run_baseline(opt.agent, opt.domain, opt.resources, opt.trials)
