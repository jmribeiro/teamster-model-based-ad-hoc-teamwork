import argparse
from random import getrandbits

import numpy as np
import yaaf
from gym import Env
from yaaf.agents import Agent, RandomAgent

from agents.AdHocAgent import AdHocAgent
from agents.LearnsEnvironmentAgent import LearnsEnvironmentAgent
from agents.LearnsTeamAgent import LearnsTeamAgent
from agents.PLASTICModelAgent import PLASTICModelAgent
from agents.PLASTICPolicyAgent import PLASTICPolicyAgent
from agents.UCTAgent import UCTAgent
from agents.plastic.PLASTICAgent import PLASTICAgent
from agents.teammates.GreedyAgent import GreedyAgent
from agents.teammates.ProbabilisticDestinationsAgent import ProbabilisticDestinationsAgent
from agents.teammates.TeammateAwareAgent import TeammateAwareAgent
from environment.Pursuit import Pursuit


PURSUIT_MAX = 150
PURSUIT_CHECKPOINT = 500
TEAMS = ["teammate aware", "greedy", "probabilistic destinations"]


def train_agent(agent: Agent, environment: Env, timesteps: int):
    log_interval = 10
    state = environment.reset()
    agent.train()
    for step in range(timesteps):
        if step % log_interval == 0:
            print(f"\t\tStep {step+1}/{timesteps}", flush=True)
        action = agent.action(state)
        next_state, reward, terminal, info = environment.step(action)
        timestep = yaaf.Timestep(state, action, reward, next_state, terminal, info)
        agent.reinforcement(timestep)
        state = environment.reset() if terminal else next_state
    print(f"\t\tStep {timesteps}/{timesteps}", flush=True)


def checkpoint_eval(agent: Agent, environment: Env, episodes: int, max_steps: int):

    if isinstance(agent, PLASTICAgent):
        agents_training_beliefs = agent.belief_distribution.copy()
        agent.belief_distribution = np.zeros((len(agent.priors),)) + 1 / len(agent.priors)

    agent.eval()
    results = np.zeros(episodes)

    for episode in range(episodes):
        terminal = False
        step = 0
        state = environment.reset()
        episode_reward = 0.0
        while not terminal and step < max_steps:
            action = agent.action(state)
            next_state, reward, terminal, info = environment.step(action)
            timestep = yaaf.Timestep(state, action, reward, next_state, terminal, info)
            agent.reinforcement(timestep)
            state = next_state
            episode_reward += reward
            step += 1
        print(f"\tEpisode {episode + 1}/{episodes}: {episode_reward}")
        results[episode] = episode_reward

    if isinstance(agent, PLASTICAgent):
        agent.belief_distribution = agents_training_beliefs

    agent.train()
    return results


def final_eval(agent: Agent, environment: Env, episodes: int, max_steps: int, verbose=False):

    agent.eval()
    results = np.zeros(episodes)
    beliefs = []

    for episode in range(episodes):

        if isinstance(agent, PLASTICAgent):
            agent.belief_distribution = np.zeros((len(agent.priors),)) + 1 / len(agent.priors)
            episode_beliefs = [agent.belief_distribution.copy()]

        terminal = False
        step = 0
        state = environment.reset()
        episode_reward = 0.0
        while not terminal and step < max_steps:
            action = agent.action(state)
            next_state, reward, terminal, info = environment.step(action)
            timestep = yaaf.Timestep(state, action, reward, next_state, terminal, info)
            agent.reinforcement(timestep)
            state = next_state
            episode_reward += reward
            step += 1
            if isinstance(agent, PLASTICAgent):
                if verbose:
                    print(agent.belief_distribution)
                episode_beliefs.append(agent.belief_distribution.copy())
        print(f"\tEpisode {episode+1}/{episodes}: {episode_reward}")
        results[episode] = episode_reward

        if isinstance(agent, PLASTICAgent):
            beliefs.append(episode_beliefs)

    print(f"\tMean: {results.mean()}")
    print(f"\tStd: {results.std()}")

    if isinstance(agent, PLASTICAgent):
        return results, beliefs
    else:
        return results


def train_eval(agent: Agent, domain: str, team: str, num_evals: int, trials: int, timesteps=None):

    env, timesteps_per_session, max_steps = setup_environment(team)
    if timesteps is None:
        timesteps = timesteps_per_session
    steps_per_eval = int(timesteps / num_evals)
    evals = []
    for eval in range(num_evals):
        print(f"\tTraining Phase {eval+1}/{num_evals} ({steps_per_eval} steps)", flush=True)
        train_agent(agent, env, steps_per_eval)
        print(f"\tEvaluating Phase {eval+1}/{num_evals}", flush=True)
        evals.append(checkpoint_eval(agent, env, trials, max_steps=max_steps))
        print(f"\tDone: {evals[-1].mean()}", flush=True)
    return evals


def instantiate_agent(agent_name: str, domain: str):

    factory = {
        "pursuit": {

            "PLASTIC-Model": PLASTICModelAgent(3, (5, 5), preload=False),
            "PLASTIC-Policy": PLASTICPolicyAgent(3, (5, 5)),
            "TEAMSTER": AdHocAgent(3, (5, 5)),

            "Learns Environment": LearnsEnvironmentAgent(num_teammates=3, world_size=(5, 5)),
            "Learns Team": LearnsTeamAgent(num_teammates=3, world_size=(5, 5)),

        }
    }

    return factory[domain][agent_name]


def run_adhoc_beliefs(agent_name: str, domain: str, resources: str, num_evals: int, trials: int, asymptotic: bool):

    job_id = getrandbits(16)

    print(f"Job {job_id}: {agent_name} on {domain}", flush=True)

    root = f"{resources}/{domain}/{agent_name}/Job_{job_id}"
    yaaf.mkdir(root)

    # Set up a fresh agent
    agent: PLASTICAgent = instantiate_agent(agent_name, domain)

    for t1, train_team in enumerate(TEAMS):

        print(flush=True)

        if agent_name in ["TEAMSTER", "PLASTIC-Policy", "Learns Team", "Learns Environment"]:

            print(f"Learning {train_team} (Tau{t1+1}/{len(TEAMS)})", flush=True)

            if agent_name == "Learns Environment":
                # Load team model
                agent.load_handcoded_prior(train_team)

            curve = train_eval(agent, domain, train_team, num_evals, trials)
            np.save(f"{root}/{t1}_train_{train_team}.npy", curve)

            if agent_name != "Learns Environment":
                # Save learned team model
                agent.learn_team(train_team, root)

        elif agent_name == "PLASTIC-Model":
            print(f"Setting up {train_team} (Tau{t1+1}/{len(TEAMS)})", flush=True)
            agent.load_handcoded_prior(train_team)

        print(f"Library: {[prior.name for prior in agent.priors]}")

        # Eval phase
        for t2, eval_team in enumerate(TEAMS):
            print(f"Evaluating {eval_team} team", flush=True)
            env, train_steps, max_steps = setup_environment(eval_team)
            evals, beliefs = final_eval(agent, env, episodes=trials, max_steps=max_steps)
            np.save(f"{root}/{t1}_eval_{eval_team}.npy", evals)
            np.save(f"{root}/{t1}_eval_{eval_team}_beliefs.npy", beliefs)

    if asymptotic:

        # Asymptotic Experiment
        asymptotic_team = "mixed"
        asymptotic_timesteps = train_steps * 3
        num_asymptotic_evals = num_evals * 3
        print(f"Running Asymptotic Experiment ({asymptotic_team} team, {asymptotic_timesteps} timesteps)")
        curve = train_eval(agent, domain, asymptotic_team, num_asymptotic_evals, trials, asymptotic_timesteps)
        np.save(f"{resources}/{domain}-asymptotic-{asymptotic_team}/{agent_name}_{asymptotic_team}.npy", curve)

    """
    # Larger World Experiment
    print(f"Running Larger World Experiment", flush=True)
    larger_world = (10, 10)
    # This updates the MCTS depth and state creation,
    # But not the team models
    agent.world_size = larger_world
    for t, eval_team in enumerate(TEAMS):
        print(f"Evaluating {eval_team} team", flush=True)
        env, train_steps, max_steps = setup_environment(eval_team, world_size=larger_world)
        agent.belief_distribution = np.zeros((len(agent.priors),)) + 1 / len(agent.priors)
        evals, beliefs = final_eval(agent, env, episodes=trials, max_steps=max_steps)
        np.save(f"{root}/larger_world_{t}_eval_{eval_team}.npy", evals)
        np.save(f"{root}/larger_world_{t}_eval_{eval_team}_beliefs.npy", beliefs)
    """


def run_baseline(agent_name, domain, resources, trials):

    job_id = getrandbits(16)
    print(f"Job {job_id}: {agent_name} on {domain}", flush=True)
    root = f"{resources}/{domain}/{agent_name}/Job_{job_id}"
    yaaf.mkdir(root)

    for eval_team in TEAMS:

        print(f"Evaluating {eval_team} team", flush=True)

        factory = {
            "pursuit": {
                "UCT": UCTAgent(3, (5, 5), eval_team),
                "Random Policy": RandomAgent(num_actions=4),
                "Original Teammate": {
                    "greedy": GreedyAgent(0, (5, 5)),
                    "teammate aware": TeammateAwareAgent(0, (5, 5)),
                    "probabilistic destinations": ProbabilisticDestinationsAgent(0, (5, 5))
                }[eval_team]
            }
        }[domain]

        agent = factory[agent_name]

        env, _, max_steps = setup_environment(eval_team)
        aux = final_eval(agent, env, episodes=trials, max_steps=max_steps)
        evals = aux[0] if isinstance(aux, tuple) else aux   # Needed for UCT which is PLASTIC-Model with known beliefs
        yaaf.mkdir(root)
        np.save(f"{root}/{0}_eval_{eval_team}.npy", evals)
        np.save(f"{root}/{1}_eval_{eval_team}.npy", evals)
        np.save(f"{root}/{2}_eval_{eval_team}.npy", evals)


def setup_environment(team: str, world_size=(5, 5)):
    env = Pursuit(teammates=team, world_size=world_size)
    train_steps = 5000
    max_eval_steps = PURSUIT_MAX
    return env, train_steps, max_eval_steps


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('agent')

    parser.add_argument('--evals', "-e", type=int, default=10)
    parser.add_argument('--trials', "-t", type=int, default=8)

    parser.add_argument('--asymptotic', action="store_true")

    parser.add_argument('--resources', "-r", type=str, default="resources")

    opt = parser.parse_args()
    opt.agent = opt.agent.replace("_", " ")
    opt.domain = "pursuit"

    # Very slow to run
    if opt.agent in ["TEAMSTER", "PLASTIC-Model", "PLASTIC-Policy", "Learns Team", "Learns Environment"]:
        run_adhoc_beliefs(opt.agent, opt.domain, opt.resources, opt.evals, opt.trials, opt.asymptotic)

    # Fast to run
    elif opt.agent in ["UCT", "Random Policy", "Original Teammate"]:
        run_baseline(opt.agent, opt.domain, opt.resources, opt.trials)
