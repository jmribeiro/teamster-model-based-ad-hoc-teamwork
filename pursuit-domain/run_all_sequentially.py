import argparse
from run import run_adhoc_beliefs, run_baseline

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--evals', "-e", type=int, default=10)
    parser.add_argument('--trials', "-t", type=int, default=8)

    parser.add_argument('--resources', "-r", type=str, default="resources/Results")

    opt = parser.parse_args()
    opt.domain = "pursuit"

    # Very slow to run
    for agent in ["TEAMSTER", "PLASTIC-Model", "PLASTIC-Policy", "Learns Team", "Learns Environment"]:
        run_adhoc_beliefs(agent, opt.domain, opt.resources, opt.evals, opt.trials, False)

    # Fast to run
    for agent in ["UCT", "Random Policy", "Original Teammate"]:
        run_baseline(agent, opt.domain, opt.resources, opt.trials)
