import argparse

from run import check_checkpoint_cache, run_with_beliefs, run_baseline

# #### #
# MAIN #
# #### #


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('domain')

    parser.add_argument('--evals', "-e", type=int, default=10)
    parser.add_argument('--trials', "-t", type=int, default=8)

    parser.add_argument('--resources', "-r", type=str, default="../ResultsCMU")

    opt = parser.parse_args()

    for agent in ["UCT", "Random Policy", "Original Teammate"]:
        run_baseline(agent, opt.domain, opt.resources, opt.trials)

    for agent in ["TEAMSTER", "PLASTIC-Policy", "Learns Team", "Learns Environment"]:
        check_checkpoint_cache(opt.domain, opt.evals, opt.trials, opt.resources)
        run_with_beliefs(agent, opt.domain, opt.resources, opt.evals, opt.trials)

    run_with_beliefs("PLASTIC-Model", opt.domain, opt.resources, opt.evals, opt.trials)
