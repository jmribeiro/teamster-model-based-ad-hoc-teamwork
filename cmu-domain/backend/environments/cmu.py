import itertools
import math
import random
from abc import ABC
from typing import List

import numpy as np
import torch.nn.functional

import yaaf
from numpy import ndarray
from yaaf import Timestep
from yaaf.agents import Agent
from yaaf.policies import random_policy

from backend.configs import TEAMS
from backend.environments.cmu_gridworld_py import layouts
from backend.environments.cmu_gridworld_py.layouts import TAGS, BLOCKED, GOAL, OPEN, START

from backend.environments.LightEnvironment import LightEnvironment
from backend.environments.pursuit_py.astar import A_star_search
from backend.environments.pursuit_py.utils import agent_directions, action_meanings


ACTION_MEANINGS = [
    "up",
    "down",
    "left",
    "right",
    "stay"
]
ACTION_SPACE = tuple(range(len(ACTION_MEANINGS)))
NUM_ACTIONS = len(ACTION_MEANINGS)
UP, DOWN, LEFT, RIGHT, STAY = range(NUM_ACTIONS)

BLOCKIN_STEPS = math.inf


def create(team, environment):

    domain = environment.replace("old_", "") if "old_" in environment else environment

    layout = layouts.load(domain)
    num_rows, num_columns = layout.shape

    _obstacles = obstacles(layout)
    _goals = goals(layout)
    _start_positions = start_positions(layout)

    num_agents = len(_goals)
    joint_action_space = list(itertools.product(ACTION_SPACE, repeat=num_agents))

    for goal in _goals:
        assert goal not in _obstacles

    factory = {
        "greedy": lambda id: OldCMUTeammate(id, "greedy", layout, _goals, _obstacles),
        "teammate aware": lambda id: OldCMUTeammate(id, "teammate aware", layout, _goals, _obstacles),
        "probabilistic destinations": lambda id: OldCMUTeammate(id, "probabilistic destinations", layout, _goals, _obstacles)
    }

    teammates = [factory[team](id) for id in range(1, num_agents)]

    reset_fn = lambda: random_state(_start_positions, num_agents)
    transition_fn = lambda state, joint_action: transition(state, joint_action, joint_action_space, layout, _goals)
    reward_fn = lambda state, joint_action: 100.0 if solved(state, _goals) else -1.0
    teammates_policy_fn = lambda state: [teammate.policy(state) for teammate in teammates]

    id = f"{domain}-v{TEAMS.index(team)}"
    env = LightEnvironment(id, reset_fn, transition_fn, reward_fn, teammates_policy_fn, 0, max(num_rows-1, num_columns-1), NUM_ACTIONS, -1.0, 100.0)
    env.layout = layout
    env.goals = _goals
    env.obstacles = _obstacles
    env.world_size = layout.shape

    env.make_original = lambda: factory[team](id=0)
    env.make_greedy = lambda: factory["greedy"](id=0)
    env.make_teammate_aware = lambda: factory["teammate aware"](id=0)
    env.make_probabilistic_destinations = lambda: factory["probabilistic destinations"](id=0)

    return env


class OldCMUTeammate(Agent):

    def __init__(self, index, type, layout, goals, obstacles):
        super().__init__(type)
        self.type = type
        self.index = index
        self.layout = layout
        self.goals = goals
        self.obstacles = obstacles

    def clean_state(self, state):
        if not isinstance(state, np.ndarray):
            state = state.numpy()
            state = np.round(state, decimals=0)
            state = state.astype(int)
        return state

    def parse_positions(self, state):
        teammates = []
        teammate_indices = []
        me = None  # Should not happen to be returned None
        for i in range(state.size):
            start_of_cell = i % 2 == 0
            if start_of_cell:
                index = int(i / 2)
                cell = (state[i], state[i + 1])
                is_my_position = self.index == index
                if is_my_position:
                    me = cell
                else:
                    teammate_indices.append(index)
                    teammates.append(cell)
        return me, teammates, teammate_indices

    def distance_to_goals(self, agent_cell, obstacles):

        num_rows, num_columns = self.layout.shape
        distances = []
        for goal in self.goals:
            action, distance = A_star_search(agent_cell, set(obstacles), goal, (num_rows, num_columns))
            distances.append(distance if action is not None else BLOCKIN_STEPS)
        return np.array(distances)

    def goto_goal(self, cell, goal, layout, obstacles):

        num_rows, num_columns = layout.shape
        action, _ = A_star_search(cell, set(obstacles), goal, (num_rows, num_columns))

        if action is None:
            return yaaf.policies.random_policy(NUM_ACTIONS)
        elif action == (0, 0):
            stay = ACTION_MEANINGS.index("stay")
            return yaaf.policies.policy_from_action(stay, NUM_ACTIONS)
        else:
            a_backend = agent_directions().index(action)
            a_backend_meaning = action_meanings()[a_backend]
            a = ["Up", "Down", "Left", "Right"].index(a_backend_meaning)
            return yaaf.policies.policy_from_action(a, NUM_ACTIONS)

    def find_closest_free_goal(self, goals_and_distances, teammates):
        best_goal = None
        for goal, distance in goals_and_distances:
            goal_is_free = True
            for teammate in teammates:
                if teammate == goal:
                    goal_is_free = False
                    break
            if goal_is_free:
                best_goal = goal
                break
        return best_goal

    def policy(self, state: ndarray):

        state = self.clean_state(state)
        me, teammates, teammate_indices = self.parse_positions(state)

        # My distance to goals
        my_obstacles = self.obstacles + teammates

        current_agent_positions = parse_positions(state)
        _layout = self.layout.copy()
        for i, agent_position in enumerate(current_agent_positions):
            _layout[agent_position[1], agent_position[0]] = 4 if self.index == i else 4
        for obs in my_obstacles:
            _layout[obs[1], obs[0]] = -1

        my_distance_to_goals = self.distance_to_goals(me, my_obstacles)

        # refactor this later into subclasses...

        if self.type == "greedy":

            goals_and_distances = {}
            for g in range(len(my_distance_to_goals)):
                goals_and_distances[self.goals[g]] = my_distance_to_goals[g]
            goals_and_distances = sorted(goals_and_distances.items(), key=lambda x: x[1])
            best_goal = self.find_closest_free_goal(goals_and_distances, teammates)

        elif self.type == "teammate aware" or self.type == "probabilistic destinations":

            # Other distances to goals
            agents_distances_to_goals = []
            for agent_id in range(len(teammates)+1):
                if agent_id == self.index:
                    agent_distance_to_goals = my_distance_to_goals
                else:
                    agent_teammates = [teammate for t2, teammate in enumerate(teammates) if t2 != agent_id]
                    agent_obstacles = self.obstacles + [me] + agent_teammates
                    agent_cell = teammates[teammate_indices.index(agent_id)]
                    agent_distance_to_goals = self.distance_to_goals(agent_cell, agent_obstacles)

                agents_distances_to_goals.append(agent_distance_to_goals)

            possible_shared_strategies = list(itertools.permutations(list(range(len(self.goals))), r=len(teammates)+1))

            steps_per_strategy = []
            for shared_strategy_goals in possible_shared_strategies:
                strategy_steps = 0.0
                for agent_id in range(len(shared_strategy_goals)):
                    agent_goal = shared_strategy_goals[agent_id]
                    agent_distance_to_goal = agents_distances_to_goals[agent_id][agent_goal]
                    strategy_steps += agent_distance_to_goal
                steps_per_strategy.append(strategy_steps)
            steps_per_strategy = np.array(steps_per_strategy)

            argmins = np.argwhere(steps_per_strategy == np.min(steps_per_strategy)).reshape(-1)

            if self.type == "teammate aware":
                chosen_strategy_id = random.choice(argmins)

            elif self.type == "probabilistic destinations":

                # Remove ones where one of the agents is blocked
                non_blocking_strategies_ids = np.argwhere(steps_per_strategy != np.inf).reshape(-1)

                if len(non_blocking_strategies_ids) > 1:

                    non_blocking_steps_per_strategy = np.array([steps_per_strategy[i] for i in non_blocking_strategies_ids])

                    # The more the cost, the lower the prob of being chosen
                    softmin_strategy_probabilities = torch.nn.functional.softmin(torch.from_numpy(non_blocking_steps_per_strategy), dim=0).numpy()

                    chosen_strategy_id = np.random.choice(non_blocking_strategies_ids, p=softmin_strategy_probabilities)

                    #try:
                    #    chosen_strategy_id = np.random.choice(non_blocking_strategies_ids, p=softmin_strategy_probabilities)
                    #except ValueError as e:
                    #    print(f"ERROR: {softmin_strategy_probabilities} doesnt add to 1.0 (sum={softmin_strategy_probabilities.sum()})")
                    #    print(softmin_strategy_probabilities)
                    #    raise e

                else:
                    # Reduced to teammate aware
                    chosen_strategy_id = random.choice(argmins)
            else:
                raise ValueError(
                    "Unreachable... "
                    "Please, implement using three sub classes... "
                    "It's making my OOP soul hurt a little")

            best_strategy = possible_shared_strategies[chosen_strategy_id]
            best_goal_id = best_strategy[self.index]
            best_goal = self.goals[best_goal_id]

        else:
            raise ValueError(f"Invalid type {self.type}")

        policy = self.goto_goal(me, best_goal, self.layout, my_obstacles)

        return policy

    def _reinforce(self, timestep: Timestep):
        pass


# ##### #
# Reset #
# ##### #

def random_state(start_positions, num_agents):
    positions = random.sample(start_positions, k=num_agents)
    state = np.array(positions).reshape(-1)
    return state


# ########## #
# Transition #
# ########## #

def parse_positions(state):
    agent_positions: List[tuple] = []
    for i in range(state.size):
        start_of_cell = i % 2 == 0
        if start_of_cell:
            cell = (state[i], state[i + 1])
            agent_positions.append(cell)
    return agent_positions


def transition(state, joint_action, joint_action_space, layout, goals):

    actions = joint_action_space[joint_action]

    if solved(state, goals):
        # Stays in terminal state (has to be .reset())
        return state

    else:

        # Regular transitions
        action_meanings = [ACTION_MEANINGS[a] for a in actions]
        current_agent_positions = parse_positions(state)

        num_agents = len(current_agent_positions)

        # Randomize move order for collision checking
        move_order = list(range(num_agents))
        #random.shuffle(move_order)

        for i1 in move_order:

            next_position = next_cell(current_agent_positions[i1][0], current_agent_positions[i1][1], action_meanings[i1], layout)
            collision_with_any_teammate = False

            for i2 in range(num_agents):
                is_other_agent = i1 != i2
                if is_other_agent:
                    teammate_position = current_agent_positions[i2]
                    collision_with_any_teammate = next_position == teammate_position
                    if collision_with_any_teammate: break

            if collision_with_any_teammate:
                next_position = current_agent_positions[i1][0], current_agent_positions[i1][1]

            current_agent_positions[i1] = next_position

        next_state = np.array(current_agent_positions).reshape(-1)
        return np.array(next_state)


def next_cell(column, row, action_meaning, layout):

    if layout[row, column] == GOAL:
        return column, row

    next_row, next_column = row, column
    num_rows, num_columns = layout.shape

    if action_meaning == "up":
        next_row = max(0, next_row - 1)
    elif action_meaning == "down":
        next_row = min(num_rows - 1, next_row + 1)
    elif action_meaning == "left":
        next_column = max(0, next_column - 1)
    elif action_meaning == "right":
        next_column = min(num_columns - 1, next_column + 1)

    if layout[next_row, next_column] == BLOCKED:
        return column, row
    else:
        return next_column, next_row

# ###### #
# Reward #
# ###### #


def reward(state, joint_action, goals):
    return 100.0 if solved(state, goals) else -1.0

def solved(state, goals):

    agent_positions = parse_positions(state)
    num_agents = len(agent_positions)

    reached_goal = []
    for i1 in range(num_agents):
        agent = agent_positions[i1]
        for i2 in range(num_agents):
            is_teammate = i1 != i2
            if is_teammate:
                teammate = agent_positions[i2]
                has_collision = agent == teammate
                if has_collision:
                    return False
        agent_in_goal = agent in goals
        reached_goal.append(agent_in_goal)

    all_agents_reached_goals = all(reached_goal)
    return all_agents_reached_goals


def goals(layout):
    _goals = []
    num_rows, num_columns = layout.shape
    for x in range(num_columns):
        for y in range(num_rows):
            if layout[y, x] == GOAL:
                _goals.append((x, y))
    return _goals


def obstacles(layout):
    _obstacles = []
    num_rows, num_columns = layout.shape
    for x in range(num_columns):
        for y in range(num_rows):
            if layout[y, x] == BLOCKED:
                _obstacles.append((x, y))
    return _obstacles


def start_positions(layout):
    spawn_start = True
    _start_positions = []
    num_rows, num_columns = layout.shape
    for x in range(num_columns):
        for y in range(num_rows):
            if (spawn_start and layout[y, x] == START) or (not spawn_start and (layout[y, x] == START or layout[y, x] == OPEN)):
                _start_positions.append((x, y))
    return _start_positions


# ######### #
# Rendering #
# ######### #

def print_state(state, layout):
    agent_positions = parse_positions(state)
    num_rows, num_columns = layout.shape
    for y in range(num_rows):
        print(" ", end="")
        for x in range(num_columns):
            position = (x, y)
            if position in agent_positions:
                i = agent_positions.index(position)
                print(f"{i} ", end="")
            else:
                tag = TAGS[layout[y, x]]
                if tag == "S" or tag == "O":
                    tag = " "
                elif tag == "X":
                    if y == 0 or y == num_rows - 1:
                        tag = "-"
                    elif x == 0 or x == num_columns - 1:
                        tag = "|"
                print(f"{tag} ", end="")
        print()


def print_layout(layout):
    num_rows, num_columns = layout.shape
    for y in range(num_rows):
        print("|", end="")
        for x in range(num_columns):
            print(f"{TAGS[layout[y, x]]}|", end="")
        print()
    print()

