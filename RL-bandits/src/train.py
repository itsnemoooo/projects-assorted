# src/train.py

import numpy as np
import collections
from functools import partial
from bandit import BernoulliBandit, NonStationaryBandit
from agent import Random, EpsilonGreedy, UCB, REINFORCE
from utils import plot

def run_experiment(bandit_constructor, algs, repetitions, number_of_steps):
    reward_dict = {}
    regret_dict = {}
    optimal_value_dict = {}

    for alg in algs:
        reward_dict[alg.name] = np.zeros((repetitions, number_of_steps))
        regret_dict[alg.name] = np.zeros((repetitions, number_of_steps))
        optimal_value_dict[alg.name] = np.zeros((repetitions, number_of_steps))

        for _rep in range(repetitions):
            bandit = bandit_constructor()
            alg.reset()

            action = None
            reward = None
            for _step in range(number_of_steps):
                action = alg.step(action, reward)
                reward = bandit.step(action)
                regret = bandit.regret(action)
                optimal_value = bandit.optimal_value()

                reward_dict[alg.name][_rep, _step] = reward
                regret_dict[alg.name][_rep, _step] = regret
                optimal_value_dict[alg.name][_rep, _step] = optimal_value

    return reward_dict, regret_dict, optimal_value_dict

def train_agents(agents, number_of_arms, number_of_steps, repetitions=100,
                 success_reward=1., fail_reward=0., bandit_class=BernoulliBandit):
    success_probabilities = np.arange(0.3, 0.7 + 1e-6, 0.4/(number_of_arms - 1))

    bandit_constructor = partial(bandit_class, success_probabilities=success_probabilities,
                                 success_reward=success_reward, fail_reward=fail_reward)
    rewards, regrets, opt_values = run_experiment(bandit_constructor, agents, repetitions, number_of_steps)

    smoothed_rewards = {}
    for agent, rs in rewards.items():
        smoothed_rewards[agent] = np.array(rs)

    PlotData = collections.namedtuple('PlotData', ['title', 'data', 'opt_values', 'log_plot'])
    total_regrets = dict([(k, np.cumsum(v, axis=1)) for k, v in regrets.items()])
    plot_data = [
        PlotData(title='Smoothed rewards', data=smoothed_rewards, opt_values=opt_values, log_plot=False),
        PlotData(title='Current Regret', data=regrets, opt_values=None, log_plot=True),
        PlotData(title='Total Regret', data=total_regrets, opt_values=None, log_plot=False),
    ]

    plot(agents, plot_data, repetitions)

if __name__ == "__main__":
    number_of_arms = 5
    number_of_steps = 1000

    agents = [
        Random("random", number_of_arms),
        EpsilonGreedy(r"$\epsilon$-greedy with $\epsilon=0$", number_of_arms, epsilon=0.),
        EpsilonGreedy(r"$\epsilon$-greedy with $\epsilon=0.1$", number_of_arms, epsilon=0.1),
        EpsilonGreedy(r"$\epsilon$-greedy with $\epsilon_t=1/t$", number_of_arms, epsilon=lambda t: 1./t),
        EpsilonGreedy(r"$\epsilon$-greedy with $\epsilon_t=1/\sqrt{t}$", number_of_arms, epsilon=lambda t: 1./t**0.5),
        UCB("UCB", number_of_arms, bonus_multiplier=1/np.sqrt(2)),
        REINFORCE(r"REINFORCE, $\alpha=0.1$", number_of_arms, step_size=0.1, baseline=False),
        REINFORCE(r"REINFORCE with baseline, $\alpha=0.1$", number_of_arms, step_size=0.1, baseline=True),
    ]

    train_agents(agents, number_of_arms, number_of_steps)
