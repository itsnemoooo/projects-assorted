from bandit import BernoulliBandit, NonStationaryBandit
from agent import Random, EpsilonGreedy, UCB, REINFORCE
from utils import plot
from train import train_agents
import numpy as np 

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

    train_agents(agents, number_of_arms, number_of_steps, bandit_class=NonStationaryBandit)
