import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(list_of_episode_returns):
    """Plot the learning curve."""
    plt.figure(figsize=(7, 5))

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    smoothed_returns = moving_average(list_of_episode_returns, 30)
    plt.plot(smoothed_returns)

    plt.ylabel('Average episode returns')
    plt.xlabel('Number of episodes')

    ax = plt.gca()
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def apply_updates(params, updates):
    return jax.tree_map(lambda p, u: p + u, params, updates)
