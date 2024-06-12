import numpy as np
import matplotlib.pyplot as plt
from mdp import generate_ws
from utils import plotting_helper_function

if __name__ == "__main__":
    mu = 0.5  # behaviour
    g = 0.99  # discount

    lambdas = np.array([0, 0.8, 0.9, 0.95, 1.])
    pis = np.array([0., 0.1, 0.2, 0.5, 1.])

    fig = plt.figure(figsize=(22, 17))
    fig.subplots_adjust(wspace=0.25, hspace=0.3)

    for r, pi in enumerate(pis):
        for c, l in enumerate(lambdas):
            plt.subplot(len(pis), len(lambdas), r*len(lambdas) + c + 1)
            w = np.ones_like([1., 1.])
            ws = generate_ws(w, pi, mu, l, g)
            title = '$\\lambda={:1.3f}$'.format(l) if r == 0 else None
            ylabel = '$\\pi={:1.1f}$'.format(pi) if c == 0 else None
            plotting_helper_function(ws[:, 0], ws[:, 1], title, ylabel)
    plt.show()
