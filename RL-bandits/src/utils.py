# src/utils.py

import numpy as np
import matplotlib.pyplot as plt

def smooth(array, smoothing_horizon=100., initial_value=0.):
    smoothed_array = []
    value = initial_value
    b = 1./smoothing_horizon
    m = 1.
    for x in array:
        m *= 1. - b
        lr = b/(1 - m)
        value += lr*(x - value)
        smoothed_array.append(value)
    return np.array(smoothed_array)

def plot(algs, plot_data, repetitions=30):
    algs_per_row = 4
    n_algs = len(algs)
    n_rows = (n_algs - 2)//algs_per_row + 1
    fig = plt.figure(figsize=(10, 4*n_rows))
    fig.subplots_adjust(wspace=0.3, hspace=0.35)
    clrs = ['#000000', '#00bb88', '#0033ff', '#aa3399', '#ff6600']
    lss = ['--', '-', '-', '-', '-']
    for i, p in enumerate(plot_data):
        for c in range(n_rows):
            ax = fig.add_subplot(n_rows, len(plot_data), i + 1 + c*len(plot_data))
            ax.grid(0)

            current_algs = [algs[0]] + algs[c*algs_per_row + 1:(c + 1)*algs_per_row + 1]
            for alg, clr, ls in zip(current_algs, clrs, lss):
                data = p.data[alg.name]
                m = smooth(np.mean(data, axis=0))
                s = np.std(smooth(data.T).T, axis=0)/np.sqrt(repetitions)
                if p.log_plot:
                    line = plt.semilogy(m, alpha=0.7, label=alg.name, color=clr, ls=ls, lw=3)[0]
                else:
                    line = plt.plot(m, alpha=0.7, label=alg.name, color=clr, ls=ls, lw=3)[0]
                    plt.fill_between(range(len(m)), m + s, m - s, color=line.get_color(), alpha=0.2)
            if p.opt_values is not None:
                plt.plot(p.opt_values[current_algs[0].name][0], ':', alpha=0.5, label='optimal')

            ax.set_facecolor('white')
            ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set(visible=True, color='black', lw=1)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set(visible=True, color='black', lw=1)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            data = np.array([smooth(np.mean(d, axis=0)) for d in p.data.values()])

            if p.log_plot:
                start, end = calculate_lims(data, p.log_plot)
                start = np.floor(np.log10(start))
                end = np.ceil(np.log10(end))
                ticks = [_*10**__ for _ in [1., 2., 3., 5.] for __ in [-2., -1., 0.]]
                labels = [r'${:1.2f}$'.format(_*10** __) for _ in [1, 2, 3, 5] for __ in [-2, -1, 0]]
                plt.yticks(ticks, labels)
            plt.ylim(calculate_lims(data, p.log_plot))
            plt.locator_params(axis='x', nbins=4)

            plt.title(p.title)
            if i == len(plot_data) - 1:
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

def calculate_lims(data, log_plot=False):
    y_min = np.min(data)
    y_max = np.max(data)
    diff = y_max - y_min
    if log_plot:
        y_min = 0.9*y_min
        y_max = 1.1*y_max
    else:
        y_min = y_min - 0.05*diff
        y_max = y_max + 0.05*diff
    return y_min, y_max

def argmax(array):
    return np.random.choice(np.flatnonzero(array == array.max()))
