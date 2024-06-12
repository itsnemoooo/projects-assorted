import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np

def colorline(x, y, z):
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=plt.get_cmap('copper_r'),
                              norm=plt.Normalize(0.0, 1.0), linewidth=3)

    ax = plt.gca()
    ax.add_collection(lc)
    return lc

def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def plotting_helper_function(_x, _y, title=None, ylabel=None):
    z = np.linspace(0, 0.9, len(_x))**0.7
    colorline(_x, _y, z)
    plt.plot(0, 0, '*', color='#000000', ms=20, alpha=0.7, label='$w^*$')
    plt.plot(1, 1, '.', color='#ee0000', alpha=0.7, ms=20, label='$w_0$')
    min_y, max_y = np.min(_y), np.max(_y)
    min_x, max_x = np.min(_x), np.max(_x)
    min_y, max_y = np.min([0, min_y]), np.max([0, max_y])
    min_x, max_x = np.min([0, min_x]), np.max([0, max_x])
    range_y = max_y - min_y
    range_x = max_x - min_x
    max_range = np.max([range_y, range_x])
    plt.arrow(_x[-3], _y[-3], _x[-1] - _x[-3], _y[-1] - _y[-3], color='k',
              head_width=0.04*max_range, head_length=0.04*max_range,
              head_starts_at_zero=False)
    plt.ylim(min_y - 0.2*range_y, max_y + 0.2*range_y)
    plt.xlim(min_x - 0.2*range_x, max_x + 0.2*range_x)
    ax = plt.gca()
    ax.ticklabel_format(style='plain', useMathText=True)
    plt.legend(loc=2)
    plt.xticks(rotation=12, fontsize=10)
    plt.yticks(rotation=12, fontsize=10)
    plt.locator_params(nbins=3)
    if title is not None:
        plt.title(title, fontsize=20)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=20)
