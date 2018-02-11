from cycler import cycler

import math
import matplotlib.pyplot as plt
import matplotlib        as mpl
import pandas            as pd
import numpy              as np

from scipy.special     import logit
from common.seasonal   import decompose
from common.timeseries import fix_inf

mpl.style.use('default')
font = {'family': 'IPAexGothic', 'weight': 'bold', 'size': 18}
mpl.rc('font' , **font)
plt.rc('lines', linewidth=2)
plt.rc('axes' , prop_cycle=(cycler('color'    , ['g' , 'b', 'r' , 'c' , 'k']) +
                            cycler('linestyle', ['--', '-', '-.', '-.', '-'])))

label_size    = 13
title_size    = 15
tick_size     = 11


def plt_configuration(plt, title=None, x_label=None, y_label=None, save_to=None, shade=None):
    if x_label is not None:
        plt.xlabel(x_label, fontsize=label_size)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=label_size)
    if title is not None:
        plt.title(title, fontsize=title_size)
    if shade is not None:
        plt.axvspan(shade[0], shade[1], facecolor='b', alpha=0.05)
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight')


def ax_configuration(ax, title=None, x_label=None, y_label=None, save_to=None):
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=label_size)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=label_size)
    if title is not None:
        ax.set_title(title, x=0.5, y=-0.27, fontsize=title_size)


def line_plot_component(ax, data, label, start_year=None):
    x = range(0, len(data[0]))
    for idx, (series, label) in enumerate(zip(data, label)):
        ax.plot(x, series, label=label)
    ax.legend(loc=2, fontsize = label_size)
    ax.tick_params(labelsize=tick_size)
    ax.yaxis.get_major_formatter().set_powerlimits((-2, 2))
    ax.yaxis.get_offset_text().set_fontsize(tick_size)
    year_coor = range(0, len(x) + 1, 52)
    year_label = range(start_year, start_year + len(year_coor))
    ax.set_xticks(year_coor)
    ax.set_xticklabels(year_label)
    ax.grid(axis='x')


def multi_plot(data, start_year=None, global_title=None,
               x_label=None, y_label=None,
               save_to=None, show=True, size=3,
               grid=None):

    ncol, nrow = grid
    fig,axs = plt.subplots(ncol, nrow, figsize=(nrow*(size+5), ncol*size))
    axs = axs.flat

    for (plot_data, labels, start_year), ax in zip(data, axs):
        line_plot_component(ax, plot_data, labels, start_year)
        ax_configuration(ax, x_label, y_label, save_to)

    fig.tight_layout()
    if global_title is not None:
        fig.suptitle(global_title, x=0.5, y=0.1)
    plt_configuration(plt, None, x_label, y_label, save_to)
    if show is True:
        plt.show()


def plot_results(pp_prediction, gft_prediction, disease, disease_no, lag):
    dir = 'data/{}/'.format(lag)
    y_train = pd.read_csv(dir + 'D{}_y_train.csv'.format(disease_no), index_col=0)['infection-rate'].values
    y_test  = pd.read_csv(dir + 'D{}_y_test.csv'.format(disease_no), index_col=0)['infection-rate'].values
    y = np.append(y_train, y_test)
    y_trend, _, y_irregular = decompose(fix_inf(logit(y)), 52)

    pp_trend, pp_irregular, pp_prediction = pp_prediction
    multi_plot([([y[52:], pp_prediction, gft_prediction[52:]],
                ['Real epidemic', 'Proposed prediction', 'GFT prediction'], 2008),
                ([y_trend, pp_trend],
                 ['Real trend', 'Proposed prediction'], 2008),
                ([y_irregular, pp_irregular],
                 ['Real irregular', 'Proposed prediction'], 2008)],
               grid=(3,1), global_title=disease)