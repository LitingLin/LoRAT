from typing import Sequence, Mapping, BinaryIO, Any
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from .ope_metrics import _bins_of_center_location_error, _bins_of_normalized_center_location_error, _bins_of_intersection_of_union, _bin_index_precision_score, _bin_index_normalized_precision_score


_plot_draw_style = [{'color': (1.0, 0.0, 0.0), 'line_style': '-'},
                    {'color': (0.0, 1.0, 0.0), 'line_style': '-'},
                    {'color': (0.0, 0.0, 1.0), 'line_style': '-'},
                    {'color': (0.0, 0.0, 0.0), 'line_style': '-'},
                    {'color': (1.0, 0.0, 1.0), 'line_style': '-'},
                    {'color': (0.0, 1.0, 1.0), 'line_style': '-'},
                    {'color': (0.5, 0.5, 0.5), 'line_style': '-'},
                    {'color': (136.0 / 255.0, 0.0, 21.0 / 255.0), 'line_style': '-'},
                    {'color': (1.0, 127.0 / 255.0, 39.0 / 255.0), 'line_style': '-'},
                    {'color': (0.0, 162.0 / 255.0, 232.0 / 255.0), 'line_style': '-'},
                    {'color': (0.0, 0.5, 0.0), 'line_style': '-'},
                    {'color': (1.0, 0.5, 0.2), 'line_style': '-'},
                    {'color': (0.1, 0.4, 0.0), 'line_style': '-'},
                    {'color': (0.6, 0.3, 0.9), 'line_style': '-'},
                    {'color': (0.4, 0.7, 0.1), 'line_style': '-'},
                    {'color': (0.2, 0.1, 0.7), 'line_style': '-'},
                    {'color': (0.7, 0.6, 0.2), 'line_style': '-'}]


def generate_plot(y: np.ndarray, x: np.ndarray, scores: np.ndarray,
                  tracker_name_list: Sequence[str], plot_draw_styles: Sequence[Mapping[str, Any]],
                  f: BinaryIO, plot_opts: dict):
    # Plot settings
    font_size = plot_opts.get('font_size', 12)
    font_size_axis = plot_opts.get('font_size_axis', 13)
    line_width = plot_opts.get('line_width', 2)
    font_size_legend = plot_opts.get('font_size_legend', 13)

    legend_loc = plot_opts['legend_loc']

    xlabel = plot_opts['xlabel']
    ylabel = plot_opts['ylabel']
    xlim = plot_opts['xlim']
    ylim = plot_opts['ylim']

    title = plot_opts['title']

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    fig, ax = plt.subplots()

    index_sort = (-scores).argsort()

    plotted_lines = []
    legend_text = []

    for id, id_sort in enumerate(index_sort):
        line = ax.plot(x.tolist(), y[id_sort, :].tolist(),
                       linewidth=line_width,
                       color=plot_draw_styles[index_sort.size - id - 1]['color'],
                       linestyle=plot_draw_styles[index_sort.size - id - 1]['line_style'])

        plotted_lines.append(line[0])

        disp_name = tracker_name_list[id_sort]

        legend_text.append('{} [{:.3f}]'.format(disp_name, scores[id_sort]))

    ax.legend(plotted_lines[::-1], legend_text[::-1], loc=legend_loc, fancybox=False, edgecolor='black',
              fontsize=font_size_legend, framealpha=1.0)

    ax.set(xlabel=xlabel,
           ylabel=ylabel,
           xlim=xlim, ylim=ylim,
           title=title)

    ax.grid(True, linestyle='-.')
    fig.tight_layout()

    fig.savefig(f, dpi=300, format='pdf', transparent=True)
    plt.draw()
    plt.close(fig)


def draw_success_plot(succ_curves: np.ndarray, tracker_names: Sequence[str], f: BinaryIO):
    success_plot_opts = {'legend_loc': 'lower left',
                         'xlabel': 'Overlap threshold', 'ylabel': 'Overlap Precision',
                         'xlim': (0, 1.0), 'ylim': (0, 1.0),
                         'title': 'Success plot'}
    threshold = np.linspace(0, 1, _bins_of_intersection_of_union)
    auc = np.mean(succ_curves, axis=1)
    generate_plot(succ_curves, threshold, auc, tracker_names, _plot_draw_style, f,
                  success_plot_opts)


def draw_precision_plot(prec_curves: np.ndarray, tracker_names: Sequence[str], f: BinaryIO):
    precision_plot_opts = {'legend_loc': 'lower right',
                           'xlabel': 'Location error threshold [pixels]', 'ylabel': 'Distance Precision',
                           'xlim': (0, _bins_of_center_location_error - 1), 'ylim': (0, 1.0),
                           'title': 'Precision plot'}
    threshold = np.arange(0, _bins_of_center_location_error)
    prec_scores = prec_curves[:, _bin_index_precision_score]
    generate_plot(prec_curves, threshold, prec_scores, tracker_names, _plot_draw_style, f,
                  precision_plot_opts)


def draw_normalized_precision_plot(norm_prec_curves: np.ndarray, tracker_names: Sequence[str], f: BinaryIO):
    norm_precision_plot_opts = {'legend_loc': 'lower right',
                                'xlabel': 'Location error threshold', 'ylabel': 'Distance Precision',
                                'xlim': (0, 0.5), 'ylim': (0, 1.0),
                                'title': 'Normalized Precision plot'}
    threshold = np.linspace(0, 0.5, _bins_of_normalized_center_location_error)
    norm_prec_scores = norm_prec_curves[:, _bin_index_normalized_precision_score]
    generate_plot(norm_prec_curves, threshold, norm_prec_scores, tracker_names, _plot_draw_style, f,
                  norm_precision_plot_opts)
