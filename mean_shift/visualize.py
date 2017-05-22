import json
import logging
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(color_codes=True)

logging.basicConfig(level=logging.INFO)
PATH_STATS = Path("images/report")
DIR_OUTPUT = Path("report/stats")
DIR_OUTPUT.mkdir(exist_ok=True)


def read_stats():
    """Reads all json files from PATH_STATS"""
    logging.info("Reading stats")
    stats = {}
    for file in PATH_STATS.iterdir():
        if file.is_file() and 'json' in str(file):
            logging.info("Reading file {}".format(file))
            name = str(file).split('/')[-1]
            stats[name]=json.loads(file.read_text())
    return stats


def preprocess(name, stats_data):
    """
    Generates a DataFrame from the JSON data of a stats file.
    :param stats_data: The JSON loaded from a stats file.
    :param name: The name of the stats file to be added to the DataFrame.
    :return: A Pandas DataFram with peaks, time, r, c, use_5d and name as columns.
    """
    peaks, time, r, c, use_5d = [], [], [], [], []
    for imgname, data in stats_data.items():
        peaks.append(data[0])
        time.append(data[1])
        sfilename, sr, sc, suse_5d, _ = imgname[:-4].split('_')
        r.append(int(sr[1:]))
        c.append(int(sc[1:]))
        use_5d.append(suse_5d[2:] == "True")
    df = pd.DataFrame({
        "peaks": peaks,
        "time": time,
        "r": r,
        "c": c,
        "use_5d": use_5d,
        "name": name
    })
    return df


def plot_all(stats, *, save=False):
    """
    Plots stats for all passed stats files.
    :param stats:  A dict containing "name": JSON mappings for *multiple* stats files.
    :param save: Set to True to not show, but save the plots to report/stats/.
    """
    df = pd.concat([preprocess(s[0], s[1]) for s in stats.items()])
    name="stats_all"
    plots = [
        (plt_peaks(df, name), "peaks"),
        (plt_time(df, name), "time")
    ]
    for plt, suffix in plots:
        if save:
            plt.savefig(str(DIR_OUTPUT / "{}_{}.jpg".format(name, suffix)))
        else:
            sns.plt.show()


def plot(name, stats, *, save=False):
    """
    Plots stats for a single stats file.
    :param name: The name of the file. Required to label the image. 
    :param stats: A dict that containts the JSON from a *single* stats file.
    :param save: Set to True to not show, but save the plots to report/stats/.
    """
    df = preprocess(name, stats)
    plots = [
        (plt_peaks(df, name), "peaks"),
        (plt_time(df, name), "time")
    ]
    for plt, suffix in plots:
        if save:
            plt.savefig(str(DIR_OUTPUT / "{}_{}.jpg".format(name, suffix)))
        else:
            sns.plt.show()


def plt_peaks(df, name):
    """
    Plots the peaks per C for different r
    :param df: The Dataframe.
    :param name: Name for the plot.
    :return: The plot.
    """
    g = sns.factorplot(x="c", y="peaks", kind="swarm", hue="use_5d", col="r", data=df, legend_out=False)
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(name, fontsize=16)
    return g


def plt_time(df, name):
    """
    Plots the time required for different r/c combinations.
    :param df: The Dataframe.
    :param name: Name for the plot.
    :return: The plot
    """
    g = sns.factorplot(x="c", y="time", hue="r", col="use_5d", data=df, legend_out=False)
    g.despine(left=True)
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(name, fontsize=16)
    return g

if __name__ == '__main__':
    all_stats = read_stats()
    plot_all(all_stats)
    for n, s in all_stats.items():
        plot(n, s, save=True)