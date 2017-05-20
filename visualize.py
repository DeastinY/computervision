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


def read_stats():
    """Reads all json files from PATH_STATS"""
    logging.info("Reading stats")
    stats = {}
    for file in PATH_STATS.iterdir():
        if file.is_file() and 'json' in str(file):
            logging.info("Reading file {}".format(file))
            stats[str(file)]=json.loads(file.read_text())
    return stats


def plot(stats):
    """Plots the passed dict."""
    peaks, time, r, c, use_5d = [], [], [] , [], []
    for imgname, data in stats.items():
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
    })
    # Peaks per C for different r
    g = sns.factorplot(x="c", y="peaks", kind="swarm", hue="use_5d", col="r", data=df, legend_out=False)

    sns.plt.show()
    # Time for r/c
    g = sns.factorplot(x="c", y="time", hue="r", col="use_5d", data=df, legend_out=False)
    g.despine(left=True)
    sns.plt.show()


if __name__ == '__main__':
    for name, stats in read_stats().items():
        plot(stats)