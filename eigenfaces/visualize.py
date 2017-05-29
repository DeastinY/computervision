import json
import logging
from pathlib import Path
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)

logging.basicConfig(level=logging.INFO)
FILE_STATS = Path("stats.json")
DIR_OUTPUT = Path("report/stats")
DIR_OUTPUT.mkdir(exist_ok=True)


def read_stats():
    """Reads stats.json"""
    logging.info("Reading stats")
    return json.loads(FILE_STATS.read_text())


def preprocess(data):
    """
    Generates a DataFrame from the JSON data of a stats file.
    :param data: The JSON loaded from a stats file.
    :return: A Pandas DataFram with train, k and accuracy
    """
    return pd.DataFrame({
        "train": [d["train"] for d in data],
        "k": [d["k"] for d in data],
        "accuracy": [d["accuracy"] for d in data]
    })


def plt_accurracy(df):
    return sns.factorplot(x="k", y="accuracy", hue="train", data=df, legend_out=False)


def plot(stats, *, save=False):
    """
    Plots stats for a single stats file.
    :param stats: A dict that containts the JSON from the stats file.
    :param save: Set to True to not show, but save the plots to report/stats/.
    """
    df = preprocess(stats)
    plots = [
        (plt_accurracy(df), "accurracy")
    ]
    for plt, name in plots:
        if save:
            plt.savefig(str(DIR_OUTPUT / f"{name}.png"))
        else:
            sns.plt.show()


if __name__ == '__main__':
    stats = read_stats()
    plot(stats, save=False)
