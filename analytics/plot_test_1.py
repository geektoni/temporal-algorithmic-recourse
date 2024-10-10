import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import os

sns.set(font_scale=1.3,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath} \usepackage{bm}',
        "font.family": "serif",
    })

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+", type=str, help="CSV containing the results")
    args = parser.parse_args()

    all_dfs = []

    for f in args.file:

        print(f)

        data = pd.read_csv(f)
        
        classifier, scm, trend, alpha, runs, n_of_individuals, mc_samples = os.path.basename(f).split("_")[0:7]

        data["scm"] = "Non-Linear ANM" if scm == "non-linear" else "Linear ANM"
        data["classifier"] = "DNN" if classifier == "dnn" else "Logistic"

        data.rename(
            columns={"variance": r"$\sigma_{\mathbf{U}}$"},
            inplace=True
        )

        all_dfs.append(data)
    
    data = pd.concat(all_dfs)

    # Filter by data
    data = data[data.timestep.isin([50])]
    data = data[data[r"$\sigma_{\mathbf{U}}$"] != 0.1]

    # Pick only valid recourse
    valid_recourse = data

    fig, ax = plt.subplots(1,1, figsize=(5,3))

    g = sns.catplot(
        valid_recourse, x="scm", y="recourse", hue=r"$\sigma_{\mathbf{U}}$",
        errorbar="sd", #capsize=.4,
        col="classifier",
        kind="violin",
        #ax=ax,
        palette = sns.color_palette("mako").as_hex(),
        legend=False,
        height=2.5,
        aspect=5/3
    )

    ax = g.axes[0]
    fig = g.figure

    ax[0].set_ylabel(
        r"\% valid recourse"
    )
    for i in range(len(ax)):
        ax[i].set_title(
            ''
        )
        ax[i].set_xlabel(
            ""#r"Time (t)"
        )
        ax[i].grid(axis='y')
    ax[0].set_ylim((0.55, 1.05))

    fig.tight_layout()
    fig.savefig(f"validity_uncertainty.pdf", format="pdf", bbox_inches='tight')

    exit()

    g = sns.catplot(
        valid_recourse, x="scm", y="cost", hue=r"$\sigma_{\mathbf{U}}$",
        errorbar="sd", capsize=.4,
        col="classifier",
        kind="bar",
        #ax=ax,
        palette = sns.color_palette("mako").as_hex(),
        legend=False,
        height=2.5,
        aspect=5/3
    )

    ax = g.axes[0]
    fig = g.figure

    ax[0].set_ylabel(
        r"$|| \bm{\theta} ||$"
    )
    for i in range(len(ax)):
        ax[i].set_xlabel(
            ""#r"Time (t)"
        )
        ax[i].grid(axis='y')

    fig.tight_layout()
    fig.savefig("costs_variance.pdf", dpi=400, format="pdf")
