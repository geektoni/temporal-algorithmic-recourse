import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

import argparse
import os

from src.utils import PALETTE

sns.set(font_scale=1.5,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath} \usepackage{bm}',
        "font.family": "serif",
    })

technique_names = {
    "CFR": r"\texttt{CAR}",
    "SPR": r"\texttt{SAR}",
    "IMF": r"\texttt{IMF}",
    "robust_time": r"\texttt{T-SAR}"
}

groups = {
    "Loan": [[5], [6], [5,6]],
    "COMPAS": [[3]],
    "Adult": [[4], [5], [4,5]],
    "Linear ANM": [[0],[1],[2], [0,1], [0,2], [1,2], [0,1,2]],
    "Non-Linear ANM": [[0],[1],[2], [0,1], [0,2], [1,2], [0,1,2]]
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+", type=str, help="CSV containing the results")
    parser.add_argument("--synthetic", default=False, action="store_true", help="Plot data for the synthetic experiments.")
    args = parser.parse_args()

    # Where to save all the data
    all_data = []
    all_dfs = []

    for f in args.file:

        data = pickle.load(open(f, "rb"))

        experiment, classifier, scm, trend, _, alpha, runs, mc_samples = os.path.basename(f).split("_")[0:8]
        
        if scm == "non-linear":
            data["scm"] = "Non-Linear ANM"
        elif scm == "linear":
            data["scm"] = "Linear ANM"
        elif scm == "adult":
            data["scm"] = r'\texttt{Adult}'
        elif scm == "compas":
            data["scm"] = r'\texttt{COMPAS}'
        else:
            data["scm"] = r'\texttt{Loan}'

        data["classifier"] = "DNN" if classifier == "dnn" else "Logistic"

        if "alpha" not in data.columns:
            data["alpha"] = float(alpha)

        data.rename(
            columns={"alpha": r"$\alpha$"},
            inplace=True
        )

        if trend == "seasonal":
            trend = "Seasonal"
        elif trend == "linear+seasonal":
            trend = "Linear+Seasonal"
        else:
            trend = "Linear"

        data["trend"] = trend
        data = data[data.type != "IMF (robust)"]

        data["type"] = data["type"].apply(lambda x: technique_names.get(x.replace(" (robust)", "")) if x != "robust_time" else r"\texttt{T-SAR}")
        data["type"] = data["type"].apply(lambda x: x+r" ($\epsilon = 0.5$)" if (x != r"\texttt{T-SAR}" and experiment == "only-robust-2")  else x)
        data["type"] = data["type"].apply(lambda x: x+r" ($\epsilon = 0.05$)" if (x != r"\texttt{T-SAR}" and experiment == "only-robust")  else x)

        all_dfs.append(data)

    data_original = pd.concat(all_dfs)

    # Available combinations
    comb = {
        4: [],
        5: [],
        (4,5): []
    }

    for t in [0, 10, 20,30,40,50] if not args.synthetic else [0, 20, 40, 60, 80, 100]:

        for scm_original in data_original.scm.unique():

            for trend_original in data_original.trend.unique():

                data = data_original[(data_original.timestep == t) & (data_original.scm == scm_original) & (data_original.trend == trend_original)]

                # Pick only the top-3 methods
                mean_of_means = data.groupby(["run_id", "type"])["validity"].mean()
                top_3_methods = mean_of_means.groupby("type").mean().sort_values(ascending=False)#.head(3)
                top_3_methods = top_3_methods.index
                filtered_df = data[data.type.isin(top_3_methods)]

                # Consider only the valid elements
                filtered_df = filtered_df[filtered_df.validity]

                # COmpute the sparsity    
                filtered_df["sparsity"] = filtered_df["actions"].apply(lambda x : np.sum(x != 0))

                given_data = filtered_df.groupby(["type", "run_id"])["sparsity"].mean()
                for (method, run_id), value in zip(given_data.index, given_data.tolist()):
                    all_data.append(
                        [t, run_id, scm_original, trend_original, method, value]
                    )

    all_data = pd.DataFrame(all_data, columns=["t", "run_id", "scm", "trend", "method", r"$|\mathcal{I}|$"])

    g = sns.catplot(
        data=all_data, x="t", y=r"$|\mathcal{I}|$", hue="method",
        col="scm" if not args.synthetic else "trend",
        kind="bar", capsize=.4,
        height=2.5,
        col_order=[r"\texttt{Adult}", r"\texttt{COMPAS}", r"\texttt{Loan}"] if not args.synthetic else ["Linear", "Seasonal", "Linear+Seasonal"],
        aspect=2,
        legend=False,
        palette=PALETTE
    )

    axs = g.axes[0]
    fig = g.figure

    for idx_ax, ax in enumerate(axs):
        
        experiment_name = ax.get_title().split("=")[1].strip()

        ax.set_title(
            experiment_name if not args.synthetic or scm == "linear" else ""
        )

        ax.set_ylabel(
            r"$|\mathcal{I}|$"
        ) 
        ax.set_xlabel(
            ""#r"Time (t)"
        )
        ax.grid(axis='y')

    plt.tight_layout()
    plt.savefig(f"sparsity.pdf", format="pdf", bbox_inches='tight')
