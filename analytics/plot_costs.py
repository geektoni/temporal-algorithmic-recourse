import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

import argparse
import os
from tqdm import tqdm

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs='+', type=str, help="CSV containing the results")
    args = parser.parse_args()

    # Where to save all the data
    all_data = []
    all_dfs = []

    for f in args.file:

        data = pickle.load(open(f, "rb"))

        experiment, classifier, scm, trend, _, alpha, runs, mc_samples = os.path.basename(f).split("_")[0:8]
        
        data["scm"] = scm

        data["classifier"] = "DNN" if classifier == "dnn" else "Logistic"
        data["trend"] = trend

        data["type"] = data["type"].apply(lambda x: technique_names.get(x.replace(" (robust)", "")) if x != "robust_time" else r"\texttt{T-SAR}")
        data["type"] = data["type"].apply(lambda x: x+r" ($\epsilon = 0.5$)" if (x != r"\texttt{T-SAR}" and experiment == "only-robust-2")  else x)
        data["type"] = data["type"].apply(lambda x: x+r" ($\epsilon = 0.05$)" if (x != r"\texttt{T-SAR}" and experiment == "only-robust")  else x)

        all_dfs.append(data)

    data_original = pd.concat(all_dfs)

    for scm in tqdm(data_original.scm.unique()):
        for trend in tqdm(data_original.trend.unique()):
            for t in [5, 10, 20, 30, 40, 50]:

                data = data_original[(data_original.timestep == t) & (data_original.scm == scm) & (data_original.trend == trend)]

                # Pick only the top-3 methods
                mean_of_means = data.groupby(["run_id", "type"])["validity"].mean()
                top_3_methods = mean_of_means.groupby("type").mean().sort_values(ascending=False).head(3)
                top_3_methods = top_3_methods.index
                filtered_df = data[data.type.isin(top_3_methods)]

                # Step 1: Group by 'id' and check if all 'correct' values for each 'id' are True
                valid_ids = filtered_df.groupby(['run_id', 'user_id'])['validity'].all()
                valid_ids = valid_ids[valid_ids].index
                filtered_df = filtered_df.set_index(['run_id', 'user_id'])
                filtered_df = filtered_df[filtered_df.index.isin(valid_ids)]

                # Convert actions into their cost using l1
                filtered_df.loc[:, "actions"] = filtered_df["actions"].apply(lambda x : np.sum(np.abs(x)))

                means = filtered_df.groupby("type")["actions"].mean().tolist()
                stds = filtered_df.groupby("type")["actions"].std().tolist()

                given_data = filtered_df.groupby(["type", "run_id"])["actions"].mean()

                for (method, run_id), value in zip(given_data.index, given_data.tolist()):
                    all_data.append(
                        [t, run_id, scm, trend, method, value]
                    )

    all_data = pd.DataFrame(all_data, columns=["t", "run_id", "scm", "trend", "method", "cost"])

    g = sns.catplot(
        data=all_data,
        x="t",
        y="cost",
        col="scm",
        hue="method",
        kind="bar",
        capsize=.4,
        height=2.5,
        aspect=5/3,
        legend=False,
        palette=PALETTE,
        margin_titles=True,
    )
    g.axes[0][0].set_title("")
    g.axes[0][0].grid(axis='y')
    g.axes[0][0].set_ylabel(
            r"$\| \mathbf{\theta} \|$"
        ) 
    g.axes[0][0].set_xlabel(
        ""
    )
    plt.tight_layout()
    plt.savefig(f"cost_{trend}_{scm}.pdf", format="pdf", bbox_inches='tight')
