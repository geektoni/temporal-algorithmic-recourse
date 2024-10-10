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

rename_groups = {
    "Loan": {
        "[5]": r"$\{income\}$",
        "[6]": r"$\{savings\}$",
        "[5, 6]": r"$\{income, savings\}$",
    }
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+", type=str, help="CSV containing the results")
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
            data["scm"] = "Adult"
        elif scm == "compas":
            data["scm"] = "COMPAS"
        else:
            data["scm"] = "Loan"

        data["classifier"] = "DNN" if classifier == "dnn" else "Logistic"

        if "alpha" not in data.columns:
            data["alpha"] = float(alpha)

        data.rename(
            columns={"alpha": r"$\alpha$"},
            inplace=True
        )

        # Skip IMF, since it acts on all features anyway
        data = data[data.type != "IMF (robust)"]

        data["type"] = data["type"].apply(lambda x: technique_names.get(x.replace(" (robust)", "")) if x != "robust_time" else r"\texttt{T-SAR}")
        data["type"] = data["type"].apply(lambda x: x+r" ($\epsilon = 0.5$)" if (x != r"\texttt{T-SAR}" and experiment == "only-robust-2")  else x)
        data["type"] = data["type"].apply(lambda x: x+r" ($\epsilon = 0.05$)" if (x != r"\texttt{T-SAR}" and experiment == "only-robust")  else x)

        all_dfs.append(data)

    data_original = pd.concat(all_dfs)

    for t in [0, 10, 30, 50]:

        for scm_original in data_original.scm.unique():

            data = data_original[(data_original.timestep == t) & (data_original.scm == scm_original)]

            # Get groups of actionable features 
            actionable_groups = groups.get(scm_original)

            # Pick only the top-3 methods
            mean_of_means = data.groupby(["run_id", "type"])["validity"].mean()
            top_3_methods = mean_of_means.groupby("type").mean().sort_values(ascending=False)#.head(3)
            top_3_methods = top_3_methods.index
            filtered_df = data[data.type.isin(top_3_methods)]

            # Consider only the valid elements
            filtered_df = filtered_df[filtered_df.validity]

            # Step 1: Group by 'id' and check if all 'correct' values for each 'id' are True
            #valid_ids = filtered_df.groupby(['run_id', 'user_id'])['validity'].all()
            #valid_ids = valid_ids[valid_ids].index
            #filtered_df = filtered_df.set_index(['run_id', 'user_id'])
            #filtered_df = filtered_df[filtered_df.index.isin(valid_ids)]

            # Assert that all actions are correct
            # Namely, we do not have to modify non-actionable features.
            all_actionable = [item for sublist in actionable_groups for item in sublist]
            def _assert(x):
                complement_set = np.array([i for i in range(len(x)) if i not in all_actionable])
                if len(complement_set) > 0:
                    assert (x[complement_set] == 0).all()
                return x
            filtered_df["actions"].apply(lambda x : _assert(x))
            

            # For each intervention set, compute the counts
            for interv_set in actionable_groups:

                def _func(x):
                    complement_set = np.array([i for i in range(len(x)) if i not in interv_set])
                    if len(complement_set) > 0:
                        return (x[interv_set] != 0).all() and (x[complement_set] == 0).all()
                    else:
                        return (x[interv_set] != 0).all()
                
                filtered_df[f"{interv_set}"] = filtered_df["actions"].apply(lambda x : _func(x))

            for interv_set in actionable_groups:
                given_data = filtered_df.groupby(["type", "run_id"])[f"{interv_set}"].mean()
                for (method, run_id), value in zip(given_data.index, given_data.tolist()):
                    all_data.append(
                        [t, run_id, method, value, f"{interv_set}"]
                    )

    all_data = pd.DataFrame(all_data, columns=["t", "run_id", "method", "cost", r"$\mathcal{I}$"])

    all_data[r"$\mathcal{I}$"] = all_data[r"$\mathcal{I}$"].apply(
        lambda x: rename_groups.get("Loan", {}).get(str(x)) if str(x) in rename_groups.get("Loan", {}) else x
    )

    g = sns.catplot(
        data=all_data, x="t", y="cost", hue="method",
        col=r"$\mathcal{I}$",
        kind="bar", capsize=.4,
        height=2.5,
        aspect=1.3,
        col_wrap=3,
        legend=False,
        palette=PALETTE
    )

    axs = g.axes
    fig = g.figure

    for idx_ax, ax in enumerate(axs):
        ax.set_ylabel(
            r"\% fraction"
        ) 
        ax.set_xlabel(
            ""#r"Time (t)"
        )
        ax.grid(axis='y')
        ax.set_ylim((0.0, 1.05))

    plt.tight_layout()
    plt.savefig(f"{scm}_{trend}_actions.pdf", format="pdf", bbox_inches='tight')
