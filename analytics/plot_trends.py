import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import os

from src.utils import PALETTE, PLOT_ORDERING, TABLE_ORDERING

technique_names = {
    "CFR": r"\texttt{CAR}",
    "SPR": r"\texttt{SAR}",
    "IMF": r"\texttt{IMF}",
    "robust_time": r"\texttt{T-SAR}"
}

def compute_mean_std_validity_cost(data):

    mean = data[data.timestep == 50].groupby(["scm", "type"])[["recourse", "cost"]].mean().to_dict()
    std = data[data.timestep == 50].groupby(["scm", "type"])[["recourse", "cost"]].std().to_dict()

    printing_data = {}

    for k in mean.get("recourse"):

        scm, method = k

        validity = round(mean.get("recourse").get(k), 2)
        cost = round(mean.get("cost").get(k), 2)
        validity_std = round(std.get("recourse").get(k), 2)
        cost_std = round(std.get("cost").get(k), 2) 

        if method not in printing_data:
            printing_data[method] = {}
        
        if scm not in printing_data.get(method):
            printing_data[method][scm] = f"${validity:.2f} \pm \scriptstyle {validity_std:.2f}$"

    for method in TABLE_ORDERING:
        result = f"{method} \t"
        for scm in ["Adult", "COMPAS", "Loan"]:
            if method in printing_data:
                if scm in printing_data[method]:
                    result += f"\t {printing_data[method][scm]}"
        print(result)

sns.set(font_scale=1.5,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath} \usepackage{bm}',
        "font.family": "serif",
    })

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+", type=str, help="CSV containing the results")
    parser.add_argument("--linear", default=False, action="store_true", help="Plot data for the linear classifier.")
    args = parser.parse_args()

    all_dfs = []

    for f in args.file:

        data = pd.read_csv(f)
        
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

        if experiment == "only-robust-2":
            data = data[data.type != "robust_time"]

        data["type"] = data["type"].apply(lambda x: technique_names.get(x.replace(" (robust)", "")) if x != "robust_time" else r"\texttt{T-SAR}")
        data["type"] = data["type"].apply(lambda x: x+r" ($\epsilon = 0.5$)" if (x != r"\texttt{T-SAR}" and experiment == "only-robust-2")  else x)
        data["type"] = data["type"].apply(lambda x: x+r" ($\epsilon = 0.05$)" if (x != r"\texttt{T-SAR}" and experiment == "only-robust")  else x)

        all_dfs.append(data)
    
    data = pd.concat(all_dfs)

    # Filter by data
    data = data[data.timestep.isin(list(range(0,51, 5)))]
    data = data[data.classifier == "DNN"] if not args.linear else data[data.classifier == "Logistic"]

    g = sns.relplot(
        data, x="timestep", y="recourse", hue="type",
        errorbar="se", #capsize=.4,
        col= "scm",
        #row=r"$\epsilon$",
        kind="line",
        style="type",
        markers=True,
        dashes=False,
        hue_order=PLOT_ORDERING,
        col_order=[r'\texttt{Adult}', r'\texttt{COMPAS}', r'\texttt{Loan}'],
        palette=PALETTE,
        legend=False,
        height=2.5,
        aspect=1.3,
        facet_kws={"margin_titles": True}
    )

    axs = g.axes
    fig = g.figure

    for idx_ax, ax in enumerate(axs):
        ax[0].set_ylabel(
            r"\% valid recourse"
        ) 
        for i in range(len(ax)):

           
            if idx_ax == 0: 
                experiment_name = ax[i].get_title().split("=")[1].strip()

            if idx_ax > 0:
                ax[i].set_title(
                    ""
                )
            else:
                ax[i].set_title(
                    experiment_name
                )

            ax[i].set_xlabel(
                ""#r"Time (t)"
            )
            ax[i].grid(axis='y')
        ax[0].set_ylim((0.0, 1.05))

    fig.tight_layout()
    fig.savefig(f"validity_experiment.pdf", format="pdf", bbox_inches='tight')


    compute_mean_std_validity_cost(data)

    exit()

    g = sns.relplot(
        data, x="timestep", y="cost", hue="type",
        errorbar="se", #capsize=.4,
        col= "scm",
        #row=r"$\epsilon$",
        kind="line",
        style="type",
        markers=True,
        dashes=False,
        col_order=["Adult", "COMPAS", "Loan"],
        palette=sns.color_palette("colorblind")[0:7],
        legend=False,
        height=2.5,
        aspect=1,
        facet_kws={"margin_titles": True}
    )

    axs = g.axes
    fig = g.figure

    for idx_ax, ax in enumerate(axs):
        ax[0].set_ylabel(
            r"$\mathbb{E}_{\mathbf{\theta}^*}\left[ h(\mathbf{x}^t) \geq \frac{1}{2} \right]$"
        ) 
        for i in range(len(ax)):

           
            if idx_ax == 0: 
                experiment_name = ax[i].get_title().split("=")[1].strip()

            if idx_ax > 0:
                ax[i].set_title(
                    ""
                )
            else:
                ax[i].set_title(
                    experiment_name
                )

            ax[i].set_xlabel(
                ""#r"Time (t)"
            )
            ax[i].grid(axis='y')
        ax[0].set_yscale("log")

    fig.tight_layout()
    fig.savefig("costs_experiment.pdf", dpi=400, format="pdf")
