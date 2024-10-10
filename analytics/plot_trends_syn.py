import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import os

sns.set(font_scale=1.6,
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

from src.utils import PALETTE_SYN, PLOT_ORDERING_SYN, TABLE_ORDERING_SYN

def compute_mean_std_validity_cost(data):

    for trend in data.trend.unique():

        mean = data[(data.timestep == data.timestep.max()) & (data.trend == trend)].groupby(["scm", "type"])[["recourse", "cost"]].mean().to_dict()
        std = data[(data.timestep == data.timestep.max()) & (data.trend == trend)].groupby(["scm", "type"])[["recourse", "cost"]].std().to_dict()

        print("TREND: ", trend)
        for k in mean.get("recourse"):

            scm, method = k

            scm = "Linear" if scm == "Linear ANM" else "Non-Lin."

            validity = round(mean.get("recourse").get(k), 2)
            cost = round(mean.get("cost").get(k), 2)
            validity_std = round(std.get("recourse").get(k), 2)
            cost_std = round(std.get("cost").get(k), 2) 

            print(r"\rotatebox[origin=c]{90}{"+f"{scm}"+"}"+f" \t {method} \t ${validity:.2f} \pm \scriptstyle {validity_std:.2f}$ \t ${cost:.2f} \pm \scriptstyle {cost_std:.2f}$")
        
        print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+", type=str, help="CSV containing the results")
    parser.add_argument("--linear", default=False, action="store_true", help="Plot data for the linear classifier.")
    parser.add_argument("--no-titles", default=False, action="store_true", help="Plot data for the linear classifier.")
    args = parser.parse_args()

    all_dfs = []

    for f in args.file:

        data = pd.read_csv(f)
        
        experiment, classifier, scm, trend, _, alpha, runs, mc_samples = os.path.basename(f).split("_")[0:8]
        
        if scm == "non-linear":
            scm_name = "Non-Linear ANM"
        elif scm == "linear":
            scm_name = "Linear ANM"
        elif scm == "adult":
            scm_name = "Adult"
        elif scm == "compas":
            scm_name = "COMPAS"
        else:
            scm_name = "Loan"
        
        data["scm"] = scm_name

        data["classifier"] = "DNN" if classifier == "dnn" else "Logistic"

        if "alpha" not in data.columns:
            data["alpha"] = float(alpha)

        if trend == "seasonal":
            trend = "Seasonal"
        elif trend == "linear+seasonal":
            trend = "Linear+Seasonal"
        else:
            trend = "Linear"


        data["trend"] = f"{trend}"

        data.rename(
            columns={"alpha": r"$\alpha$"},
            inplace=True
        )

        data["type"] = data["type"].apply(lambda x: technique_names.get(x.replace(" (robust)", "")) if x != "robust_time" else r"\texttt{T-SAR}")
        data["type"] = data["type"].apply(lambda x: x+r" ($\epsilon = 5$)" if (x != r"\texttt{T-SAR}" and experiment == "only-robust-2")  else x)
        data["type"] = data["type"].apply(lambda x: x+r" ($\epsilon = 3$)" if (x != r"\texttt{T-SAR}" and experiment == "only-robust")  else x)

        all_dfs.append(data)
    
    data = pd.concat(all_dfs)

    # Filter by data
    data = data[data.timestep.isin(list(range(0,101, 5)))]
    data = data[data.classifier == "DNN"] if not args.linear else data[data.classifier == "Logistic"]

    g = sns.relplot(
        data, x="timestep", y="recourse", hue="type", style="type",
        errorbar="se",
        col= "trend",
        row= "scm",
        kind="line",
        col_order=["Linear", "Seasonal", "Linear+Seasonal"],
        hue_order=PLOT_ORDERING_SYN,
        legend=False,
        height=2.5,
        aspect=1.5,
        markers=True,
        dashes=False,
        palette=PALETTE_SYN,
        facet_kws={"margin_titles": True}
    )

    g.fig.subplots_adjust(hspace=0.05, wspace=0.1)

    #handles, labels = g.legend.legend_handles, [t.get_text() for t in g.legend.get_texts()]
    #fig_legend = plt.figure(figsize=(4, 2))  # Adjust size as necessary
    #fig_legend.legend(handles=handles, labels=labels, loc='center', frameon=False)
    #fig_legend.savefig("validity_experiment_syn_legend.pdf", format="pdf", bbox_inches='tight')

    #g._legend.remove()

    axs = g.axes
    fig = g.figure

    #sns.move_legend(axs[0][0], 'center left', bbox_to_anchor=(1, 0.5))

    g.set_titles(row_template="{row_name}")

    for idx_ax, ax in enumerate(axs):

        if args.no_titles:
            ax[0].set_ylabel(
                            ""
                        ) 
        else:
            ax[0].set_ylabel(
                r"\% valid recourse"
            ) 
        for i in range(len(ax)):

            if idx_ax > 0:
                ax[i].set_title(
                    ""
                )
            else:
                experiment_name = ax[i].get_title().split("=")[1].strip()
                ax[i].set_title(
                    experiment_name
                )

            ax[i].set_xlabel(
                ""#r"Time (t)"
            )
            ax[i].grid(axis='y')
        ax[0].set_ylim((0.0, 1.05))


    fig.tight_layout()
    fig.savefig(f"validity_experiment_syn.pdf", format="pdf", bbox_inches='tight')
    
    exit()
    
    g = sns.relplot(
        data, x="timestep", y="cost", hue="type", style="type",
        errorbar="se", #capsize=.4,
        col= "trend",
        row= "scm",
        kind="line",
        col_order=["Linear", "Seasonal", "Linear+Seasonal"],
        hue_order=PLOT_ORDERING_SYN,
        legend=False,
        height=3,
        aspect=1,
        markers=True,
        dashes=False,
        palette=PALETTE_SYN,
        facet_kws={"margin_titles": True},
    )

    axs = g.axes
    fig = g.figure

    g.set_titles(row_template="{row_name}")

    for idx_ax, ax in enumerate(axs):


        ax[0].set_ylabel(
            r"$\| \mathbf{\theta} \|$"
        ) 
        for i in range(len(ax)):

            if idx_ax > 0:
                ax[i].set_title(
                    ""
                )
            else:
                experiment_name = ax[i].get_title().split("=")[1].strip()
                ax[i].set_title(
                    experiment_name
                )

            ax[i].set_xlabel(
                ""#r"Time (t)"
            )
            ax[i].grid(axis='y')
            ax[0].set_yscale("log")
    
    fig.tight_layout()
    fig.savefig(f"cost_experiment_syn.pdf", dpi=400)
