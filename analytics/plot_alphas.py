import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import os

sns.set(font_scale=1.8,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath} \usepackage{bm}',
        "font.family": "serif",
    })

technique_names = {
    "CFR": r"\texttt{CAR}",
    "SPR": r"\texttt{SAR}",
    "IMF": r"\texttt{IMF}"
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+", type=str, help="CSV containing the results")
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

        all_dfs.append(data)
    
    data = pd.concat(all_dfs)

    data["type"] = data["type"].apply(lambda x: x.replace(" (robust)", ""))

    # Filter by data
    data = data[data.timestep.isin([49])]
    #data = data[data.type != "IMF"]
    data = data[data.classifier == "DNN"]

    print(sns.color_palette("rocket").as_hex()[0:5])

    data["type"] = data["type"].apply(lambda x: technique_names.get(x))
    print(data)

    fig, ax = plt.subplots(1,1, figsize=(5,3))

    g = sns.catplot(
        data, x="type", y="recourse", hue=r"$\alpha$",
        capsize=.4,
        errorbar="sd",
        col= "scm",
        kind="bar",
        col_order=["Linear ANM", "Non-Linear ANM", r'\texttt{Adult}', r'\texttt{COMPAS}', r'\texttt{Loan}'],
        palette = sns.color_palette("rocket").as_hex()[0:5],
        legend=False,
        height=3,
        aspect=1
    )

    axs = g.axes
    fig = g.figure

    for idx_ax, ax in enumerate(axs):
        ax[0].set_ylabel(
            r"\% valid recourse"
        ) 
        for i in range(len(ax)):
            
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
    fig.savefig(f"validity_alpha.pdf", format="pdf", bbox_inches='tight')

    exit()
    g = sns.catplot(
        data, x="type", y="cost", hue=r"$\alpha$",
        errorbar="sd",
        col= "scm",
        kind="violin",
        col_order=["Linear ANM", "Non-Linear ANM", "Adult", "COMPAS", "Loan"],
        palette = sns.color_palette("rocket").as_hex()[0:5],
        legend=False,
        height=3,
        aspect=5/5
    )

    axs = g.axes
    fig = g.figure
    
    for idx_ax, ax in enumerate(axs):
        ax[0].set_ylabel(
            r"$|| \bm{\theta} ||$"
        ) 
        for i in range(len(ax)):
            
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
                ""
            )
            ax[i].grid(axis='y')

    fig.tight_layout()
    fig.savefig("costs_alpha.pdf", dpi=400, format="pdf")
