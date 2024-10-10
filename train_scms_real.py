from src.experiments.real import LearnedAdultTemporal, SemiSyntheticCOMPASTemporal, SemiSyntheticAdultTemporal, LearnedCOMPASTemporal, LearnedLoanTemporal, SemiSyntheticLoanTemporal
import pandas as pd
import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import argparse

sns.set(font_scale=3.0,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        "font.family": "serif",
    })


if __name__ == "__main__":

    # Fix this for reproducibility purposes
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="adult", type=str, choices=["adult", "loan", "compas"], help="Environment we would like to train")
    parser.add_argument("--classifier", default="linear", type=str, choices=["linear", "dnn", "cvae"], help="Type of black-box model")
    parser.add_argument("--trend", default="linear", type=str, choices=["linear", "seasonal", "linear+seasonal"], help="SCM's type for the experiment")
    parser.add_argument("--alpha", default=1.0, type=float, choices=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0], help="Strength of the trend component")
    parser.add_argument("--retrain", default=False, action="store_true", help="Force retraining of the structural equations")
    parser.add_argument("--eval", default=False, action="store_true", help="Plot a visual evaluation of the result.")
    args = parser.parse_args()

    print(f"[*] Training SCM structural equations: env={args.env}, alpha={args.alpha}, trend={args.trend}, model={args.classifier}")

    trend_parameters = {
        "linear": (1,0),
        "seasonal": (0, 1),
        "linear+seasonal": (1, 1)
    }
    linear_trend, seasonal_trend = trend_parameters.get(args.trend)

    # Load the semi-synthetic SCM and train some data
    if args.env == "adult":
        scm = SemiSyntheticAdultTemporal(args.alpha, linear_trend, seasonal_trend, seed=2030)
        scm.load("data/scms/adult")
    elif args.env == "compas":
        scm = SemiSyntheticCOMPASTemporal(args.alpha, linear_trend, seasonal_trend, seed=2030)
        scm.load("data/scms/compas")
    else:
        scm = SemiSyntheticLoanTemporal(args.alpha, linear_trend, seasonal_trend, seed=2030)
    X, _, _ = scm.generate(2000, 50)
    T, N, D = X.shape

    # Load the correct class learning SCMs
    if args.env == "adult":
        lrn = LearnedAdultTemporal(linear=args.classifier == "linear")
    elif args.env == "compas":
        lrn = LearnedCOMPASTemporal(model_type=args.classifier)
    else:
        lrn = LearnedLoanTemporal(linear=args.classifier == "linear")
    
    # Fit the structural equations
    lrn.fit_eqs(
        X[:50, :, :].copy(),
        output_name=f"{args.env}_{args.alpha}_{args.trend}",
        path=f"./learned_scms/{args.env}",
        force_train=args.retrain
    )
    lrn.load(output_name=f"{args.env}_{args.alpha}_{args.trend}", path=f"./learned_scms/{args.env}")

    pred, _, _ = lrn.generate(
        2000, 100, past=torch.Tensor(X[:50, :, :].copy())
    )

    if args.eval:
        import matplotlib.pyplot as plt

        std = np.std(pred, axis=1)
        pred = np.mean(pred, axis=1)

        X = np.mean(X, axis=1)

        for i in range(len(lrn.f)):
            plt.plot(pred[:, i], label=f"x{i}", linewidth=3)
            plt.plot(X[:, i], label=f"x{i} (true)", linewidth=3)
        plt.legend()
        plt.show()
   