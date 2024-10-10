from src.experiments.synthetic import LinearTemporalSCM, NonLinearTemporalSCM, LearnedTemporalSCM
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
    parser.add_argument("--env", default="linear", type=str, choices=["linear", "non-linear"], help="Environment we would like to train")
    parser.add_argument("--classifier", default="linear", type=str, choices=["linear", "dnn"], help="Type of black-box model")
    parser.add_argument("--trend", default="linear", type=str, choices=["linear", "seasonal", "linear+seasonal"], help="SCM's type for the experiment")
    parser.add_argument("--alpha", default=1.0, type=float, choices=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0], help="Strength of the trend component")
    parser.add_argument("--retrain", default=False, action="store_true", help="Force retraining of the structural equations")
    args = parser.parse_args()

    print(f"[*] Training SCM structural equations: env={args.env}, alpha={args.alpha}, trend={args.trend}, model={args.classifier}")

    trend_parameters ={
        "linear": {
            "linear": (1,0),
            "seasonal": (0,1.5),
            "linear+seasonal": (1,1.5)
        },
        "non-linear": {
            "linear": (2,0),
            "seasonal": (0,5),
            "linear+seasonal": (2,5)
        }
    }
    linear_trend, seasonal_trend = trend_parameters.get(args.env).get(args.trend)

    # Load the semi-synthetic SCM and train some data
    if args.env == "linear":
        scm = LinearTemporalSCM(args.alpha, linear_trend, seasonal_trend, seed=2030)
    else:
        scm = NonLinearTemporalSCM(args.alpha, linear_trend, seasonal_trend, seed=2030)
    
    # Generate data to train the models
    X, _, _ = scm.generate(2000, 50)
    T, N, D = X.shape

    # Load the correct class learning SCMs
    lrn = LearnedTemporalSCM(linear=args.classifier == "linear")
    
    # Fit the structural equations
    lrn.fit_eqs(
        X[:50, :, :].copy(),
        output_name=f"{args.env}_{args.alpha}_{args.trend}",
        path=f"./learned_scms/synthetic/",
        force_train=args.retrain
    )

    # Re-load the learned SCM equations
    lrn.load(output_name=f"{args.env}_{args.alpha}_{args.trend}", path=f"./learned_scms/synthetic/")

    # Generate new data as testing
    pred, _, _ = lrn.generate(
        2000, 100, past=torch.Tensor(X[:50, :, :].copy())
    )

    import matplotlib.pyplot as plt

    std = np.std(pred, axis=1)
    pred = np.mean(pred, axis=1)

    X = np.mean(X, axis=1)

    for i in range(len(lrn.f)):
        plt.plot(pred[:, i], label=f"x{i}", linewidth=3)
        plt.plot(X[:, i], label=f"x{i} (true)", linewidth=3)
    plt.title(f"env={args.env}, alpha={args.alpha}, trend={args.trend}, model={args.classifier}")
    plt.legend()
    plt.show()
   