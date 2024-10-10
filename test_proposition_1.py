from src.experiments.synthetic import LinearTemporalSCMVariance, NonLinearTemporalSCMVariance
from src.models.MLP import MLP1, LogisticRegression, MLPTrainer
from src.baselines.causal_recourse import causal_recourse, DifferentiableRecourse
from src.utils import apply_solution

from sklearn.model_selection import train_test_split

import argparse

import torch
import numpy as np
import os
import pandas as pd

from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--scm", default="linear", type=str, choices=["linear", "non-linear"], help="SCM's type for the experiment")
    parser.add_argument("--trend", default="linear", type=str, choices=["linear", "seasonal", "linear+seasonal"], help="SCM's type for the experiment")
    parser.add_argument("--alpha", default=1.0, type=float, choices=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0], help="Strength of the trend component")
    parser.add_argument("--classifier", default="dnn", type=str, choices=["logistic", "dnn"], help="Type of black-box model")
    parser.add_argument("--runs", default=10, type=int, help="Number of run to compute the average and std")
    parser.add_argument("--n-of-individuals", default=250, type=int, help="How many individuals with negative classification to use")
    parser.add_argument("--skip-ours", default=False, action="store_true", help="Skip computation of robust recourse over time (testing only)")
    parser.add_argument("--mc-samples", default=20, type=int, help="How many sample to use for the Monte Carlo estimation of E[h(x)] (interventional recourse)")
    parser.add_argument("--output", default=".", type=str, help="Location where to save the result files.")
    args = parser.parse_args()

    RUNS = args.runs
    NEGATIVE_CLASSIFIED = args.n_of_individuals

    # Set the seed
    torch.manual_seed(52)
    np.random.seed(52)

    # Results of all the experiments
    full_experimental_results = []
    avg_experimental_results = []

    # Multiple runs for the same seed
    for run_id in tqdm(range(RUNS)):

        # Generate initial data to prime the process
        scm = LinearTemporalSCMVariance() if args.scm == "linear" else NonLinearTemporalSCMVariance()
        X_base, _, _ = scm.generate(10000, 1)

        for variance in tqdm([0.0, 0.1, 0.3, 0.5, 0.7, 1.0], desc=f"Variance {run_id}: "):

            # Simple SCM
            if args.scm == "linear":
                scm = LinearTemporalSCMVariance(variance=variance)
            elif args.scm == "non-linear":
                scm = NonLinearTemporalSCMVariance(variance=variance)

            # Where to start sampling
            initial_T = 0
            max_T = 100

            # Generate a simple dataset for this example
            # We pick the data at time t=0 to train the model
            X_original, y_original, U = scm.generate(10000, max_T, past=torch.Tensor(X_base))
            X = X_original[0]
            y = y_original[0]
                
            recourse_config = {
                    "lr": 0.2,
                    "lambd_init": 0.02,
                    "decay_rate": 0.0001,
                    "inner_iters": 10,
                    "outer_iters": 30,
                    "n_of_samples": args.mc_samples,
                    "early_stopping": True
                }

            # Define some constraints for the recourse method
            constraints = {
                "actionable": [0, 1, 2],
                "increasing": [],
                "decreasing": [],
                "limits": torch.Tensor([[-100, 100], [-100, 100], [-100, 100]])
            }
                
            # Create indeces
            indeces = np.array(np.arange(10000))
            train_indeces, test_indeces = train_test_split(indeces, test_size=0.2, stratify=y, random_state=run_id)

            # Split the training set in test and train
            X_train, X_test, y_train, y_test = X[train_indeces, :], X[test_indeces, :], y[train_indeces], y[test_indeces]

            # Create a model and train it
            model = MLP1(3, hidden_size=10) if args.classifier == "dnn" else LogisticRegression(3)
            trainer = MLPTrainer(print_freq=1, verbose=False)
            trainer.train(model, X_train, y_train, X_test, y_test, epochs=15)

            # Get all the instances which are negatively classified
            with torch.no_grad():
                negatively_classified = model.predict_torch(torch.FloatTensor(
                    X_original[initial_T, test_indeces, :]
                )).numpy() == 0

            # Select only those instances which are negatively classified
            from itertools import compress
            test_index_negatively_classified = list(compress(test_indeces, negatively_classified))

            # Sample only 500 negatively classified individuals
            test_index_negatively_classified = test_index_negatively_classified[0:NEGATIVE_CLASSIFIED]

            # Implement the differentiable recourse method
            recourse_method = DifferentiableRecourse(
                model,
                recourse_config
            )

            # Apply the causal recourse method
            actions, validity, costs, cfs, interv_mask = causal_recourse(
                X_original[:initial_T+1, test_index_negatively_classified, :],
                recourse_method,
                constraints,
                when_to_apply=initial_T,
                scm=scm,
                counterfactual=True,
                robust=True,
                verbose=False,
                epsilon=0.05
            )

            # Apply the solution over time
            avg_recourse, full_recourse = apply_solution(initial_T, max_T, actions, model, scm, X_original, test_index_negatively_classified, validity)
            
            # Iterate over the results and add them to the file
            for t in range(len(avg_recourse)):
                avg_experimental_results.append(
                    [run_id, args.classifier, "robust_time", t, variance,
                     avg_recourse[t], np.mean(costs[full_recourse[t]]) if np.sum(full_recourse[t]) > 0 else 0]
                )


    # Name of the results file
    filename = f"{args.classifier}_{args.scm}_{args.trend}_{args.alpha}_{args.runs}_{args.n_of_individuals}_{args.mc_samples}"

    # Save the results for average cost and average recourse
    avg_experimental_results = pd.DataFrame(
        avg_experimental_results, columns=["run_id", "classifier", "type", "timestep", "variance", "recourse", "cost"]
    )
    avg_experimental_results.to_csv(
        os.path.join(
            args.output,
            f"{filename}_avg_results_variance.csv"
        ),
        index=None
    )