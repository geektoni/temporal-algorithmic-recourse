from src.experiments.synthetic import LinearTemporalSCM, NonLinearTemporalSCM, LearnedTemporalSCM
from src.models.MLP import MLP1, LogisticRegression, MLPTrainer
from src.baselines.causal_recourse import causal_recourse, DifferentiableRecourse
from src.utils import apply_solution

from sklearn.model_selection import train_test_split

import argparse

import torch
import numpy as np
import os
import sys
import pandas as pd

from tqdm import tqdm

from mpi4py import MPI
import dill

# Available synthetic experiments
def get_available_experiments(scm):
    return {
        "full": [
            (True, True, scm, 3, "CFR (robust)"),
            (True, False, scm, 3, "CFR"),
            (False, True, scm, 8, "SPR (robust)"),
            (False, False, scm, 3, "SPR"),
            (True, True, None, 3, "IMF (robust)"),
            (True, False, None, 3, "IMF")
                ],
        "simple": [
            (True, False, scm, 3, "CFR"),
            (False, False, scm, 3, "SPR"),
            (True, False, None, 3, "IMF")
        ],
        "only-robust": [
            (True, True, scm, 3, "CFR (robust)"),
            (False, True, scm, 3, "SPR (robust)"),
            (True, True, None, 3, "IMF (robust)"),
        ],
        "only-robust-2": [
            (True, True, scm, 5, "CFR (robust)"),
            (False, True, scm, 5, "SPR (robust)"),
            (True, True, None, 5, "IMF (robust)"),
        ],
        "only-interventional": [
            (False, False, scm, 3, "SPR"),
        ]

    }

trend_parameters = {
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--scm", default="linear", type=str, choices=["linear", "non-linear"], help="SCM's type for the experiment")
    parser.add_argument("--classifier", default="dnn", type=str, choices=["linear", "dnn"], help="Type of black-box model")
    parser.add_argument("--trend", default="linear", type=str, choices=["linear", "seasonal", "linear+seasonal"], help="SCM's type for the experiment")
    parser.add_argument("--experiment", default="simple", type=str, choices=["simple", "full", "only-robust", "only-robust-2", "only-interventional"], help="Name of the experiment")
    parser.add_argument("--alpha", default=1.0, type=float, choices=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0], help="Strength of the trend component")
    parser.add_argument("--runs", default=10, type=int, help="Number of run to compute the average and std")
    parser.add_argument("--timesteps", default=100, type=int, help="How many timesteps to consider for the experiments")
    parser.add_argument("--n-of-individuals", default=250, type=int, help="How many individuals with negative classification to use")
    parser.add_argument("--skip-ours", default=False, action="store_true", help="Skip computation of robust recourse over time (testing only)")
    parser.add_argument("--learned", default="ground_truth", type=str, choices=["ground_truth", "linear", "dnn", "cvae"], help="Type of learned structural equations.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print many diagnostic messages (testing only)")
    parser.add_argument("--mc-samples", default=20, type=int, help="How many sample to use for the Monte Carlo estimation of E[h(x)] (interventional recourse)")
    parser.add_argument("--output", default=".", type=str, help="Location where to save the result files.")
    args = parser.parse_args()

    # Avoid serialization errors with lambda
    MPI.pickle.__init__(dill.dumps, dill.loads)

    # OPENMPI params
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()  # Get the rank of the process
    size = comm.Get_size()  # Get the number of processes

    RUNS = args.runs
    NEGATIVE_CLASSIFIED = args.n_of_individuals
    STEP = 5 # Only needed by the temporal aware solution

    # Set the seed
    torch.manual_seed(52)
    np.random.seed(52)

    # Get trend parameters
    beta_linear, beta_seasonal = trend_parameters.get(args.scm).get(args.trend)

    # Simple SCM
    if args.scm == "linear":
        scm_ground = LinearTemporalSCM(args.alpha, beta_linear, beta_seasonal, seed=2024+mpi_rank)
    elif args.scm == "non-linear":
        scm_ground = NonLinearTemporalSCM(args.alpha, beta_linear, beta_seasonal, seed=2024+mpi_rank)

    # Where to start sampling
    initial_T = 0
    max_T = args.timesteps + 1

    # Generate a simple dataset for this example
    # We pick the data at time t=0 to train the model
    # We then send the data to each process
    if mpi_rank == 0:
        X_original, y_original, _ = scm_ground.generate(10000, max_T)
        for i in range(1,size):
            comm.send((X_original, y_original), dest=i)
    else:
        X_original, y_original = comm.recv(source=0)
    
    # Get data for training
    X = X_original[0]
    y = y_original[0]
        
    recourse_config = {
            "lr": 0.5,
            "lambd_init": 0.02,
            "decay_rate": 0.0001,
            "inner_iters": 10,
            "outer_iters": 30,
            "n_of_samples": args.mc_samples,
            "early_stopping": True
        }

    # Define some constraints for the recourse method
    constraints = {
        "actionable": [0,1,2],
        "increasing": [],
        "decreasing": [],
        "limits": torch.Tensor([[-100, 100], [-100, 100], [-100, 100]])
    }

    # Results of all the experiments
    full_experimental_results = []
    avg_experimental_results = []
    actions_experimental_results = []

    # Learn an SCM if needed
    if args.learned != "ground_truth":
        scm  = LearnedTemporalSCM(linear=args.learned == "linear")
        scm.fit_eqs(
            X_original[:50, :, :].copy(),
            output_name=f"{args.scm}_{args.alpha}_{args.trend}",
            path="./learned_scms/synthetic"
        )
        scm.load(
            output_name=f"{args.scm}_{args.alpha}_{args.trend}",
            path="./learned_scms/synthetic"
        )

        # Bump the initial T, since we start learning from here
        initial_T = 50
    else:
        scm = scm_ground

    # Multiple runs for the same seed
    for run_id in tqdm(range(RUNS), disable=mpi_rank != 0):
        
        # Create indeces
        indeces = np.array(np.arange(10000))
        train_indeces, test_indeces = train_test_split(indeces, test_size=0.2, stratify=y, random_state=run_id)

        # Split the training set in test and train
        X_train, X_test, y_train, y_test = X[train_indeces, :], X[test_indeces, :], y[train_indeces], y[test_indeces]

        # Create a model and train it
        trainer = MLPTrainer(print_freq=1, verbose=args.verbose)
        if args.classifier == "dnn":
            model = MLP1(3, hidden_size=10)
        else:
            model = LogisticRegression(3)
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

        if mpi_rank == 0 and not args.skip_ours:

            # Simulate the world evolution from the initial_t we have
            _, _, _, X_new_data = scm.generate(
                len(test_index_negatively_classified),
                max_T,
                past=torch.tensor(X_original[:initial_T+1, test_index_negatively_classified, :]),
                sample_size=20,
                return_sample=True
            )

            # Determine the portion of the loop this process will handle
            # Divide the data evenly across the available processes
            chunk_size = (max_T-initial_T) // size

            # Scatter iterations across processes
            for i in range(1, size):
                start = i * chunk_size
                end = start + chunk_size
                if i == size-1 and end < (max_T-initial_T):
                    end = (max_T-initial_T)
                comm.send((initial_T+start, initial_T+end, X_new_data), dest=i)

            start = initial_T
            end = initial_T+chunk_size

            # Simulate some data for temporal recourse
            full_cost = []
            full_avg = []
            full_actions = []
            full_validity = []
            for sol_idx, t in enumerate(range(start, end, STEP)):

                # Apply the causal recourse method
                actions, validity, costs, cfs, interv_mask = causal_recourse(
                    X_new_data[:, :t+1, :, :],
                    recourse_method,
                    constraints,
                    when_to_apply=t,
                    scm=scm,
                    counterfactual=False,
                    verbose=args.verbose,
                    time_robust=True
                )

                avg_recourse, full_recourse = apply_solution(t, end, actions, model, scm_ground, X_original, test_index_negatively_classified, validity)

                full_cost.append(np.mean(costs[full_recourse[0]]) if np.sum(full_recourse[0]) > 0 else 0)
                full_validity.append(full_recourse[0])

                full_avg.append(avg_recourse[0])
                full_actions.append(actions[0, :, :])
        
            # Gather results from other processes
            for i in range(1, size):
                full_cost_worker, full_avg_worker, full_actions_worker = comm.recv(source=i)
                full_cost += full_cost_worker
                full_avg += full_avg_worker
                full_actions += full_actions_worker
        
        elif mpi_rank != 0 and not args.skip_ours:
            assert False
            
            start, end, X_new_data = comm.recv(source=0)

            # Simulate some data for temporal recourse
            full_cost = []
            full_avg = []
            full_actions = []
            for sol_idx, t in enumerate(range(start, end, STEP)):

                # Apply the causal recourse method
                actions, validity, costs, cfs, interv_mask = causal_recourse(
                    X_new_data[:, :t+1, :, :],
                    recourse_method,
                    constraints,
                    when_to_apply=t,
                    scm=scm,
                    counterfactual=False,
                    verbose=False,
                    time_robust=True
                )

                avg_recourse, full_recourse = apply_solution(t, end, actions, model, scm_ground, X_original, test_index_negatively_classified, validity)

                full_cost.append(np.mean(costs[validity]) if np.sum(validity) > 0 else 0)
                full_avg.append(avg_recourse[0])
                full_actions.append(actions[0, :, :])
            
            # Send everything to the main process
            comm.send((
                full_cost,
                full_avg,
                full_actions
            ), dest=0)

        if mpi_rank == 0 and not args.skip_ours:

            # Iterate over the results and add them to the file
            for t in range(len(full_cost)):
                costs = full_cost[t]
                avg_experimental_results.append(
                    [run_id, "robust_time", STEP*t, full_avg[t], costs, args.alpha]
                )
            
            for t in range(len(full_actions)):
                actions_current = full_actions[t]
                validity_t = full_validity[t]
                for user_id, (action, validity_single) in enumerate(zip(actions_current, validity_t)):
                    actions_experimental_results.append(
                        [run_id, "robust_time", STEP*t, user_id, action, validity_single]
                    )

        if mpi_rank == 0:

            experiment_mapping = get_available_experiments(scm)
            assert args.experiment in experiment_mapping, f"The requested experiment ({args.experiment}) cannot be found!"
            experiment_mapping = experiment_mapping.get(
                args.experiment
            )

            # For each configuration run the experiments
            for counterfactual, robust, scm_experiment, epsilon, method_name in experiment_mapping:                
                
                # Parameters for the find recourse function
                find_recourse_params = {
                    "counterfactual": counterfactual,
                    "robust": robust,
                    "epsilon": epsilon,
                    "scm": scm_experiment
                }

                # Apply the causal recourse method
                actions, validity, costs, cfs, interv_mask = causal_recourse(
                    torch.tensor(X_original[:initial_T+1, test_index_negatively_classified, :]),
                    recourse_method,
                    constraints,
                    when_to_apply=initial_T,
                    verbose=args.verbose,
                    **find_recourse_params
                )


                # Compute the actual counterfactual distribution and recourse information
                # Compute also the average costs
                avg_recourse, full_recourse = apply_solution(initial_T, max_T, actions, model, scm_ground, X_original, test_index_negatively_classified, validity)
                avg_cost = np.mean(costs[full_recourse[0]]) if np.sum(full_recourse[0]) > 0 else 0

                for timestep, validity_t in enumerate(full_recourse):
                    for user_id, (action, validity_single) in enumerate(zip(actions[0, :, :], validity_t)):
                        actions_experimental_results.append(
                            [run_id, method_name, timestep, user_id, action, validity_single]
                        )

                # Add the average results
                for t, recourse in enumerate(avg_recourse):
                    avg_experimental_results.append(
                        [run_id, method_name, t, recourse, avg_cost, args.alpha]
                    )
            
            sys.stderr.flush()
            sys.stdout.flush()

    if mpi_rank == 0:

        # Name of the results file
        filename = f"{args.experiment}_{args.classifier}_{args.scm}_{args.trend}_{args.timesteps}_{args.alpha}_{args.runs}_{args.learned}_{args.n_of_individuals}_{args.mc_samples}"

        # Save the results for average cost and average recourse
        avg_experimental_results = pd.DataFrame(
            avg_experimental_results, columns=["run_id", "type", "timestep", "recourse", "cost", "alpha"]
        )
        avg_experimental_results.to_csv(
            os.path.join(
                args.output,
                f"{filename}_results.csv"
            ),
            index=None
        )

        # Save the suggested recourse actions
        actions_experimental_results = pd.DataFrame(
            actions_experimental_results, columns=["run_id", "type", "timestep", "user_id", "actions", "validity"]
        )
        actions_experimental_results.to_pickle(
            os.path.join(
                args.output,
                f"{filename}_actions_results.pkl"
            )
        )