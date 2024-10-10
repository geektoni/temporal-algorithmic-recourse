from src.experiments.real import SemiSyntheticAdultTemporal, SemiSyntheticLoanTemporal, SemiSyntheticCOMPASTemporal, LearnedAdultTemporal, LearnedLoanTemporal, LearnedCOMPASTemporal
from src.models.MLP import MLP1, LogisticRegression, MLPTrainer
from src.baselines.causal_recourse import causal_recourse, DifferentiableRecourse
from src.utils import apply_solution

from data.data_utils import process_causal_adult, process_compas_causal_data

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
            (True, False, scm, 0, "CFR"),
            (False, False, scm, 0, "SPR"),
            (True, False, None, 0, "IMF")
        ],
        "only-robust": [
            (True, True, scm, 0.05, "CFR (robust)"),
            (False, True, scm, 0.05, "SPR (robust)"),
            (True, True, None, 0.05, "IMF (robust)"),
        ],
        "only-robust-2": [
            (True, True, scm, 0.5, "CFR (robust)"),
            (False, True, scm, 0.5, "SPR (robust)"),
            (True, True, None, 0.5, "IMF (robust)"),
        ],
        "only-interventional": [
            (False, False, scm, 3, "SPR"),
        ]

    }

trend_parameters = {
    "adult": {
        "linear": (1,0),
        "seasonal": (0,1),
        "linear+seasonal": (1,1)
    },
    "compas": {
        "linear": (0.3,0),
        "seasonal": (0,1),
        "linear+seasonal": (0.3, 1)
    }, 
    "loan": {
        "linear": (0.5,0),
        "seasonal": (0,5),
        "linear+seasonal": (0.5, 5)
    }
}

# Number of individual we generate
N_INDIVIDUALS = 1000

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="adult", type=str, choices=["adult", "loan", "compas"], help="Experiment type")
    parser.add_argument("--classifier", default="linear", type=str, choices=["linear", "dnn"], help="Type of black-box model")
    parser.add_argument("--trend", default="linear", type=str, choices=["linear", "seasonal", "linear+seasonal"], help="SCM's type for the experiment")
    parser.add_argument("--experiment", default="simple", type=str, choices=["simple", "full", "only-robust", "only-robust-2", "only-interventional"], help="Name of the experiment")
    parser.add_argument("--alpha", default=1.0, type=float, choices=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0], help="Strength of the trend component")
    parser.add_argument("--runs", default=10, type=int, help="Number of run to compute the average and std")
    parser.add_argument("--timesteps", default=100, type=int, help="How many timesteps to consider for the experiments")
    parser.add_argument("--n-of-individuals", default=250, type=int, help="How many individuals with negative classification to use")
    parser.add_argument("--skip-ours", default=False, action="store_true", help="Skip computation of robust recourse over time (testing only)")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print many diagnostic messages (testing only)")
    parser.add_argument("--learned", default=False, action="store_true", help="Use a learned temporal SCM rather than the ground truth.")
    parser.add_argument("--mc-samples", default=20, type=int, help="How many sample to use for the Monte Carlo estimation of E[h(x)] (interventional recourse)")
    parser.add_argument("--start-t", default=0, type=int, help="How many sample to use for the Monte Carlo estimation of E[h(x)] (interventional recourse)")
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
    STEPS = 5 # Interval for computing the solution

    # Set the seed
    torch.manual_seed(52)
    np.random.seed(52)

    # Get trend parameters
    beta_linear, beta_seasonal = trend_parameters.get(args.env).get(args.trend)

    # Simple SCM
    if args.env == "adult":
        scm_ground = SemiSyntheticAdultTemporal(args.alpha, beta_linear, beta_seasonal, seed=2024+mpi_rank)
        scm_ground.load("data/scms/adult")
    elif args.env == "compas":
        scm_ground = SemiSyntheticCOMPASTemporal(args.alpha, beta_linear, beta_seasonal, seed=2024+mpi_rank)
        scm_ground.load("data/scms/compas")
    else:
        scm_ground = SemiSyntheticLoanTemporal(args.alpha, beta_linear, beta_seasonal, seed=2024+mpi_rank)

    # Where to start sampling
    initial_T = args.start_t
    max_T = args.timesteps + 1 # To account for the step size

    # Get data for training. The functions returns also the constrains
    # we need.
    if args.env == "adult":
        X, y, constraints = process_causal_adult()
        X = X.to_numpy()
        y = y.to_numpy()

        # Generate original data from which to train the model
        X_original, _, _ = scm_ground.generate(1000, max_T)
    elif args.env == "compas":
        X, y, constraints = process_compas_causal_data()
        X = X.to_numpy()
        y = y.to_numpy()

        # Generate original data from which to train the model
        X_original, _, _ = scm_ground.generate(1000, max_T)
    else:
        X_original, y, _ = scm_ground.generate(10000, max_T)
        X = X_original[0, :, :]
        y = y[0, :]
        
        constraints = {
            "actionable": [5,6],
            "increasing": [],
            "decreasing": [],
            "limits": torch.Tensor([[-100, 100] for _ in range(7)])
        }

    # Generate a simple dataset for this example
    # We pick the data at time t=0 to train the model
    # We then send the data to each process
    if mpi_rank == 0:
        for i in range(1,size):
            comm.send(X_original, dest=i)
    else:
        X_original = comm.recv(source=0)

    constraints['limits'] = torch.Tensor(constraints['limits'])
        
    recourse_config = {
            "lr": 0.1 if (args.env == "adult" or args.env == "compas") else 3,
            "lambd_init": 0.02,
            "decay_rate": 0.0001,
            "inner_iters": 10,
            "outer_iters": 30,
            "n_of_samples": args.mc_samples,
            "early_stopping": True
        }

    # Results of all the experiments
    full_experimental_results = []
    avg_experimental_results = []
    actions_experimental_results = []

    # Learn an SCM if needed
    if args.learned:
        
        # we use a linear approximation of the structural equations
        if args.env == "adult":   
            scm  = LearnedAdultTemporal(linear=True)
        elif args.env == "loan":
            scm  = LearnedLoanTemporal(linear=True)
        else:
            scm  = LearnedCOMPASTemporal(model_type="linear")
        
        # Load the learned structural equations
        scm.load(
            output_name=f"{args.env}_{args.alpha}_{args.trend}",
            path=f"./learned_scms/{args.env}"
        )

        # Bump the initial T, since we start learning from here
        initial_T = 50
    else:
        scm = scm_ground

    # Multiple runs for the same seed
    for run_id in tqdm(range(RUNS), disable=mpi_rank != 0):
        
        # Create indeces
        indeces = np.array(np.arange(len(X)))
        train_indeces, test_indeces = train_test_split(indeces, test_size=0.2, stratify=y, random_state=run_id)

        # Split the training set in test and train
        X_train, X_test, y_train, y_test = X[train_indeces, :], X[test_indeces, :], y[train_indeces], y[test_indeces]

        # Create a model and train it
        trainer = MLPTrainer(print_freq=1, verbose=args.verbose)
        if args.classifier == "dnn":
            model = MLP1(X.shape[1], hidden_size=20)
        else:
            model = LogisticRegression(X.shape[1])
        trainer.train(model, X_train, y_train, X_test, y_test, epochs=15)

        # Get all the instances which are negatively classified
        with torch.no_grad():
            negatively_classified = model.predict_torch(torch.FloatTensor(
                X_original[initial_T, :, :]
            )).numpy() == 0
        
        # Select only those instances which are negatively classified
        from itertools import compress
        test_index_negatively_classified = list(compress(range(10000), negatively_classified))

        # Sample only 500 negatively classified individuals
        test_index_negatively_classified = test_index_negatively_classified[0:NEGATIVE_CLASSIFIED]

        # Implement the differentiable recourse method
        recourse_method = DifferentiableRecourse(
            model,
            recourse_config
        )

        # Re implement for our solution (just for compas)
        recourse_config2 = recourse_config.copy()
        if args.env == "compas":
            recourse_config2["lr"] = 3
        if args.env == "adult":
            recourse_config2["lr"] = 3
        recourse_method2 = DifferentiableRecourse(
            model,
            recourse_config2
        )


        if mpi_rank == 0 and not args.skip_ours:

            # Simulate the world evolution from the initial_t we have
            _, _, _, X_new_data = scm.generate(
                len(test_index_negatively_classified),
                max_T,
                past=torch.tensor(X_original[:initial_T, test_index_negatively_classified, :]),
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
            for sol_idx, t in enumerate(range(start, end, STEPS)):

                # Apply the causal recourse method
                actions, validity, costs, cfs, interv_mask = causal_recourse(
                    X_new_data[:, :t+1, :, :],
                    recourse_method2,
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
                full_cost_worker, full_avg_worker, full_actions_worker,full_validity, full_validity_worker = comm.recv(source=i)
                full_cost += full_cost_worker
                full_avg += full_avg_worker
                full_actions += full_actions_worker
                full_validity += full_validity_worker
        
        elif mpi_rank != 0 and not args.skip_ours:
            
            start, end, X_new_data = comm.recv(source=0)

            # Simulate some data for temporal recourse
            full_cost = []
            full_avg = []
            full_actions = []
            full_validity = []
            for sol_idx, t in enumerate(range(start, end)):

                # Apply the causal recourse method
                actions, validity, costs, cfs, interv_mask = causal_recourse(
                    X_new_data[:, :t+1, :, :],
                    recourse_method2,
                    constraints,
                    when_to_apply=t,
                    scm=scm,
                    counterfactual=False,
                    verbose=args.verbose,
                    time_robust=True
                )

                avg_recourse, full_recourse = apply_solution(t, end, actions, model, scm_ground, X_original, test_index_negatively_classified, validity)

                full_cost.append(np.mean(costs[validity]) if np.sum(validity) > 0 else 0)
                full_avg.append(avg_recourse[0])
                full_actions.append(actions[0, :, :])
                full_validity.append(full_recourse[0])
            
            # Send everything to the main process
            comm.send((
                full_cost,
                full_avg,
                full_actions,
                full_validity
            ), dest=0)

        if mpi_rank == 0 and not args.skip_ours:

            # Iterate over the results and add them to the file
            for t in range(len(full_cost)):
                costs = full_cost[t]
                avg_experimental_results.append(
                    [run_id, "robust_time", t*STEPS, full_avg[t], costs]
                )
            
            for t in range(len(full_actions)):
                actions_current = full_actions[t]
                validity_current = full_validity[t]
                for user_id, (action, validity_single) in enumerate(zip(actions_current, validity_current)):
                    actions_experimental_results.append(
                        [run_id, "robust_time", t*STEPS, user_id, action, validity_single]
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
                    recourse_method2,
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
                        [run_id, method_name, t, recourse, avg_cost]
                    )
            
            sys.stderr.flush()
            sys.stdout.flush()

    if mpi_rank == 0:

        # Name of the results file
        filename = f"{args.experiment}_{args.classifier}_{args.env}_{args.trend}_{args.timesteps}_{args.alpha}_{args.runs}_{args.n_of_individuals}_{args.mc_samples}"

        # Save the results for average cost and average recourse
        avg_experimental_results = pd.DataFrame(
            avg_experimental_results, columns=["run_id", "type", "timestep", "recourse", "cost"]
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