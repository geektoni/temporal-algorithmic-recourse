"""
This files implements the proposed methods to generate robust recourse for linear and differentiable classifiers.
"""

import numpy as np
import cvxpy as cp
import torch

from tqdm import tqdm

def build_feasibility_sets(X, actionable, constraints):
    """
    Transforms the actionability constrains into a L_inf set into which to project.

    Inputs:     X: np.array (N, D)
                actionable: list of int, indices of the features which are actionable
                constraints: dict with the following keys
                    increasing: list of int, indices of the features which can only increase
                    decreasing: list of int, indices of the features which can only decrease
                    feature_limits: np.array (D, 2), min and max values for each of the D features

    Returns:    bounds: torch.Tensor (N, D, 2): min and max deltas for each of the N data points and D features
    """
    bounds = (torch.Tensor([[[-1, 1]]]).repeat(X.shape[0], X.shape[-1], 1) * 1e10).numpy()
    for i in range(X.shape[1]):
        if i in actionable:
            if i in constraints['increasing']:
                bounds[:, i, 0] = 0
            elif i in constraints['decreasing']:
                bounds[:, i, 1] = 0
        else:
            bounds[:, i, 0] = 0
            bounds[:, i, 1] = 0

    # Take into account maximum feature magnitude
    delta_limits = constraints['limits'][None] - X[..., None]  # (N, D, 2)
    bounds[..., 0] = np.maximum(delta_limits[..., 0], bounds[..., 0])
    bounds[..., 1] = np.minimum(delta_limits[..., 1], bounds[..., 1])
    return torch.Tensor(bounds)


class LinearRecourse:
    """ Provide recourse for the linear classifier h(x) = <w, x> >= b """
    def __init__(self, w, b, c=None):
        """
        Inputs:     w: weights of the linear classifier, np.array with shape (D, 1)
                    b: float, bias of the linear classifier
                    c: np.array with shape (D, ), weight given to each function in the 1-norm cost function
        """
        self.w = w
        self.b = b

        if c is None:
            c = np.ones((w.shape[0], ))
        self.c = c

    def solve_lp(self, x, J, unactionable, bounds):
        """
        Inputs:     x: np.array (D), individual for which to compute recourse
                    J: np.array (D, D), Jacobian of the structural equations
                    unactionable: np.array (D), 1 if the corresponding feature is not actionable
                    bounds: np.array (D, 2), max and min values for each of the D features of the action

        Outputs:    a: np.array (D, ), recourse action found
                    cost: float, cost of the suggested recourse action
        """
        a = cp.Variable(x.shape[0])  # recourse action
        obj = cp.norm1(cp.multiply(self.c, a))  # loss is the 1 norm

        # Feasibility constraint
        b = self.b - self.w.T @ x  # <w, x + a> >= b --> <w, a> >= b + <w, x>
        constraints = [self.w.T @ J @ a >= b + 1e-6]  # add 1e-6 to ensure that constrain is met even with numerical errors

        # Actionability constrains
        constraints += [cp.multiply(unactionable, a) == 0]

        # Feasibility contrains (ensure feature within bounds, may only increase or decrease...)
        mask_lower = np.array([1. if bounds[i, 0] > -1e5 else 0. for i in range(bounds.shape[0])])
        mask_upper = np.array([1. if bounds[i, 1] < 1e5 else 0. for i in range(bounds.shape[0])])
        constraints += [cp.multiply(mask_lower, a) >= cp.multiply(mask_lower, bounds[:, 0])]
        constraints += [cp.multiply(mask_upper, a) <= cp.multiply(mask_upper, bounds[:, 1])]

        # Solve the linear problem and return the solution
        optim_problem = cp.Problem(cp.Minimize(obj), constraints)
        try:
            optim_problem.solve()
        except cp.error.SolverError:
            return np.zeros(x.shape), False, np.inf, np.zeros(x.shape)

        found_result = a.value is not None
        cf = x + J @ a.value if found_result else None
        found_result = found_result and self.w.T @ cf >= self.b
        return a.value, found_result, optim_problem.value, cf

    def find_recourse(self, x, interv_set, bounds, scm=None, verbose=False):
        """
        Inputs:     x: np.array (N, D), individual for which to compute recourse
                    interv_set: list of int, variables that can be acted upon (i.e., intervened)
                    bounds: np.array (N, D, 2), max and min values for each of the D features of the action
                    scm: None or scm.SCM

        Outputs:    action: np.array (N, D), recourse interventions found
                    finished: np.array (N, ), whether a valid counterfactual is returned
                    cost: np.array (N, D), cost of the suggested recourse action
                    cfs: np.array (N, D), counterfactuals
        """
        N, D = x.shape
        action = np.zeros((N, D)); finished = np.zeros(N); cost = np.zeros(N); cfs = np.zeros((N, D))

        unactionable = np.ones(D)
        unactionable[interv_set] = 0.

        J = np.eye(D) if scm is None else scm.get_Jacobian_interv(interv_set)
        iter_range = tqdm(range(N)) if verbose else range(N)
        for i in iter_range:
            action[i], finished[i], cost[i], cfs[i] = self.solve_lp(x[i], J, unactionable, bounds[i])

        return action, finished.astype(np.bool), cost, cfs

class DifferentiableRecourse:
    """
    Implementation of Algorithm 1 in the paper

    Reference implementations:  https://github.com/carla-recourse/CARLA
                                https://docs.seldon.io/projects/alibi/en/stable/methods/CF.html
    """
    def __init__(self, model, hyperparams, inner_max_pgd=False, early_stop=False):
        """
        Inputs:     model: torch.nn.Model, classifier for which to generate recourse
                    hyperparams: dict with the following hyperparameters
                        lr: float, learning rate of Adam
                        lambd_init: float, initial lambda regulating BCE loss and L2 loss
                        decay_rate: float < 1, at each outer iteration lambda is decreased by a factor of "decay_rate"
                        inner_iters: int, number of inner optimization steps (for a fixed lambda)
                        outer_iters: int, number of outer optimization steps (where lambda is decreased)
                    inner_max_pgd: bool, whether to use PGD or a first order approximation (FGSM) to solve the inner max
                    early_stop: bool, whether to do early stopping for the inner iterations
        """
        self.model = model
        self.lr = hyperparams['lr']
        self.lambd_init = hyperparams['lambd_init']
        self.decay_rate = hyperparams['decay_rate']
        self.inner_iters = hyperparams['inner_iters']
        self.outer_iters = hyperparams['outer_iters']
        self.n_of_samples = hyperparams['n_of_samples']
        self.inner_max_pgd = inner_max_pgd
        self.early_stop = early_stop
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def cost_function(self, delta: torch.tensor, W: torch.tensor):
        return torch.sum(torch.abs(delta[0, :, :] * W[-1, :, :]), -1)

    def find_recourse(self, x, interv_set, bounds, target=1., robust=False, epsilon=0.1, scm=None, verbose=True, counterfactual=True, W=None):
        """
        Find a recourse action for some particular intervention set (implementation of Algorithm 1 in the paper)

        Inputs:     x: torch.Tensor with shape (T, N, D), negatively classified instances for which to generate recourse
                    bounds: torch.Tensor with shape (N, D, 2), containing the min and max interventions
                    target: float, target label for the BCE loss (normally 1, that is, favourably classifier)
                    robust: bool, whether to guard against epsilon uncertainty
                    epsilon: float, amount of uncertainty, maximum perturbation magnitude (2-norm)
                    scm: type scm.SCM, structural causal model governing the causal relationships between features
                    interv_set: list of int, indices of the actionable features
                    verbose: bool
                    W: torch.tensor, the weights of the cost function

        Outputs:    actions: np.array with shape (T, N, D), recourse actions found
                    valid: np.array with shape (N, ), whether the corresponding recourse action is valid
                    cost: np.array with shape (N, ), cost of the recourse actions found (L1 norm)
                    cfs: np.array with shape (N, D), counterfactuals found (follow from x and actions)
        """
        T, N, D = x.shape[0], x.shape[1], x.shape[2]

        # If the cost are not specified, then it is a vector of zeros
        W = torch.ones((T, N, D)) if W is None else W

        x_og = torch.Tensor(x)
        x_pertb = torch.autograd.Variable(torch.zeros(x.shape), requires_grad=True)  # to calculate the adversarial
                                                                                     # intervention on the features
        ae_tol = 1e-4  # for early stopping
        actions = torch.zeros((1, N, D))  # store here valid recourse found so far

        target_vec = torch.zeros((N, 1))  # to feed into the BCE loss
        target_vec[:, 0] = torch.ones(N) * target
        unfinished = torch.ones((N, 1))  # instances for which recourse was not found so far

        # Define variable for which to do gradient descent, which can be updated with optimizer
        delta = torch.zeros((1, N, D))
        delta.requires_grad = True
        optimizer = torch.optim.Adam([delta], self.lr)

        # Models the effect of the recourse intervention on the features
        def recourse_model(x, delta, samples=1, return_samples=False):
            if scm is None:
                return x + delta * interv_set  # IMF
            else:
                if counterfactual:
                    return scm.counterfactual(x, delta, interv_set, True)  # counterfactual
                else:
                    return scm.interventional(x, delta, interv_set, True, sample=samples, return_samples=return_samples)

        # Perturbation model is only used when generating robust recourse, models perturbations on the features
        def perturbation_model(x, pertb, delta):
            if scm is None:
                return recourse_model(x, delta) + pertb
            else:
                x_prime = scm.counterfactual(x, pertb, None, True)
                if counterfactual:
                    return recourse_model(x_prime, delta)
                else:
                    return recourse_model(x_prime, delta)

        # Solve the first order approximation to the inner maximization problem
        def solve_first_order_approx(x_og, x_pertb, delta, target_vec):
            x_adv = perturbation_model(x_og, x_pertb, delta.detach())  # x_pertb is 0, only to backprop
            loss_x = self.bce_loss(self.model.predict_torch(x_adv[T-1, :, :], flatten=False), target_vec) * target_vec
            loss_x = torch.mean(loss_x[:, :])
            grad = torch.autograd.grad(loss_x, x_pertb, create_graph=False)[0]
            return grad / torch.linalg.norm(grad[:, :], dim=-1, keepdims=True) * epsilon  # akin to FGSM attack

        lambd = self.lambd_init
        prev_batch_loss = np.inf  # for early stopping
        pbar = tqdm(range(self.outer_iters)) if verbose else range(self.outer_iters)
        for outer_iter in pbar:
            for inner_iter in range(self.inner_iters):
                optimizer.zero_grad()

                # Find the adversarial perturbation (first order approximation, as in the paper)
                if robust:
                    pertb = solve_first_order_approx(x_og, x_pertb, delta, target_vec)
                    if self.inner_max_pgd:
                        # Solve inner maximization with projected gradient descent
                        pertb = torch.autograd.Variable(pertb, requires_grad=True)
                        optimizer2 = torch.optim.SGD([pertb], lr=0.1)

                        for _ in range(10):
                            optimizer2.zero_grad()
                            loss_pertb = torch.mean(self.bce_loss(self.model.predict_torch(x_og + pertb + delta.detach(), flatten=False),
                                                                  torch.zeros((T, N, 1))))
                            loss_pertb.backward()
                            optimizer2.step()

                            # Project to L2 ball, and with the linearity mask
                            with torch.no_grad():
                                norm = torch.linalg.norm(pertb, dim=-1)
                                too_large = norm > epsilon
                                pertb[too_large] = pertb[too_large] / norm[too_large, None] * epsilon
                            x_cf = x_og + pertb.detach() + delta
                    else:
                        x_cf = perturbation_model(x_og, pertb.detach(), delta)
                else:
                    x_cf = recourse_model(x_og, delta)

                with torch.no_grad():
                    # To continue optimazing, either the counterfactual or the adversarial counterfactual must be
                    # negatively classified
                    if not counterfactual:                        
                        pre_unfinished_cf = recourse_model(x_og, delta.detach(), samples=self.n_of_samples, return_samples=True) # (samples, T, N, D)
                        pre_unfinished_1 = self.model.predict_torch(pre_unfinished_cf[:, T-1, :, :]) == 0
                        pre_unfinished_1 = torch.any(pre_unfinished_1, dim=0, keepdim=True).squeeze(0)
                    else:
                        pre_unfinished_cf = recourse_model(x_og, delta.detach())
                        pre_unfinished_1 = self.model.predict_torch(pre_unfinished_cf[T-1, :, :]) == 0  # cf +1

                    pre_unfinished_2 = self.model.predict_torch(x_cf[T-1, :, :]) == 0  # cf adversarial
                    pre_unfinished = torch.logical_or(pre_unfinished_1[:, :], pre_unfinished_2[:, :])

                    # Add new solution to solutions
                    new_solution = torch.logical_and(unfinished, torch.logical_not(pre_unfinished)).flatten()
                    actions[0, new_solution, :] = torch.clone(delta[0, new_solution, :].detach())
                    unfinished = torch.logical_and(pre_unfinished, unfinished)

                # Compute loss. If the loss is computed over the interventional distribution,
                # we simply average over the loss of multiple samples
                if not counterfactual and scm is not None:
                    clf_loss = torch.zeros((T, N, 1))
                    x_cf = recourse_model(x_og, delta, samples=self.n_of_samples, return_samples=True)
                    clf_loss += torch.mean(
                        self.bce_loss(
                            self.model.predict_torch(x_cf[:, T-1, :, :], flatten=False),
                            target_vec.unsqueeze(0).expand(self.n_of_samples, -1, -1) # we extend this to match the sample size
                        ), 
                        axis=0
                    )
                else:
                    clf_loss = self.bce_loss(self.model.predict_torch(x_cf[T-1, :, :], flatten=False), target_vec)
                l1_loss = self.cost_function(delta, W)
                loss = clf_loss + lambd * l1_loss

                # Apply mask over the ones where recourse has already been found
                loss_mask = unfinished.to(torch.float).reshape(unfinished.shape[0], -1) * loss
                loss_mean = torch.mean(loss_mask[:])

                # Update x_cf
                loss_mean.backward()
                optimizer.step()

                # TODO: Satisfy the constraints on the features, by projecting delta
                #with torch.no_grad():
                #    delta[:, :] = torch.min(torch.max(delta, bounds[..., 0]), bounds[..., 1])

                # For early stopping
                if self.early_stop and inner_iter % (self.inner_iters // 10) == 0:
                    if loss_mean > prev_batch_loss * (1 - ae_tol):
                        break
                    prev_batch_loss = loss_mean

            lambd *= self.decay_rate

            if verbose:
                pbar.set_description("Pct left: %.3f Lambda: %.5f" % (float(unfinished[:].sum()/N), lambd))

            # Get out of the loop if recourse was found for every individual
            if not torch.any(unfinished):
                break        

        valid = torch.logical_not(unfinished).detach().cpu().numpy()
        cfs = recourse_model(x_og, actions).detach().cpu().numpy()
        cost = torch.sum(torch.sum(torch.abs(actions), 0), 1).detach().cpu().numpy()

        return actions.detach().cpu().numpy(), valid, cost, cfs

    def find_recourse_batch(self, x, interv_set, target=1., robust=False, epsilon=0.1, scm=None, verbose=True, counterfactual=True, W=None):
        """
        Find a recourse action for some particular intervention set (implementation of Algorithm 1 in the paper)

        Inputs:     x: torch.Tensor with shape (T, N, D), negatively classified instances for which to generate recourse
                    bounds: torch.Tensor with shape (N, D, 2), containing the min and max interventions
                    target: float, target label for the BCE loss (normally 1, that is, favourably classifier)
                    robust: bool, whether to guard against epsilon uncertainty
                    epsilon: float, amount of uncertainty, maximum perturbation magnitude (2-norm)
                    scm: type scm.SCM, structural causal model governing the causal relationships between features
                    interv_set: list of int, indices of the actionable features
                    verbose: bool
                    W: torch.tensor, the weights of the cost function

        Outputs:    actions: np.array with shape (T, N, D), recourse actions found
                    valid: np.array with shape (N, ), whether the corresponding recourse action is valid
                    cost: np.array with shape (N, ), cost of the recourse actions found (L1 norm)
                    cfs: np.array with shape (N, D), counterfactuals found (follow from x and actions)
        """

        S, T, N, D = x.shape

        # If the cost are not specified, then it is a vector of zeros
        W = torch.ones((T, N, D)) if W is None else W

        x_og = torch.Tensor(x)
        x_pertb = torch.autograd.Variable(torch.zeros((1, N, D)), requires_grad=True)  # to calculate the adversarial
                                                                                     # intervention on the features
        ae_tol = 1e-4  # for early stopping
        actions = torch.zeros((1, N, D))  # store here valid recourse found so far

        target_vec = torch.zeros((N, 1))  # to feed into the BCE loss
        target_vec[:, 0] = torch.ones(N) * target
        unfinished = torch.ones((N, 1))  # instances for which recourse was not found so far

        # Define variable for which to do gradient descent, which can be updated with optimizer
        delta = torch.zeros((1, N, D))
        delta.requires_grad = True
        optimizer = torch.optim.Adam([delta], self.lr)

        # Models the effect of the recourse intervention on the features
        def recourse_model(x, delta):
            if scm is None:
                return x + delta  # IMF
            else:
                if counterfactual:
                    return scm.counterfactual(x, delta, interv_set, True)  # counterfactual
                else:
                    return scm.interventional_batch(x, delta, interv_set, True)

        # Perturbation model is only used when generating robust recourse, models perturbations on the features
        def perturbation_model(x, pertb, delta):
            if scm is None:
                return recourse_model(x, delta) + pertb
            else:
                if counterfactual:
                    x_prime = scm.counterfactual(x, pertb, True)
                else:
                    return recourse_model(x, delta+pertb)
                return recourse_model(x_prime, delta)

        # Solve the first order approximation to the inner maximization problem
        def solve_first_order_approx(x_og, x_pertb, delta, target_vec):
            x_adv = perturbation_model(x_og, x_pertb, delta.detach())  # x_pertb is 0, only to backprop
            loss_x = self.bce_loss(
                self.model.predict_torch(x_adv[:, T-1, :, :], flatten=False),
                target_vec.unsqueeze(0).expand(S, -1, -1)
            )
            loss_x = torch.mean(torch.mean(loss_x, axis=0))
            grad = torch.autograd.grad(loss_x, x_pertb, create_graph=False)[0]
            return grad / torch.linalg.norm(grad[:, :], dim=-1, keepdims=True) * epsilon  # akin to FGSM attack

        lambd = self.lambd_init
        prev_batch_loss = np.inf  # for early stopping
        pbar = tqdm(range(self.outer_iters)) if verbose else range(self.outer_iters)
        for outer_iter in pbar:
            for inner_iter in range(self.inner_iters):
                optimizer.zero_grad()

                # Find the adversarial perturbation (first order approximation, as in the paper)
                if robust:
                    pertb = solve_first_order_approx(x_og, x_pertb, delta, target_vec)
                    x_cf = perturbation_model(x_og, pertb.detach(), delta)
                else:
                    x_cf = recourse_model(x_og, delta) # (S,T,N,D)

                with torch.no_grad():
                    # To continue optimazing, either the counterfactual or the adversarial counterfactual must be
                    # negatively classified
                    if not counterfactual:                        
                        pre_unfinished_cf = recourse_model(x_og, delta.detach()) # (S, T, N, D)
                        pre_unfinished_1 = self.model.predict_torch(pre_unfinished_cf[:, T-1, :, :]) == 0
                        pre_unfinished_1 = torch.any(pre_unfinished_1, dim=0, keepdim=True).squeeze(0)
                    else:
                        pre_unfinished_cf = recourse_model(x_og, delta.detach())
                        pre_unfinished_1 = self.model.predict_torch(pre_unfinished_cf[T-1, :, :]) == 0  # cf +1

                    pre_unfinished_2 = self.model.predict_torch(x_cf[:, T-1, :, :]) == 0  # cf adversarial
                    pre_unfinished_2 = torch.any(pre_unfinished_2, dim=0, keepdim=True).squeeze(0)
                    pre_unfinished = torch.logical_or(pre_unfinished_1[:, :], pre_unfinished_2[:, :])

                    # Add new solution to solutions
                    new_solution = torch.logical_and(unfinished, torch.logical_not(pre_unfinished)).flatten()
                    actions[0, new_solution, :] = torch.clone(delta[0, new_solution, :].detach())
                    unfinished = torch.logical_and(pre_unfinished, unfinished)

                # Compute loss. If the loss is computed over the interventional distribution,
                # we simply average over the loss of multiple samples
                if not counterfactual and scm is not None:
                    clf_loss = torch.zeros((T, N, 1))
                    x_cf = recourse_model(x_og, delta)
                    clf_loss += torch.mean(
                        self.bce_loss(
                            self.model.predict_torch(x_cf[:, T-1, :, :], flatten=False),
                            target_vec.unsqueeze(0).expand(S, -1, -1) # we extend this to match the sample size
                        ), 
                        axis=0
                    )
                else:
                    clf_loss = self.bce_loss(self.model.predict_torch(x_cf[T-1, :, :], flatten=False), target_vec)
                l1_loss = self.cost_function(delta, W)
                loss = clf_loss + lambd * l1_loss

                # Apply mask over the ones where recourse has already been found
                loss_mask = unfinished.to(torch.float).reshape(unfinished.shape[0], -1) * loss
                loss_mean = torch.mean(loss_mask[:])

                # Update x_cf
                loss_mean.backward()
                optimizer.step()

                # For early stopping
                if self.early_stop and inner_iter % (self.inner_iters // 10) == 0:
                    if loss_mean > prev_batch_loss * (1 - ae_tol):
                        break
                    prev_batch_loss = loss_mean

            lambd *= self.decay_rate

            if verbose:
                pbar.set_description("Pct left: %.3f Lambda: %.5f" % (float(unfinished[:].sum()/N), lambd))

            # Get out of the loop if recourse was found for every individual
            if not torch.any(unfinished):
                break        

        valid = torch.logical_not(unfinished).detach().cpu().numpy()
        cfs = recourse_model(x_og, actions).detach().cpu().numpy()
        cost = torch.sum(torch.sum(torch.abs(actions), 0), 1).detach().cpu().numpy()

        return actions.detach().cpu().numpy(), valid, cost, cfs

def causal_recourse(x, explainer, constraints, scm=None, verbose=True, when_to_apply=1, time_robust=False, **kwargs):
    """
    Function to find causal recourse, by calling rec_fcn for every possible intervention set

    Inputs:     x: torch.Tensor with shape (T, N, D), negatively classified instances for which to generate recourse
                explainer: Method to generate recourse solutions
                constraints: dict with the following keys:
                    actionable: list of int, indices of the actionable features
                    increasing: list of int, indices of the features which can only increase
                    decreasing: list of int, indices of the features which can only decrease
                    limits: list of list of float, feature bounds
                scm: type scm.SCM, structural causal model governing the causal relationships between features
                verbose: bool
                interval: windown within we want to find recourse
                **kwargs: correspond to the arguments of self.find_recourse()

    Outputs:    actions: np.array with shape (T, N, D), recourse actions found
                valids: np.array with shape (N, ), whether the corresponding recourse action is valid
                costs: np.array with shape (N, ), cost of the recourse actions found (L1 norm)
                counterfacs: np.array with shape (N, D), counterfactuals found (follow from x and actions)
                valid_interventions: list of lists of int, intervention set for each valid recourse action
    """

    if not time_robust:
        T, N, D = x.shape[0], x.shape[1], x.shape[2]
    else:
        S, T, N, D = x.shape

    if scm is not None:
        sets = scm.getPowerset(constraints['actionable'], when_to_apply)  # every possible intervention set
    else:
        sets = [[(when_to_apply, i, True) for i in constraints['actionable']]]

    # Arrays to store the best recourse found so far (as one iterates through intervention sets)
    actions = np.zeros((1, N, D))
    valids = np.zeros(N).astype(bool)
    costs = np.ones(N) * np.inf
    counterfacs = np.zeros((1, N, D)) if not time_robust else np.zeros((S, 1, N, D))
    interventions = [None]*N

    for i in tqdm(range(len(sets)), disable=not verbose):
        interv_set = list(sets[i])

        mask_intervention = torch.zeros((N, D))
        for (_, i, _) in interv_set:
            mask_intervention[:, i] = 1

        # For the feasibility set, we can consider a single instance alone
        bounds = build_feasibility_sets(x[0, :, :], interv_set, constraints) if not time_robust else build_feasibility_sets(x[0, 0, :, :], interv_set, constraints)

        # Recourse for that subset
        if not time_robust:
            action, finished, cost, cfs = explainer.find_recourse(x, mask_intervention, bounds, scm=scm, verbose=verbose, **kwargs)
        else:
            action, finished, cost, cfs = explainer.find_recourse_batch(x, mask_intervention, scm=scm, verbose=verbose, **kwargs)
        
        # Get only the last step for everyone
        finished = finished.flatten()
        cost = cost.flatten()

        # Which recourse actions have lower cost of recourse?
        arg_finished = np.argwhere(finished) # Get only those with a positive recourse at the last timestep
        less_cost = cost[finished] < costs[finished] # Get those recourse options with a smaller last cost
        arg_less_cost = arg_finished[less_cost]

        # Example arg_less_cost given the rows
        actions[0, arg_less_cost.flatten(), :] = action[0, arg_less_cost.flatten(), :]

        valids[arg_less_cost] = True
        costs[arg_less_cost] = cost[arg_less_cost]
        if not time_robust:
            counterfacs[0, arg_less_cost.flatten(), :] = cfs[0, arg_less_cost.flatten(), :]
        else:
            counterfacs[:, 0, arg_less_cost.flatten(), :] = cfs[:, 0, arg_less_cost.flatten(), :]

        for j in arg_less_cost:
            interventions[j[0]] = interv_set

    # Only return the intervention sets for which valid recourse was found
    valid_interventions = []
    for val in interventions:
        if val is not None:
            valid_interventions.append(val)

    # Convert interventions to mask
    interv_mask = torch.zeros((T, len(valid_interventions), D))
    for i, interv_set in enumerate(valid_interventions):
        for timestep, feature, act_upon in interv_set:
            interv_mask[timestep, i, feature] = 1

    return actions, valids, costs, counterfacs, interv_mask