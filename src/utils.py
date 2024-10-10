import torch
import numpy as np

import seaborn as sns

TABLE_ORDERING = [
    r"\texttt{CAR} ($\epsilon = 0.05$)",
    r"\texttt{CAR} ($\epsilon = 0.5$)",
    r"\texttt{SAR} ($\epsilon = 0.05$)",
    r"\texttt{SAR} ($\epsilon = 0.5$)",
    r"\texttt{IMF} ($\epsilon = 0.05$)",
    r"\texttt{IMF} ($\epsilon = 0.5$)",
    r"\texttt{T-SAR}"
]

PLOT_ORDERING = [
    r"\texttt{IMF} ($\epsilon = 0.5$)",
    r"\texttt{SAR} ($\epsilon = 0.5$)",
    r"\texttt{CAR} ($\epsilon = 0.5$)",
    r"\texttt{IMF} ($\epsilon = 0.05$)",
    r"\texttt{SAR} ($\epsilon = 0.05$)",
    r"\texttt{CAR} ($\epsilon = 0.05$)",
    r"\texttt{T-SAR}"
]

TABLE_ORDERING_SYN = [
    r"\texttt{CAR} ($\epsilon = 3$)",
    r"\texttt{CAR} ($\epsilon = 5$)",
    r"\texttt{SAR} ($\epsilon = 3$)",
    r"\texttt{SAR} ($\epsilon = 5$)",
    r"\texttt{IMF} ($\epsilon = 3$)",
    r"\texttt{IMF} ($\epsilon = 5$)",
    r"\texttt{T-SAR}"
]

PLOT_ORDERING_SYN = [
    r"\texttt{IMF} ($\epsilon = 5$)",
    r"\texttt{SAR} ($\epsilon = 5$)",
    r"\texttt{CAR} ($\epsilon = 5$)",
    r"\texttt{IMF} ($\epsilon = 3$)",
    r"\texttt{SAR} ($\epsilon = 3$)",
    r"\texttt{CAR} ($\epsilon = 3$)",
    r"\texttt{T-SAR}"
]

# Palette needed for the plots
color_palette_dimmed = sns.color_palette("colorblind")[0:7]
color_palette_dimmed.reverse()
color_palette_dimmed = [color if k==6 else (color[0], color[1], color[2], 1) for k, color in enumerate(color_palette_dimmed)]
PALETTE = {
    m: c for m,c in zip(PLOT_ORDERING, color_palette_dimmed)
}
PALETTE_SYN = {
    m: c for m,c in zip(PLOT_ORDERING_SYN, color_palette_dimmed)
}

def apply_solution(initial_T, max_T, actions, model, scm, X_original, test_index_negatively_classified, validity):
    
    # Change type
    validity = torch.Tensor(validity).int()

    # All arrays
    full_recourse = []
    avg_recourse = []

    for t in range(initial_T, max_T):
        
        # We consider all the elements as valid and we filter later
        # We also regenerate the actionable matrix (which is all by default zero)
        # Basically, we set to 1 the entries where extended_actions is not empty
        result = scm.counterfactual(
            torch.Tensor(X_original[:t+1, test_index_negatively_classified, :]),
            torch.Tensor(actions),
            soft_interv=True
        )

        # Given the elements we think we got recourse for, we compute again the validity
        # by checking if we got really recourse for those.
        variation_in_recourse = model.predict_torch(torch.FloatTensor(result))[t, :]
        variation_in_recourse = np.array(variation_in_recourse * validity.view(validity.shape[0], 1), dtype=bool).flatten()

        # Append the results
        full_recourse.append(variation_in_recourse)
        avg_recourse.append(np.mean(variation_in_recourse))
    
    return avg_recourse, full_recourse