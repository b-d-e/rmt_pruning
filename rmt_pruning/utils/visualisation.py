import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import wandb

# def plot_accuracy_vs_parameters(accuracies, num_params_pruned, num_params_unpruned, min_accuracy=90):
#     # Calculate percentages
#     params_kept_percentages = [100 * num / num_params_unpruned for num in num_params_pruned]

#     # Filter points based on minimum accuracy
#     mask = [acc >= min_accuracy for acc in accuracies]
#     filtered_nums = [n for n, m in zip(num_params_pruned, mask) if m]
#     filtered_accs = [a for a, m in zip(accuracies, mask) if m]
#     filtered_pcts = [p for p, m in zip(params_kept_percentages, mask) if m]

#     # Create scatter plot
#     fig = go.Figure()

#     fig.add_trace(go.Scatter(
#         x=filtered_nums,
#         y=filtered_accs,
#         mode='markers+text',
#         text=[f"{pct:.1f}%" for pct in filtered_pcts],
#         textposition="top center",
#         hovertemplate="Parameters: %{x}<br>Accuracy: %{y:.2f}%<br>Percentage: %{text}<extra></extra>"
#     ))

#     fig.update_layout(
#         title=f"Test Set Accuracy vs Parameters Kept (Accuracy >= {min_accuracy}%)",
#         xaxis_title="Number of Parameters Kept",
#         yaxis_title="Test Set Accuracy",
#         showlegend=False,
#         template="plotly_white"
#     )

#     if wandb.run is not None:
#         wandb.log({"accuracy_vs_parameters": fig})

#     return fig



def plot_spectra(v, lamda_plus, gamma, sigma_sq, pTilde):
    """Visualize eigenvalue spectrum and MP distribution"""

    def mp_density_wrapper(gamma, sigma_sq, sample_points):
        """Compute MP density at sample points"""
        lp = sigma_sq*np.power(1+np.sqrt(gamma), 2)
        lm = sigma_sq*np.power(1-np.sqrt(gamma), 2)

        return np.array([
            mp_density_inner(gamma, sigma_sq, x) if lm <= x <= lp else 0
            for x in sample_points
        ])

    def mp_density_inner(gamma, sigma_sq, x):
        """Helper function to compute MP density at a point"""
        lp = sigma_sq*np.power(1+np.sqrt(gamma), 2)
        lm = sigma_sq*np.power(1-np.sqrt(gamma), 2)
        return np.sqrt((lp-x)*(x-lm))/(gamma*x*2*np.pi*sigma_sq)

    # Create subplot figure
    fig = sp.make_subplots(rows=1, cols=2,
                          subplot_titles=["Empirical Distribution Density",
                                        "Density Comparison Zoomed"])

    # Full spectrum plot
    eigenvalues = v[-pTilde:]
    fig.add_trace(
        go.Histogram(x=eigenvalues, histnorm='probability density',
                    name="Empirical Density", nbinsx=100),
        row=1, col=1
    )

    fig.add_vline(x=lamda_plus, line_dash="dash", line_color="red",
                  annotation_text="Lambda Plus", row=1, col=1)

    # Truncated spectrum plot
    eigsTruncated = [i for i in eigenvalues if i < lamda_plus]
    if len(eigsTruncated):
        fig.add_trace(
            go.Histogram(x=eigsTruncated, histnorm='probability density',
                        name="Truncated Empirical Density", nbinsx=100),
            row=1, col=2
        )

        # Add theoretical density
        Z = np.linspace(min(eigsTruncated), max(eigsTruncated), 100)
        Y = mp_density_wrapper(gamma, sigma_sq, Z)
        fig.add_trace(
            go.Scatter(x=Z, y=Y, name="Predicted Density",
                      line=dict(color='orange')),
            row=1, col=2
        )

    fig.update_layout(
        height=500,
        width=1200,
        showlegend=True,
        template="plotly_white"
    )

    if wandb.run is not None:
        wandb.log({"eigenvalue_spectrum": fig})

    return fig