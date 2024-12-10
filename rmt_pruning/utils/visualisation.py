import matplotlib.pyplot as plt
import numpy as np
import wandb

def plot_accuracy_vs_parameters(accuracies, num_params_pruned, num_params_unpruned, min_accuracy=90):
    plt.figure(figsize=(10, 6))
    params_kept_percentages = [100 * num / num_params_unpruned for num in num_params_pruned]

    for num, acc, pct in zip(num_params_pruned, accuracies, params_kept_percentages):
        if acc >= min_accuracy:
            plt.scatter(num, acc, marker='o')
            plt.annotate(f"{pct:.1f}%", (num, acc), textcoords="offset points",
                        xytext=(0,10), ha='center')

    plt.xlabel("Number of Parameters Kept")
    plt.ylabel("Test Set Accuracy")
    plt.title(f"Test Set Accuracy vs Parameters Kept (Accuracy >= {min_accuracy}%)")
    plt.grid(True)

    if wandb.run is not None:
        wandb.log({"accuracy_vs_parameters": wandb.Image(plt)})

    # plt.show()

def plot_spectra(v, lamda_plus, gamma, sigma_sq, pTilde):
    """Visualize eigenvalue spectrum and MP distribution"""

    # helper functions
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

    plt.figure(figsize=(15, 5))

    # Full spectrum plot
    plt.subplot(1, 2, 1)
    plt.hist(v[-pTilde:], bins=100, color="black",
            label="Empirical Density", density=True)
    plt.axvline(x=lamda_plus, label="Lambda Plus", color="red")
    plt.legend()
    plt.title("Empirical Distribution Density")

    # Truncated spectrum plot
    plt.subplot(1, 2, 2)
    eigsTruncated = [i for i in v[-pTilde:] if i < lamda_plus]
    plt.hist(eigsTruncated, bins=100, color="black",
            label="Truncated Empirical Density", density=True)

    # Add theoretical density
    if len(eigsTruncated):
        Z = np.linspace(min(eigsTruncated), max(eigsTruncated), 100)
        Y = mp_density_wrapper(gamma, sigma_sq, Z)
        plt.plot(Z, Y, color="orange", label="Predicted Density")
        plt.legend()
        plt.title("Density Comparison Zoomed")

    plt.tight_layout()
    # plt.show()

    wandb.log({"eigenvalue_spectrum": plt})