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

    plt.show()