import torch
import yaml
import wandb
from pathlib import Path
from models.network_model import NetworkModel
from utils.training import train_epoch, test_epoch
from utils.pruning import compute_eigs_to_keep
from utils.visualization import plot_accuracy_vs_parameters
from utils.data import get_data_loaders
from utils.optimizer import get_optimizer

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        config=config
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data loaders
    train_loader, test_loader = get_data_loaders(
        batch_size=config['training']['batch_size'],
        seed=config.get('seed', 42)
    )

    # Create model
    model = NetworkModel(
        config['network']['use_relu'],
        config['network']['dims'],
        config['pruning']['alpha'],
        config['pruning']['beta'],
        config['network']['goodness_of_fit_cutoff']
    ).to(device)

    # Initialize optimizer
    optimizer = get_optimizer(
        model,
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum']
    )

    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer,
        lr_lambda=lambda epoch: 0.96
    )

    # Initial parameter count
    initial_params = model.get_parameter_count()
    wandb.log({"initial_parameter_count": initial_params})

    # Training loop
    for epoch in range(config['training']['epochs']):
        # Pruning step
        if epoch % config['training']['split_frequency'] == 0 and epoch != 0:
            for i, layer in enumerate(model.fc):
                lp_transformed, eigs_to_keep, good_fit = compute_eigs_to_keep(
                    model,
                    model.get_layer_matrix(i),
                    model.dims,
                    epoch,
                    config['network']['goodness_of_fit_cutoff']
                )

                # Log pruning metrics
                wandb.log({
                    f"layer_{i}_eigs_kept": eigs_to_keep,
                    f"layer_{i}_good_fit": good_fit,
                    "epoch": epoch
                })

                # Update optimizer if layer structure changes
                optimizer = get_optimizer(
                    model,
                    lr=optimizer.param_groups[0]['lr'],
                    momentum=config['training']['momentum']
                )

        # Train epoch
        train_accuracy, train_loss = train_epoch(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            config['training']
        )

        # Test epoch
        test_accuracy, test_loss = test_epoch(
            model,
            device,
            test_loader,
            epoch
        )

        # Step LR scheduler
        lr_scheduler.step()

        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "parameter_count": model.get_parameter_count(),
            "parameter_reduction": 1 - (model.get_parameter_count() / initial_params)
        })

        # Save model checkpoint
        if epoch % config['training'].get('checkpoint_frequency', 10) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, f'checkpoints/model_epoch_{epoch}.pt')

            # Log checkpoint to wandb
            wandb.save(f'checkpoints/model_epoch_{epoch}.pt')

if __name__ == "__main__":
    main()