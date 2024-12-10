from pytorch_lightning.cli import LightningCLI
import torch
import os
import sys
import yaml
from pathlib import Path
from typing import Optional
import questionary

def get_config_files() -> list[Path]:
    """Get all YAML config files from the config directory."""
    config_dir = Path("config")
    yaml_files = list(config_dir.glob("*.yaml"))
    yml_files = list(config_dir.glob("*.yml"))
    return sorted(yaml_files + yml_files)

def select_config() -> Optional[str]:
    """Interactive CLI for selecting a config file."""
    config_files = get_config_files()

    if not config_files:
        print("No config files found in config/ directory")
        return None

    choices = [str(f.relative_to("config")) for f in config_files]
    selected = questionary.select(
        "Select a config file:",
        choices=choices
    ).ask()

    if selected:
        return str(Path("config") / selected)
    return None

def select_mode() -> str:
    """Interactive CLI for selecting Lightning mode."""
    modes = ["fit", "validate", "test", "predict"]
    selected = questionary.select(
        "Select training mode:",
        choices=modes
    ).ask()
    return selected

def cli_main():
    # Select mode if not provided
    if not any(arg in sys.argv[1:] for arg in ["fit", "validate", "test", "predict"]):
        mode = select_mode()
        if mode:
            sys.argv.append(mode)

    # Check if config flag is present
    if not any(arg in sys.argv for arg in ["-c", "--config"]):
        config_path = select_config()
        if config_path:
            sys.argv.extend(["--config", config_path])

    if "--trainer.logger.init_args.name" not in sys.argv:
        name = input("Enter wandb name (or press ENTER to use default random name): ")
        if len(name) > 0:
            sys.argv.extend(["--trainer.logger.init_args.name", name])


    cli = LightningCLI(save_config_callback=None)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    cli_main()