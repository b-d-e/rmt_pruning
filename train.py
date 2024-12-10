from pytorch_lightning.cli import LightningCLI
import os
import sys

def cli_main():
    cli = LightningCLI(save_config_callback=None)

if __name__ == "__main__":

    if "--trainer.logger.init_args.name" not in sys.argv:
        name = input("Enter wandb name (or press ENTER to use default random name): ")
        if len(name) > 0:
            sys.argv.append("--trainer.logger.init_args.name")
            sys.argv.append(name)

    cli_main()