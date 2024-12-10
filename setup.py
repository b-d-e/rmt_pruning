from setuptools import setup, find_packages

setup(
    name="rmt_pruning",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "wandb>=0.12.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "tracywidom>=0.1.0",
        "jsonargparse[signatures]>=4.27.7",
        "plotly>=5.24.1",
    ],
)