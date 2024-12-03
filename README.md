_repository is a work in progress._

# Random Matrix Theory Informed Pruning

This is a refactoring of the iPython notebook located at [yspennstate/RMT_pruning_2](https://github.com/yspennstate/RMT_pruning_2) - code used by Beryland et al in their preprint _"Enhancing Accuracy in Deep Learning Using Random Matrix
Theory"_ (`arXiv:2310.03165`).

I take no credit for the original work and this is currently a simple re-factor of their code, for my own convenience, rather than contributing anything new - nearly all functionality in this repo is from the original notebook. If you are intersested in a RMT-informed approach to pruning you should [read their paper](https://arxiv.org/abs/2310.03165)!

### Getting started

Install packages with `pip install -e .` and your favourite environment manager.

If desired, modify the PyTorch Lightning configuration in `config/config.yaml` - in particular though, make sure to set a correct weights and biases entity (user/team) name.
If you haven't used wandb before, run `wandb login` and follow instructions to update your `.netrc` with an access token.

Run `python train.py fit --config config/config.yaml` to execute the training process.

