import torch.optim as optim

def get_optimizer(model, lr, momentum):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)