import torch
import torch.nn.functional as F
import wandb

def train_epoch(model, device, train_loader, optimizer, epoch, config, log_to_wandb=True):
    model.train()
    model.to(device)

    l1_lambda = 0.000005
    l2_lambda = 0.000005

    correct = 0
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)

        # Regularization
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        l2_norm = sum((p ** 2).sum() for p in model.parameters())
        loss += l1_lambda * l1_norm + l2_lambda * l2_norm

        total_loss += loss.item()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % config['log_interval'] == 0 and config.get('show_training_loss', False):
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)

    print(f'Train set: Average loss: {avg_loss:.4f}, '
          f'Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.0f}%)')

    if log_to_wandb:
        wandb.log({
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "epoch": epoch
        })

    return accuracy, avg_loss

def test_epoch(model, device, test_loader, epoch, log_to_wandb=True):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')

    if log_to_wandb:
        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "epoch": epoch
        })

    return accuracy, test_loss