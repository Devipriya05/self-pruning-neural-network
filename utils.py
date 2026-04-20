import torch

def sparsity_loss(model):
    loss = 0
    for module in model.modules():
        if hasattr(module, 'gate_scores'):
            gates = torch.sigmoid(module.gate_scores)
            loss += gates.sum()
    return loss


def calculate_sparsity(model):
    total = 0
    pruned = 0

    for module in model.modules():
        if hasattr(module, 'gate_scores'):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            pruned += (gates < 1e-2).sum().item()

    return 100 * pruned / total
