from rl.common.device import device
import torch


def to_torch(x, device=device):
    if isinstance(x, torch.Tensor):
        return x.to(device, dtype=torch.float32)
    else:
        return torch.tensor(x, dtype=torch.float32, device=device)

def weighted_smooth_l1_loss(x, y, weights):
    if weights is None:
        weights = torch.ones_like(x)
    diff = x - y
    loss = torch.where(
        torch.abs(diff) < 1,
        0.5 * weights * diff ** 2,
        (torch.abs(diff) - 0.5)*weights,
    )
    return loss.mean()
