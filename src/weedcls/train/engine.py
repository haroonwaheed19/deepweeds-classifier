import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def predict_logits(model, loader, device="cpu"):
    """Returns (logits, labels) as numpy arrays, in loader order."""
    model.eval()
    outs, ys = [], []
    for x, y in loader:
        outs.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(outs).numpy(), torch.cat(ys).numpy()


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return F.softmax(torch.from_numpy(logits) / temperature, dim=1).numpy()
