import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_ohem_weights(
    model,
    loader,
    device,
    keep_ratio=0.3,
):
    """
    Higher sampling weight for hard examples.
    """

    model.eval()

    losses = []

    for idx, (x, y) in enumerate(loader):

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = F.cross_entropy(logits, y, reduction="none")

        loss = loss.mean(dim=(1, 2))

        losses.extend(loss.cpu().tolist())

    losses = torch.tensor(losses)

    threshold = torch.quantile(losses, 1.0 - keep_ratio)

    weights = {}

    for i, l in enumerate(losses):

        if l >= threshold:
            weights[i] = 1.0
        else:
            weights[i] = 0.2

    return weights
