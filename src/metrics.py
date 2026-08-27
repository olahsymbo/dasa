import numpy as np
import torch
from tqdm.auto import tqdm


def _iou(p, t, c):
    p = p.reshape(-1)
    t = t.reshape(-1)
    vals = []
    for k in range(c):
        pi = p == k
        ti = t == k
        inter = (pi & ti).sum().item()
        union = (pi | ti).sum().item()
        vals.append(np.nan if union == 0 else inter / union)
    return vals


def _dice(p, t, c):
    p = p.reshape(-1)
    t = t.reshape(-1)
    vals = []
    for k in range(c):
        pi = p == k
        ti = t == k
        inter = (pi & ti).sum().item()
        denom = pi.sum().item() + ti.sum().item()
        vals.append(np.nan if denom == 0 else 2 * inter / denom)
    return vals


@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()
    ious = []
    dices = []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(1)
        for p, t in zip(pred.cpu(), y.cpu()):
            ious.append(_iou(p, t, num_classes))
            dices.append(_dice(p, t, num_classes))
    pci = np.nanmean(np.asarray(ious, dtype=np.float32), 0)
    pcd = np.nanmean(np.asarray(dices, dtype=np.float32), 0)
    out = {"mIoU": float(np.nanmean(pci)), "mDice": float(np.nanmean(pcd))}
    for k in range(num_classes):
        out[f"class_{k}_IoU"] = float(pci[k])
        out[f"class_{k}_Dice"] = float(pcd[k])
    return out
