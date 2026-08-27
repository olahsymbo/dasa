import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .metrics import evaluate


def train_epochs(
    model, train_loader, val_loader, epochs, lr, weight_decay, num_classes, device
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    rows = []
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f"train {ep}/{epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        row = {
            "epoch": ep,
            "train_loss": float(np.mean(losses)),
            **evaluate(model, val_loader, num_classes, device),
        }
        print(row)
        rows.append(row)
    return model, pd.DataFrame(rows)
