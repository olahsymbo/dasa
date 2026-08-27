import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm.auto import tqdm


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()


def rarity_scores(ds, ids, c):
    counts = np.zeros(c)
    for i in tqdm(ids, desc="class pixels", leave=False):
        _, m = ds[i]
        arr = m.numpy()
        for k in range(c):
            counts[k] += (arr == k).sum()
    freq = counts / max(counts.sum(), 1)
    rarity = 1 / (freq + 1e-8)
    rarity = rarity / rarity.max()
    scores = []
    for i in tqdm(ids, desc="rarity", leave=False):
        _, m = ds[i]
        present = [int(k) for k in np.unique(m.numpy()) if int(k) < c]
        scores.append(float(np.mean([rarity[k] for k in present])) if present else 0.0)
    return np.asarray(scores), rarity


def boundary_score(mask):
    m = mask.numpy().astype(np.int32)
    dx = np.abs(m[:, 1:] - m[:, :-1]) > 0
    dy = np.abs(m[1:, :] - m[:-1, :]) > 0
    return float((dx.sum() + dy.sum()) / (m.shape[0] * m.shape[1]))


@torch.no_grad()
def estimate_difficulty(model, ds, ids, bs, c, mc, device, num_workers):
    loader = DataLoader(
        Subset(ds, ids),
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    amb = []
    loss = []
    model.eval()
    for x, y in tqdm(loader, desc="model scores", leave=False):
        x = x.to(device)
        y = y.to(device)
        probs = []
        for _ in range(mc):
            enable_dropout(model)
            probs.append(torch.softmax(model(x), 1))
        mean = torch.stack(probs).mean(0)
        a = -(mean * torch.log(mean + 1e-8)).sum(1)
        amb.extend(a.mean((1, 2)).cpu().numpy().tolist())
        nll = F.nll_loss(torch.log(mean + 1e-8), y, reduction="none")
        loss.extend(nll.mean((1, 2)).cpu().numpy().tolist())
    rare, _ = rarity_scores(ds, ids, c)
    bound = np.asarray(
        [boundary_score(ds[i][1]) for i in tqdm(ids, desc="boundary", leave=False)]
    )
    return pd.DataFrame(
        {
            "original_idx": ids,
            "ambiguity": norm(amb),
            "loss": norm(loss),
            "rarity": norm(rare),
            "boundary": norm(bound),
        }
    )


def combine(df, method):
    if method == "loss_only":
        z = df.loss.values
    elif method == "rarity_only":
        z = df.rarity.values
    elif method == "boundary_only":
        z = df.boundary.values
    elif method == "DASA_EM":
        z = (
            0.35 * df.ambiguity.values
            + 0.35 * df.loss.values
            + 0.20 * df.rarity.values
            + 0.10 * df.boundary.values
        )
    elif method == "random_weighted":
        z = np.random.default_rng(42).random(len(df))
    else:
        z = np.zeros(len(df))
    out = df.copy()
    out["difficulty_z"] = norm(z)
    return out


def weighted_loader(diff, train_aug_fn, bs, budget, num_workers, strong=True):
    ids = diff.original_idx.astype(int).tolist()
    z = diff.difficulty_z.values.astype(np.float32)
    weights = torch.DoubleTensor(1 + budget * z)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    strengths = {int(r.original_idx): float(r.difficulty_z) for r in diff.itertuples()}
    ds = train_aug_fn(adaptive=strengths, strong=strong)
    return DataLoader(
        Subset(ds, ids),
        batch_size=bs,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
