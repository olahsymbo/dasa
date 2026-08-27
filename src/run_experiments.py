import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.ohem import compute_ohem_weights
from src.ohem_sampler import build_ohem_sampler

from .dasa import combine, estimate_difficulty, weighted_loader
from .datasets import build_dataset_factory
from .metrics import evaluate
from .models import build_model, count_params
from .train import train_epochs


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_cfg(p):
    with open(p) as f:
        return yaml.safe_load(f)


def loader(ds, ids=None, bs=16, shuffle=False, nw=2):
    if ids is not None:
        ds = Subset(ds, ids)
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle, num_workers=nw, pin_memory=True
    )


def run_one(cfg, dataset, model_name, method):
    seed_all(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===== {dataset} | {model_name} | {method} | {device} =====")
    out = Path(cfg["output_root"])
    csv = out / "csv"
    ckpt = out / "checkpoints"
    plots = out / "plots"
    for d in [csv, ckpt, plots, out / "logs"]:
        d.mkdir(parents=True, exist_ok=True)
    base, test, train_aug, val_clean, c = build_dataset_factory(
        dataset, cfg["data_root"], cfg["image_size"]
    )
    ids = list(range(len(base)))
    if cfg.get("max_samples") is not None:
        ids = ids[: int(cfg["max_samples"])]
    tr, va = train_test_split(ids, test_size=0.15, random_state=cfg["seed"])
    bs = (
        cfg.get("segformer_batch_size", cfg["batch_size"])
        if model_name == "segformer_b0"
        else cfg["batch_size"]
    )
    nw = cfg.get("num_workers", 2)
    val_loader = loader(val_clean(), va, bs, False, nw)
    test_loader = loader(test, None, bs, False, nw)
    strong = method == "strong_uniform"
    # train_loader=loader(train_aug(None,strong),tr,bs,True,nw)
    train_dataset = train_aug(None, strong)

    train_loader = loader(train_dataset, tr, bs, True, nw)
    model = build_model(model_name, c).to(device)
    print("params", count_params(model))
    if method in ["baseline", "strong_uniform"]:
        model, hist = train_epochs(
            model,
            train_loader,
            val_loader,
            cfg["baseline_epochs"],
            cfg["lr"],
            cfg["weight_decay"],
            c,
            device,
        )
    elif method == "ohem":

        sample_weights = compute_ohem_weights(
            model, train_loader, device, keep_ratio=0.3
        )

        sampler = build_ohem_sampler(train_dataset, sample_weights)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )

        model, hist = train_epochs(
            model,
            train_loader,
            val_loader,
            cfg["baseline_epochs"],
            cfg["lr"],
            cfg["weight_decay"],
            n_classes,
            device,
        )
    else:
        model, warm = train_epochs(
            model,
            train_loader,
            val_loader,
            cfg["baseline_epochs"],
            cfg["lr"],
            cfg["weight_decay"],
            c,
            device,
        )
        hists = [warm]
        for r in range(1, cfg["em_rounds"] + 1):
            raw = estimate_difficulty(
                model, base, tr, bs, c, cfg["mc_passes"], device, nw
            )
            diff = combine(raw, method)
            diff["round"] = r
            diff["dataset"] = dataset
            diff["model"] = model_name
            diff["method"] = method
            diff.to_csv(
                csv / f"difficulty_{dataset}_{model_name}_{method}_round{r}.csv",
                index=False,
            )
            wl = weighted_loader(
                diff,
                train_aug,
                bs,
                cfg["budget_strength"],
                nw,
                strong=(method != "random_weighted"),
            )
            model, rh = train_epochs(
                model,
                wl,
                val_loader,
                cfg["epochs_per_m_step"],
                cfg["lr"],
                cfg["weight_decay"],
                c,
                device,
            )
            rh["round"] = r
            hists.append(rh)
        hist = pd.concat(hists, ignore_index=True)
    metrics = evaluate(model, test_loader, c, device)
    res = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "model": model_name,
                "method": method,
                "params": count_params(model),
                **metrics,
            }
        ]
    )
    stem = f"{dataset}_{model_name}_{method}"
    hist.to_csv(csv / f"history_{stem}.csv", index=False)
    res.to_csv(csv / f"result_{stem}.csv", index=False)
    if cfg.get("save_checkpoints", True):
        torch.save(model.state_dict(), ckpt / f"{stem}.pt")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    datasets = cfg.get("datasets", [cfg.get("dataset", "pet")])
    rows = []
    for ds in datasets:
        for m in cfg["models"]:
            for method in cfg["methods"]:
                start = time.time()
                try:
                    r = run_one(cfg, ds, m, method)
                    r["runtime_min"] = (time.time() - start) / 60
                    rows.append(r)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("OOM skip", ds, m, method)
                        torch.cuda.empty_cache()
                        rows.append(
                            pd.DataFrame(
                                [
                                    {
                                        "dataset": ds,
                                        "model": m,
                                        "method": method,
                                        "error": "OOM",
                                    }
                                ]
                            )
                        )
                    else:
                        raise
                summary = pd.concat(rows, ignore_index=True)
                Path(cfg["output_root"], "csv").mkdir(parents=True, exist_ok=True)
                summary.to_csv(
                    Path(cfg["output_root"]) / "csv" / "ALL_RESULTS_SUMMARY.csv",
                    index=False,
                )
                print(summary.tail())
    print("DONE", Path(cfg["output_root"]) / "csv" / "ALL_RESULTS_SUMMARY.csv")


if __name__ == "__main__":
    main()
