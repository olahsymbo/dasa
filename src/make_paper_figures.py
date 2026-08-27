import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dasa import combine, estimate_difficulty
from src.datasets import build_dataset_factory
from src.models import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denorm_img(x):
    x = x.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(x, 0, 1)


def load_model(checkpoint_path, model_name, num_classes):
    model = build_model(model_name, num_classes).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_mask(model, image):
    x = image.unsqueeze(0).to(DEVICE)
    pred = model(x).argmax(1).squeeze(0).cpu()
    return pred


def save_difficulty_examples(
    dataset_name,
    data_root,
    image_size,
    model_name,
    checkpoint_path,
    output_dir,
    batch_size=8,
    num_workers=2,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base, _, train_aug_fn, _, num_classes = build_dataset_factory(
        dataset_name, data_root, image_size
    )

    model = load_model(checkpoint_path, model_name, num_classes)

    indices = list(range(len(base)))

    raw = estimate_difficulty(
        model=model,
        ds=base,
        ids=indices,
        bs=batch_size,
        c=num_classes,
        mc=4,
        device=DEVICE,
        num_workers=num_workers,
    )

    diff = combine(raw, "DASA_EM").sort_values("difficulty_z")

    picks = {
        "Low": diff.iloc[int(0.10 * len(diff))],
        "Medium": diff.iloc[int(0.50 * len(diff))],
        "High": diff.iloc[int(0.90 * len(diff))],
    }

    strengths = {
        int(row.original_idx): float(row.difficulty_z) for row in diff.itertuples()
    }

    aug_ds = train_aug_fn(adaptive=strengths, strong=True)

    fig, axes = plt.subplots(3, 4, figsize=(13, 9))

    for r, (level, row) in enumerate(picks.items()):
        idx = int(row.original_idx)
        score = float(row.difficulty_z)

        image, mask = base[idx]
        aug_image, aug_mask = aug_ds[idx]

        axes[r, 0].imshow(denorm_img(image))
        axes[r, 0].set_title(f"{level} difficulty\nscore={score:.3f}")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(mask, vmin=0, vmax=max(1, num_classes - 1))
        axes[r, 1].set_title("Mask")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(denorm_img(aug_image))
        axes[r, 2].set_title("Augmented image")
        axes[r, 2].axis("off")

        axes[r, 3].imshow(aug_mask, vmin=0, vmax=max(1, num_classes - 1))
        axes[r, 3].set_title("Augmented mask")
        axes[r, 3].axis("off")

    plt.tight_layout()
    out = output_dir / f"fig_difficulty_examples_{dataset_name}_{model_name}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


def save_qualitative_segmentation_panel(
    dataset_name,
    data_root,
    image_size,
    model_name,
    checkpoint_dir,
    output_dir,
    methods=("baseline", "loss_only", "boundary_only", "DASA_EM"),
    sample_indices=(0, 10, 25, 50),
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, test_ds, _, _, num_classes = build_dataset_factory(
        dataset_name, data_root, image_size
    )

    models = {}
    for method in methods:
        ckpt = Path(checkpoint_dir) / f"{dataset_name}_{model_name}_{method}.pt"
        if not ckpt.exists():
            print("Missing checkpoint:", ckpt)
            continue
        models[method] = load_model(ckpt, model_name, num_classes)

    if "baseline" not in models or "DASA_EM" not in models:
        print("Need at least baseline and DASA_EM checkpoints.")
        return

    n_rows = len(sample_indices)
    n_cols = 2 + len(models)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.0 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = ["Image", "Ground Truth"] + list(models.keys())

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title)

    for r, idx in enumerate(sample_indices):
        if idx >= len(test_ds):
            continue

        image, gt = test_ds[idx]

        axes[r, 0].imshow(denorm_img(image))
        axes[r, 0].axis("off")

        axes[r, 1].imshow(gt, vmin=0, vmax=max(1, num_classes - 1))
        axes[r, 1].axis("off")

        for c, (method, model) in enumerate(models.items(), start=2):
            pred = predict_mask(model, image)
            axes[r, c].imshow(pred, vmin=0, vmax=max(1, num_classes - 1))
            axes[r, c].axis("off")

    plt.tight_layout()
    out = output_dir / f"fig_qualitative_{dataset_name}_{model_name}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


def save_best_qualitative_panel_by_dasa_gain(
    dataset_name,
    data_root,
    image_size,
    model_name,
    checkpoint_dir,
    output_dir,
    methods=("baseline", "loss_only", "boundary_only", "DASA_EM"),
    top_k=4,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, test_ds, _, _, num_classes = build_dataset_factory(
        dataset_name, data_root, image_size
    )

    models = {}
    for method in methods:
        ckpt = Path(checkpoint_dir) / f"{dataset_name}_{model_name}_{method}.pt"
        if ckpt.exists():
            models[method] = load_model(ckpt, model_name, num_classes)

    if "baseline" not in models or "DASA_EM" not in models:
        print("Need baseline and DASA_EM for automatic selection.")
        return

    def sample_iou(pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        vals = []
        for c in range(num_classes):
            pi = pred == c
            ti = target == c
            inter = (pi & ti).sum().item()
            union = (pi | ti).sum().item()
            if union > 0:
                vals.append(inter / union)
        return float(np.mean(vals)) if vals else 0.0

    gains = []

    for idx in range(min(len(test_ds), 300)):
        image, gt = test_ds[idx]
        base_pred = predict_mask(models["baseline"], image)
        dasa_pred = predict_mask(models["DASA_EM"], image)

        base_iou = sample_iou(base_pred, gt)
        dasa_iou = sample_iou(dasa_pred, gt)

        gains.append((idx, dasa_iou - base_iou, base_iou, dasa_iou))

    gains = sorted(gains, key=lambda x: x[1], reverse=True)
    selected = [g[0] for g in gains[:top_k]]

    print("Selected qualitative samples:", gains[:top_k])

    save_qualitative_segmentation_panel(
        dataset_name=dataset_name,
        data_root=data_root,
        image_size=image_size,
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        methods=methods,
        sample_indices=selected,
    )


def save_result_bar_charts(csv_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Remove rows with errors if any
    if "error" in df.columns:
        df = df[df["error"].isna()]

    wanted = [
        "baseline",
        "strong_uniform",
        "random_weighted",
        "loss_only",
        "rarity_only",
        "boundary_only",
        "DASA_EM",
    ]

    df = df[df["method"].isin(wanted)]

    for dataset in sorted(df["dataset"].unique()):
        ddf = df[df["dataset"] == dataset]

        for model in sorted(ddf["model"].unique()):
            mdf = ddf[ddf["model"] == model].copy()

            if mdf.empty:
                continue

            mdf["method"] = pd.Categorical(
                mdf["method"],
                categories=wanted,
                ordered=True,
            )
            mdf = mdf.sort_values("method")

            plt.figure(figsize=(10, 5))
            plt.bar(mdf["method"].astype(str), mdf["mIoU"])
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("mIoU")
            plt.title(f"{dataset} / {model}: mIoU comparison")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            out = output_dir / f"fig_bar_miou_{dataset}_{model}.png"
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()
            print("Saved:", out)

            # Foreground chart for binary VOC
            if dataset == "voc_binary" and "class_1_IoU" in mdf.columns:
                plt.figure(figsize=(10, 5))
                plt.bar(mdf["method"].astype(str), mdf["class_1_IoU"])
                plt.xticks(rotation=35, ha="right")
                plt.ylabel("Foreground IoU")
                plt.title(f"{dataset} / {model}: foreground IoU comparison")
                plt.grid(axis="y", alpha=0.3)
                plt.tight_layout()

                out = output_dir / f"fig_bar_foreground_iou_{dataset}_{model}.png"
                plt.savefig(out, dpi=300, bbox_inches="tight")
                plt.close()
                print("Saved:", out)


def save_grouped_model_chart(csv_path, dataset, metric, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if "error" in df.columns:
        df = df[df["error"].isna()]

    wanted = [
        "baseline",
        "strong_uniform",
        "random_weighted",
        "loss_only",
        "rarity_only",
        "boundary_only",
        "DASA_EM",
    ]

    df = df[(df["dataset"] == dataset) & (df["method"].isin(wanted))]

    if df.empty:
        print("No rows for", dataset)
        return

    pivot = df.pivot_table(
        index="method",
        columns="model",
        values=metric,
        aggfunc="mean",
    )

    pivot = pivot.reindex(wanted)

    ax = pivot.plot(kind="bar", figsize=(12, 5))
    ax.set_ylabel(metric)
    ax.set_title(f"{dataset}: {metric} by method and model")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    out = output_dir / f"fig_grouped_{dataset}_{metric}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--dataset", default="pet")
    parser.add_argument("--model", default="deeplabv3")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--summary-csv", default="./outputs/csv/ALL_RESULTS_SUMMARY.csv"
    )
    args = parser.parse_args()

    seed_all(42)

    output_root = Path(args.output_root)
    fig_dir = output_root / "figures"
    ckpt_dir = output_root / "checkpoints"

    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Difficulty-based augmentation examples
    dasa_ckpt = ckpt_dir / f"{args.dataset}_{args.model}_DASA_EM.pt"
    if dasa_ckpt.exists():
        save_difficulty_examples(
            dataset_name=args.dataset,
            data_root=args.data_root,
            image_size=args.image_size,
            model_name=args.model,
            checkpoint_path=dasa_ckpt,
            output_dir=fig_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        print("Missing DASA checkpoint for difficulty figure:", dasa_ckpt)

    # 2. Qualitative segmentation results
    save_best_qualitative_panel_by_dasa_gain(
        dataset_name=args.dataset,
        data_root=args.data_root,
        image_size=args.image_size,
        model_name=args.model,
        checkpoint_dir=ckpt_dir,
        output_dir=fig_dir,
        methods=("baseline", "loss_only", "boundary_only", "DASA_EM"),
        top_k=4,
    )

    # 3. Bar charts from CSV
    if Path(args.summary_csv).exists():
        save_result_bar_charts(args.summary_csv, fig_dir)

        for dataset in ["pet", "voc_binary"]:
            save_grouped_model_chart(
                csv_path=args.summary_csv,
                dataset=dataset,
                metric="mIoU",
                output_dir=fig_dir,
            )

        save_grouped_model_chart(
            csv_path=args.summary_csv,
            dataset="voc_binary",
            metric="class_1_IoU",
            output_dir=fig_dir,
        )
    else:
        print("Missing summary CSV:", args.summary_csv)


if __name__ == "__main__":
    main()
