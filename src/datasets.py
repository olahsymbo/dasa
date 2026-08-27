import random

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet, VOCSegmentation
from torchvision.transforms import functional as TF


class PetSegDataset(Dataset):
    def __init__(
        self,
        root,
        split="trainval",
        image_size=128,
        augment=False,
        strong=False,
        adaptive_strengths=None,
    ):
        self.ds = OxfordIIITPet(
            root=root, split=split, target_types="segmentation", download=True
        )
        self.image_size = image_size
        self.augment = augment
        self.strong = strong
        self.adaptive_strengths = adaptive_strengths

    def __len__(self):
        return len(self.ds)

    def _mask(self, mask):
        m = np.array(mask).astype(np.uint8)
        out = np.zeros_like(m, dtype=np.uint8)
        out[m == 3] = 0
        out[m == 1] = 1
        out[m == 2] = 2
        return Image.fromarray(out)

    def _strength(self, idx):
        return (
            float(self.adaptive_strengths.get(int(idx), 0.5))
            if self.adaptive_strengths is not None
            else (1.0 if self.strong else 0.5)
        )

    def _aug(self, image, mask, s):
        if random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        p = min(1.0, 0.2 + 0.8 * s)
        if random.random() < p:
            a = random.uniform(-5 - 20 * s, 5 + 20 * s)
            image = TF.rotate(image, a, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, a, interpolation=TF.InterpolationMode.NEAREST)
        if random.random() < p:
            mt = int(10 * s)
            tx = random.randint(-mt, mt)
            ty = random.randint(-mt, mt)
            image = TF.affine(
                image,
                0,
                [tx, ty],
                1.0,
                [0.0, 0.0],
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            mask = TF.affine(
                mask,
                0,
                [tx, ty],
                1.0,
                [0.0, 0.0],
                interpolation=TF.InterpolationMode.NEAREST,
            )
        if random.random() < p:
            image = TF.adjust_brightness(
                image, random.uniform(1 - 0.35 * s, 1 + 0.35 * s)
            )
            image = TF.adjust_contrast(
                image, random.uniform(1 - 0.35 * s, 1 + 0.35 * s)
            )
            image = TF.adjust_saturation(
                image, random.uniform(1 - 0.35 * s, 1 + 0.35 * s)
            )
        return image, mask

    def __getitem__(self, idx):
        image, mask = self.ds[idx]
        image = image.convert("RGB")
        mask = self._mask(mask)
        image = TF.resize(image, (self.image_size, self.image_size))
        mask = TF.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=TF.InterpolationMode.NEAREST,
        )
        if self.augment:
            image, mask = self._aug(image, mask, self._strength(idx))
        return TF.to_tensor(image), torch.as_tensor(np.array(mask), dtype=torch.long)


class VOCBinarySegDataset(Dataset):
    def __init__(
        self,
        root,
        image_set="train",
        image_size=128,
        augment=False,
        strong=False,
        adaptive_strengths=None,
    ):
        self.ds = VOCSegmentation(
            root=root, year="2012", image_set=image_set, download=True
        )
        self.image_size = image_size
        self.augment = augment
        self.strong = strong
        self.adaptive_strengths = adaptive_strengths

    def __len__(self):
        return len(self.ds)

    def _mask(self, mask):
        m = np.array(mask).astype(np.uint8)
        out = np.zeros_like(m, dtype=np.uint8)
        out[(m > 0) & (m < 255)] = 1
        out[m == 255] = 0
        return Image.fromarray(out)

    def _strength(self, idx):
        return (
            float(self.adaptive_strengths.get(int(idx), 0.5))
            if self.adaptive_strengths is not None
            else (1.0 if self.strong else 0.5)
        )

    def _aug(self, image, mask, s):
        if random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        p = min(1.0, 0.2 + 0.8 * s)
        if random.random() < p:
            a = random.uniform(-5 - 15 * s, 5 + 15 * s)
            image = TF.rotate(image, a, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, a, interpolation=TF.InterpolationMode.NEAREST)
        if random.random() < p:
            image = TF.adjust_brightness(
                image, random.uniform(1 - 0.25 * s, 1 + 0.25 * s)
            )
            image = TF.adjust_contrast(
                image, random.uniform(1 - 0.25 * s, 1 + 0.25 * s)
            )
        return image, mask

    def __getitem__(self, idx):
        image, mask = self.ds[idx]
        image = image.convert("RGB")
        mask = self._mask(mask)
        image = TF.resize(image, (self.image_size, self.image_size))
        mask = TF.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=TF.InterpolationMode.NEAREST,
        )
        if self.augment:
            image, mask = self._aug(image, mask, self._strength(idx))
        return TF.to_tensor(image), torch.as_tensor(np.array(mask), dtype=torch.long)


def build_dataset_factory(name, data_root, image_size):
    if name == "pet":
        base = PetSegDataset(data_root, "trainval", image_size, False)
        test = PetSegDataset(data_root, "test", image_size, False)
        return (
            base,
            test,
            lambda adaptive=None, strong=False: PetSegDataset(
                data_root, "trainval", image_size, True, strong, adaptive
            ),
            lambda: PetSegDataset(data_root, "trainval", image_size, False),
            3,
        )
    if name == "voc_binary":
        base = VOCBinarySegDataset(data_root, "train", image_size, False)
        test = VOCBinarySegDataset(data_root, "val", image_size, False)
        return (
            base,
            test,
            lambda adaptive=None, strong=False: VOCBinarySegDataset(
                data_root, "train", image_size, True, strong, adaptive
            ),
            lambda: VOCBinarySegDataset(data_root, "train", image_size, False),
            2,
        )
    raise ValueError(name)
