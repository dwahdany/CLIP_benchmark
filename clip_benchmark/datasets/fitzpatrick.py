import os

import numpy as np
import pandas as pd
import skimage
import torch
from PIL import Image
from skimage import io
from sklearn.model_selection import train_test_split
from torchvision.datasets.vision import VisionDataset


class FitzpatrickDataset(VisionDataset):
    """
    Fitzpatrick 17k Dataset
    """

    def __init__(
        self, root_dir, stage="training", transform=None, seed: int = 4242
    ):
        self.seed = seed
        df = pd.read_csv(os.path.join(root_dir, "fitzpatrick17k.csv"))
        df["low"] = df["label"].astype("category").cat.codes
        df["mid"] = df["nine_partition_label"].astype("category").cat.codes
        df["high"] = df["three_partition_label"].astype("category").cat.codes
        df["hasher"] = df["md5hash"]

        self.in_label = "three_partition_label"
        self.out_label = "high"

        train = df[df.qc.isnull()]
        test = df[df.qc == "1 Diagnostic"]

        train, val = train_test_split(
            train,
            test_size=0.2,
            random_state=seed,
            stratify=train[self.out_label],
        )
        if stage.startswith("train"):
            self.df = train
        elif stage.startswith("val"):
            self.df = val
        elif stage == "test":
            self.df = test
        else:
            raise ValueError(f"Invalid stage: {stage}")

        self.root_dir = root_dir
        self.transform = transform

    @staticmethod
    def _stage_map_(stage):
        match stage.lower():
            case s if s.startswith("train"):
                return "training"
            case s if s.startswith("val"):
                return "validation"
            case "test":
                return "test"
            case _:
                return stage

    @property
    def classes(self):
        return [*self.df[self.in_label].astype("category").cat.categories]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(
            self.root_dir,
            self.df.loc[self.df.index[idx], "hasher"] + ".jpg",
        )
        image = io.imread(img_name)
        if len(image.shape) < 3:
            image = skimage.color.gray2rgb(image)
        image = Image.fromarray((image).astype(np.uint8))

        label = self.df.loc[self.df.index[idx], self.out_label]
        if self.transform:
            image = self.transform(image)

        return image, label
