import os

import numpy as np
import pandas as pd
import skimage
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class FairvisionDataset(VisionDataset):
    """
    FairVision Dataset
    Returns gray-scale SLO funduns images and labels
    """

    def __init__(self, csv_file, root_dir, stage="training", transform=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["use"] == stage]
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

    @staticmethod
    def crop_center(img, cropx, cropy):
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty : starty + cropy, startx : startx + cropx]

    @staticmethod
    def crop_image(img, tol=-2):
        # img is 2D image data
        # tol  is tolerance
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.df.loc[self.df.index[idx], "filename"]
        img_name = os.path.join(
            self.root_dir,
            self.df.loc[self.df.index[idx], "use"].capitalize(),
            filename,
        )
        image = np.load(img_name)["slo_fundus"]
        if len(image.shape) < 3:
            image = skimage.color.gray2rgb(image)
        image = Image.fromarray((image).astype(np.uint8))

        label = self.df.loc[self.df.index[idx], "label"]
        if self.transform:
            image = self.transform(image)

        return image, label


class AMD(FairvisionDataset):
    def __init__(self, root_dir, stage="train", transform=None):
        self.stage = self._stage_map_(stage)
        csv_file = os.path.join(
            root_dir, "AMD", "ReadMe", "data_summary_amd.csv"
        )
        data_dir = os.path.join(root_dir, "AMD", "Dataset")
        super().__init__(csv_file, data_dir, self.stage, transform)
        label_map = {
            "normal": 0,
            "early amd": 1,
            "intermediate amd": 2,
            "late amd": 3,
        }
        self.df["label"] = self.df["amd"].map(label_map)

    @property
    def classes(self):
        return [
            "no age-related macular degeneration",
            "early age-related macular degeneration",
            "intermediate age-related macular degeneration",
            "late age-related macular degeneration",
        ]


class DR(FairvisionDataset):
    def __init__(self, root_dir, stage="train", transform=None):
        self.stage = self._stage_map_(stage)
        csv_file = os.path.join(
            root_dir, "DR", "ReadMe", "data_summary_dr.csv"
        )
        data_dir = os.path.join(root_dir, "DR", "Dataset")
        super().__init__(csv_file, data_dir, self.stage, transform)
        label_map = {
            "non-vision threatening dr": 0,
            "vision threatening dr": 1,
        }
        self.df["label"] = self.df["dr"].map(label_map)

    @property
    def classes(self):
        return [
            "no vision threatening diabetic retinopathy",
            "vision threatening diabetic retinopathy",
        ]


class Glaucoma(FairvisionDataset):
    def __init__(self, root_dir, stage="train", transform=None):
        self.stage = self._stage_map_(stage)
        csv_file = os.path.join(
            root_dir, "Glaucoma", "ReadMe", "data_summary_glaucoma.csv"
        )
        data_dir = os.path.join(root_dir, "Glaucoma", "Dataset")
        super().__init__(csv_file, data_dir, self.stage, transform)
        label_map = {
            "no": 0,
            "yes": 1,
        }
        self.df["label"] = self.df["glaucoma"].map(label_map)

    @property
    def classes(self):
        return ["without glaucoma", "with glaucoma"]
