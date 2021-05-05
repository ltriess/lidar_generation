#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Larissa Triess"
__email__ = "mail@triess.eu"

import numpy as np
import torch

from dgm.common import utils


class KittiDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_file: str,
        normalize: bool = True,
        use_xyz: bool = False,
        debug: bool = False,
    ):
        super().__init__()

        with open(dataset_file, "r") as fin:
            self.filenames = [s.strip() for s in fin.readlines()]
        if debug:
            self.filenames = self.filenames[: min(128, len(self.filenames))]
        self.size = len(self.filenames)

        self.use_norm = normalize
        self.use_xyz = use_xyz

        self.d_min, self.d_max = 0.0, 50.0
        self.z_min, self.z_max = -2.5, 1.0

    def denormalize(self, x_norm):
        d_norm = x_norm[0]
        z_norm = x_norm[1]

        # Revert normalization.
        d = d_norm * (self.d_max - self.d_min) + self.d_min
        z = z_norm * (self.z_max - self.z_min) + self.z_min

        # Stack channels and return.
        return np.stack([d, z], axis=0)

    def normalize(self, x):
        d = x[0]
        z = x[1]

        # Clip the values in range.
        d = np.clip(d, self.d_min, self.d_max)
        z = np.clip(z, self.z_min, self.z_max)

        # Normalize in [0, 1]
        d_norm = (d - self.d_min) / (self.d_max - self.d_min)
        z_norm = (z - self.z_min) / (self.z_max - self.z_min)

        # Stack channels and return.
        return np.stack([d_norm, z_norm], axis=0)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Load from file.
        item = np.load(self.filenames[idx])
        # Normalize into [0, 1].
        if self.use_norm:
            item = self.normalize(item)
        # Convert Polar to Cartesian.
        if self.use_xyz:
            item = utils.from_polar_np(item)
        # Remove every other column (512 -> 256).
        item = item[..., ::2]

        return item, idx
