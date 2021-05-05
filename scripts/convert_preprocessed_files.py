#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Larissa Triess"
__email__ = "mail@triess.eu"

import argparse
import os
from typing import Union

import numpy as np


def generate_filename(
    out_dir: str, region: str, recording: str, idx: Union[int, str]
) -> str:
    return os.path.join(out_dir, region, recording, f"{int(idx):06d}.npy")


def get_filenames(out_dir: str, region: str, recordings: list) -> list:
    filenames = []
    for r, c in recordings:
        filenames.extend(
            [f"{generate_filename(out_dir, region, r, i)}\n" for i in range(c)]
        )
    return filenames


def main(root_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    env_recs = {}
    for region in os.listdir(root_dir):
        if region not in {"city", "residential", "road"}:
            continue

        print(region)
        env_recs[region] = []
        for recording in os.listdir(os.path.join(root_dir, region)):
            path = os.path.join(root_dir, region, recording, "processed.npz")
            os.makedirs(os.path.join(out_dir, region, recording), exist_ok=True)

            print(recording)
            data = np.load(path)
            env_recs[region].append((recording, len(data)))
            for idx, scan in data.items():
                fname = generate_filename(out_dir, region, recording, idx)
                np.save(fname, scan)

    with open(os.path.join(out_dir, "train.dataset"), "w") as fout:
        fout.writelines(get_filenames(out_dir, "road", env_recs["road"][2:]))
        fout.writelines(
            get_filenames(out_dir, "residential", env_recs["residential"][3:])
        )
        fout.writelines(get_filenames(out_dir, "city", env_recs["city"][2:]))

    with open(os.path.join(out_dir, "val.dataset"), "w") as fout:
        fout.writelines(get_filenames(out_dir, "road", env_recs["road"][:2]))
        fout.writelines(
            get_filenames(out_dir, "residential", env_recs["residential"][:3])
        )
        fout.writelines(get_filenames(out_dir, "city", env_recs["city"][:2]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        "-d",
        type=str,
        required=True,
        help="Path to the root of the processed KITTI data.",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        required=True,
        help="Path to the root of the output dataset.",
    )
    args, _ = parser.parse_known_args()

    main(
        root_dir=os.path.abspath(os.path.expanduser(args.root_dir)),
        out_dir=os.path.abspath(os.path.expanduser(args.out_dir)),
    )
    print("DONE")
