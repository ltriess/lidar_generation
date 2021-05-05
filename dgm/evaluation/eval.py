#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import torch.utils.data
from emd import EMD

from dgm.common import utils
from dgm.common.loader import KittiDataset

"""
Expect two arguments:
    1) path_to_model_folder
    2) epoch of model you wish to load
    3) metric to evaluate on
e.g. python eval.py runs/test_baseline 149 emd
"""

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

nb_samples = 200
out_dir = os.path.join(sys.argv[1], "final_samples")
utils.maybe_create_dir(out_dir)
size = 10 if "emd" in sys.argv[3] else 5
fast = True

# fetch metric
if "emd" in sys.argv[3]:
    loss = EMD
elif "chamfer" in sys.argv[3]:
    loss = utils.get_chamfer_dist
else:
    raise ValueError(
        f"{sys.argv[2]} is not a valid metric for point cloud eval. Either 'emd' or 'chamfer'"
    )

with torch.no_grad():

    # 1) load trained model
    model = utils.load_model_from_file(
        sys.argv[1], epoch=int(sys.argv[2]), model="gen"
    )[0]
    model = model.cuda()
    model.eval()
    if "panos" in sys.argv[1] or "atlas" in sys.argv[1]:
        model.args.no_polar = 1

    # 2) load data
    print("test set reconstruction")
    # Create the dataset and the loader.
    dataset_test = KittiDataset(
        dataset_file=os.path.join(root_dir, "kitti_data", "converted", "test.dataset"),
        debug=fast,
    )
    loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=size, shuffle=True, num_workers=4, drop_last=True
    )

    loss_fn = loss()
    process_input = (lambda x: x) if model.args.no_polar else utils.to_polar

    # noisy reconstruction
    for (
        noise
    ) in (
        []
    ):  # 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.][::(2 if fast else 1)]:
        losses = []
        for batch, _ in loader:
            batch = batch.cuda()
            batch_xyz = utils.from_polar(batch)
            noise_tensor = torch.zeros_like(batch_xyz).normal_(0, noise)

            means = batch_xyz.transpose(1, 0).reshape((3, -1)).mean(dim=-1)
            stds = batch_xyz.transpose(1, 0).reshape((3, -1)).std(dim=-1)
            means, stds = [
                x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) for x in [means, stds]
            ]

            # normalize data
            norm_batch_xyz = (batch_xyz - means) / (stds + 1e-9)
            # add the noise
            inp = norm_batch_xyz + noise_tensor

            # unnormalize
            inp = inp * (stds + 1e-9) + means

            recon = model(process_input(inp))[0]
            recon_xyz = utils.from_polar(recon)

            losses += [loss_fn(recon_xyz, batch_xyz)]

        losses = torch.stack(losses).mean().item()
        print(f"{sys.argv[3]} with noise {noise} : {losses:.4f}")

        del inp, recon, recon_xyz, losses

    process_input = utils.from_polar if model.args.no_polar else (lambda x: x)

    # missing reconstruction
    for missing in [
        0.97,
        0.98,
        0.99,
        0.999,
    ]:  # [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45][::(2 if fast else 1)]:
        losses = []
        for batch, _ in loader:
            batch = batch.cuda()
            batch_xyz = utils.from_polar(batch)

            is_present = (
                torch.zeros_like(batch[:, [0]]).uniform_(0, 1) + (1 - missing)
            ).floor()
            inp = batch * is_present

            # SMOOTH OUT ZEROS
            if missing > 0:
                inp = torch.Tensor(utils.remove_zeros(inp)).float().cuda()

            recon = model(process_input(inp))[0]
            recon_xyz = utils.from_polar(recon)

            # TODO: remove this
            # recon_xyz[:, 0].uniform_(batch_xyz[:, 0].min(), batch_xyz[:, 0].max())
            # recon_xyz[:, 1].uniform_(batch_xyz[:, 1].min(), batch_xyz[:, 1].max())
            # recon_xyz[:, 2].uniform_(batch_xyz[:, 2].min(), batch_xyz[:, 2].max())

            losses += [loss_fn(recon_xyz, batch_xyz)]

        losses = torch.stack(losses).mean().item()
        print(f"{sys.argv[3]} with missing p {missing} : {losses:.4f}")
