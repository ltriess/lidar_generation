#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Code to find the closest example w.r.t EMD in the training common
# For now let's only do MSE, as EMD is long and painful.

import argparse
import os

import numpy as np
import torch.utils.data

from dgm.common import utils
from dgm.common.loader import KittiDataset

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(
    description="Find Closest Neighbor of a set of samples"
)
parser.add_argument(
    "--sample_path",
    type=str,
    default="/mnt/data/lpagec/lidar_generation/gan_classic/Conv0_Selu0_SN1_Loss0_BS32_OPTrmspropGLR:0.0001_DLR:0.0001XYZ:0/final_samples/undond_inter.npy",
)
parser.add_argument("--output_path", type=str, default="samples_nn")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument(
    "--sample_indices", type=int, nargs="+", default=[11, 174, 208, 245, 548]
)
parser.add_argument("--path_to_dis", type=str, default="trained_models/uncond_gan")
parser.add_argument(
    "--metric",
    choices=["dis", "emd", "mse"],
    default="dis",
    help="dis: closest image in disriminator latent space "
    + "emd: earth movers distance for point clouds",
)
args = parser.parse_args()

# args validation
assert args.metric != "dis" or args.path_to_dis is not None
assert args.metric != "emd", "Not supported yet"
utils.maybe_create_dir(args.output_path)

# load samples
target_pcs = np.load(args.sample_path)

# Create the target dataset and the loader.
dataset_train = KittiDataset(
    dataset_file=os.path.join(root_dir, "kitti_data", "converted", "train.dataset"),
    use_xyz=target_pcs.shape[1] == 3,
)

train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    drop_last=True,
)

# build network if needed
if args.metric == "dis":
    dis = utils.load_model_from_file(args.path_to_dis, 999, model="dis")[0]
    dis = dis.cuda()

# iterate over samples
if args.sample_indices is None or -1 in args.sample_indices:
    args.sample_indices = np.arange(target_pcs.shape[0])

for index in args.sample_indices:
    target = target_pcs[index]

    assert len(target.shape) == 3 and target.shape[0] in [2, 3]

    target = torch.cuda.FloatTensor(target).unsqueeze(0)
    best_val = 1e10
    best_ind = -1

    # iterate over "real" dataset
    for i, (batch, _) in enumerate(train_loader):
        batch = batch.cuda()

        if args.metric == "mse":
            diff = ((target - batch) ** 2).sum(-1).sum(-1).sum(-1, keepdim=True)
            curr_val, curr_ind = diff.min(dim=0)
        elif args.metric == "dis":
            hidden_real = dis(batch, return_hidden=True)[-1]
            hidden_fake = dis(target, return_hidden=True)[-1]
            diff = (
                ((hidden_real - hidden_fake) ** 2).sum(-1).sum(-1).sum(-1, keepdim=True)
            )
            curr_val, curr_ind = diff.min(dim=0)
        elif args.metric == "emd":
            pass

        if curr_val < best_val:
            best_val = curr_val.data[0].item()
            best_ind = i * args.batch_size + curr_ind.data[0].item()
            # print('best val {:.4f}, best ind {}'.format(best_val, best_ind))

    # instead of save the point clouds, we just save the picture of the point cloud.
    print(f"closest point cloud has MSE {best_val:.4f} with pc @ index {index}")
    closest = dataset_train[best_ind]
    target = target.cpu().data.numpy()

    if closest.shape[1] == 2:
        closest = utils.from_polar_np(closest)
        target = utils.from_polar_np(target)

    # save picture of source and target
    utils.show_pc(
        closest.squeeze(),
        save_path=os.path.join(args.output_path, "%d_real_inter.png" % index),
    )
    utils.show_pc(
        target.squeeze(),
        save_path=os.path.join(args.output_path, "%d_target_inter.png" % index),
    )
