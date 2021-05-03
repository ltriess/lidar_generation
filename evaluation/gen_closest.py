# Code to find the closest example w.r.t EMD in the training data
# For now let's only do MSE, as EMD is long and painful.

import argparse
import os
import pdb

import numpy as np
import torch.utils.data
from kitti_loader import Kitti

import utils

parser = argparse.ArgumentParser(
    description="Find Closest Neighbor of a set of samples"
)
parser.add_argument("--gen_path", type=str, default="../trained_models/uncond_gan")
parser.add_argument("--output_path", type=str, default="../samples_gen_nn_new")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--path_to_dis", type=str, default="../trained_models/uncond_gan")
parser.add_argument(
    "--sample_indices", type=int, nargs="+", default=[11, 174, 208, 245, 548]
)
args = parser.parse_args()

# args validation
utils.maybe_create_dir(args.output_path)

# load gen
gen = utils.load_model_from_file(args.gen_path, 999, model="gen")[0].cuda()
dis = utils.load_model_from_file(args.gen_path, 999, model="dis")[0].cuda()

# load target dataset
# dataset = np.load('kitti_data/lidar.npz')[::100] #args.sample_indices]
# dataset = preprocess(dataset).astype('float32')

kitti_ds = Kitti(n_points_post=256)

pdb.set_trace()
indices = range(0, len(kitti_ds), len(kitti_ds) // 1000)
dataset = [kitti_ds.__getitem__(i) for i in indices]
dataset = torch.cuda.FloatTensor(np.stack(dataset))

learned_z = torch.randn(dataset.size(0), 100).cuda()
learned_z.requires_grad = True
optim = torch.optim.Adam([learned_z])

hidden_real = dis(dataset, return_hidden=True)[-1].detach()
for i in range(2000):
    print(f"{i} / 2000")
    fake = gen(learned_z)
    hidden_fake = dis(fake, return_hidden=True)[-1]

    loss = ((hidden_real - hidden_fake) ** 2).sum()

    optim.zero_grad()
    loss.backward()
    optim.step()


# save samples
dataset = utils.from_polar(dataset).cpu().data.numpy()
fake = utils.from_polar(fake).cpu().data.numpy()

for i in range(fake.shape[0]):
    real = dataset[i]
    est = fake[i]

    # save picture of source and target
    utils.show_pc(
        est.squeeze(),
        save_path=os.path.join(args.output_path, "%d_real.png" % indices[i]),
    )
    utils.show_pc(
        real.squeeze(),
        save_path=os.path.join(args.output_path, "%d_target.png" % indices[i]),
    )
