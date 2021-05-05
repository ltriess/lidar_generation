#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import tensorboardX
import torch.optim as optim
import torch.utils.data

from dgm.common import utils
from dgm.common.loader import KittiDataset
from dgm.common.models import netD, netG

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="GAN training of LiDAR")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--loss", type=int, help="0 == LSGAN, 1 == RaLSGAN")
    parser.add_argument(
        "--use_selu", type=int, default=0, help="replaces batch_norm + act with SELU"
    )
    parser.add_argument("--use_spectral_norm", type=int, default=0)
    parser.add_argument("--use_round_conv", type=int, default=0)
    parser.add_argument("--base_dir", type=str, default="runs/test")
    parser.add_argument(
        "--dis_iters", type=int, default=1, help="disc iterations per 1 gen iter"
    )
    parser.add_argument("--no_polar", type=int, default=0)
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--gen_lr", type=float, default=1e-4)
    parser.add_argument("--dis_lr", type=float, default=1e-4)

    args = parser.parse_args()
    utils.maybe_create_dir(args.base_dir)
    utils.print_and_save_args(args, args.base_dir)

    # reproducibility is good
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # construct model and ship to GPU
    dis = netD(args, ndf=64, nc=3 if args.no_polar else 2, lf=(2, 16)).cuda()
    gen = netG(args, ngf=64, nc=3 if args.no_polar else 2, ff=(2, 16)).cuda()

    # Logging
    utils.maybe_create_dir(os.path.join(args.base_dir, "samples"))
    utils.maybe_create_dir(os.path.join(args.base_dir, "models"))
    writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.base_dir, "TB"))
    writes = 0

    # Create the dataset and the loader.
    dataset = KittiDataset(
        dataset_file=os.path.join(root_dir, "kitti_data", "converted", "train.dataset"),
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    print(gen)
    print(dis)

    gen.apply(utils.weights_init)
    dis.apply(utils.weights_init)

    if args.optim.lower() == "adam":
        gen_optim = optim.Adam(
            gen.parameters(), lr=args.gen_lr, betas=(0.5, 0.999), weight_decay=0
        )
        dis_optim = optim.Adam(
            dis.parameters(), lr=args.dis_lr, betas=(0.5, 0.999), weight_decay=0
        )
    elif args.optim.lower() == "rmsprop":
        gen_optim = optim.RMSprop(gen.parameters(), lr=args.gen_lr)
        dis_optim = optim.RMSprop(dis.parameters(), lr=args.dis_lr)

    # gan training
    # ------------------------------------------------------------------------------------------------
    for epoch in range(1000):
        print("##############################")
        print(f"Epoch {epoch:4d} / 1000")
        print("##############################")
        data_iter = iter(loader)
        iters = 0
        real_d, fake_d, fake_g, losses_g, losses_d, delta_d = [[] for _ in range(6)]

        while iters < len(loader):
            print(f"Iteration {iters:3d} / {len(loader)}")
            j = 0

            """ Update Discriminator Network """
            for p in dis.parameters():
                p.requires_grad = True

            while j < args.dis_iters and iters < len(loader):
                j += 1
                iters += 1

                inp, _ = next(data_iter)
                inp = inp.cuda()

                # train with real data
                real_out = dis(inp)
                real_d += [real_out.mean().data]

                # train with fake data
                noise = torch.cuda.FloatTensor(args.batch_size, 100).normal_()
                fake = gen(noise)
                fake_out = dis(fake)
                fake_d += [fake_out.mean().data]

                if args.loss == 0:
                    dis_loss = (
                        ((real_out - fake_out.mean() - 1) ** 2).mean()
                        + ((fake_out - real_out.mean() + 1) ** 2).mean()
                    ) / 2
                else:
                    dis_loss = (
                        torch.mean((real_out - 1) ** 2)
                        + torch.mean((fake_out - 0) ** 2)
                    ) / 2

                losses_d += [dis_loss.mean().data]
                delta_d += [(real_out.mean() - fake.mean()).data]

                dis_optim.zero_grad()
                dis_loss.backward()
                dis_optim.step()

            """ Update Generator network """
            for p in dis.parameters():
                p.requires_grad = False

            noise = torch.cuda.FloatTensor(args.batch_size, 100).normal_()
            fake = gen(noise)
            fake_out = dis(fake)
            fake_g += [fake_out.mean().data]

            if args.loss == 0:
                iters += 1
                inp = data_iter.next().cuda() if iters < len(loader) else inp
                real_out = dis(inp)
                gen_loss = (
                    ((real_out - fake_out.mean() + 1) ** 2).mean()
                    + ((fake_out - real_out.mean() - 1) ** 2).mean()
                ) / 2
            else:
                gen_loss = torch.mean((fake_out - 1.0) ** 2)

            losses_g += [gen_loss.data]

            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()

        utils.print_and_log_scalar(writer, "real_out", real_d, writes)
        utils.print_and_log_scalar(writer, "fake_out", fake_d, writes)
        utils.print_and_log_scalar(writer, "fake_out_g", fake_g, writes)
        utils.print_and_log_scalar(writer, "delta_d", delta_d, writes)
        utils.print_and_log_scalar(writer, "losses_gen", losses_g, writes)
        utils.print_and_log_scalar(writer, "losses_dis", losses_d, writes)
        writes += 1

        # save some training reconstructions
        fake = fake[:20].cpu().data.numpy()
        with open(os.path.join(args.base_dir, f"samples/{epoch}.npz"), "wb") as f:
            np.save(f, fake)

        if (epoch + 1) % 10 == 0:
            torch.save(
                gen.state_dict(),
                os.path.join(args.base_dir, f"models/gen_{epoch}.pth"),
            )
            torch.save(
                dis.state_dict(),
                os.path.join(args.base_dir, f"models/dis_{epoch}.pth"),
            )
            print("saved models")
