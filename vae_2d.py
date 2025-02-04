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
from dgm.common.models import VAE

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="VAE training of LiDAR")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="size of minibatch used during training",
    )
    parser.add_argument(
        "--use_selu", type=int, default=0, help="replaces batch_norm + act with SELU"
    )
    parser.add_argument(
        "--base_dir", type=str, default="runs/test", help="root of experiment directory"
    )
    parser.add_argument(
        "--no_polar",
        type=int,
        default=0,
        help="if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate value")
    parser.add_argument(
        "--z_dim",
        type=int,
        default=128,
        help="size of the bottleneck dimension in the VAE, or the latent noise size in GAN",
    )
    parser.add_argument(
        "--autoencoder",
        type=int,
        default=1,
        help="if True, we do not enforce the KL regularization cost in the VAE",
    )
    parser.add_argument(
        "--atlas_baseline",
        type=int,
        default=0,
        help="If true, Atlas model used. Also determines the number of primitives used in the model",
    )
    parser.add_argument(
        "--panos_baseline",
        type=int,
        default=0,
        help="If True, Model by Panos Achlioptas used",
    )
    parser.add_argument(
        "--kl_warmup_epochs",
        type=int,
        default=150,
        help="number of epochs before fully enforcing the KL loss",
    )
    parser.add_argument("--debug", action="store_true")

    # ------------------------------------------------------------------------------
    args = parser.parse_args()
    utils.maybe_create_dir(args.base_dir)
    utils.print_and_save_args(args, args.base_dir)

    # the baselines are very memory heavy --> we split minibatches into mini-minibatches
    if args.atlas_baseline or args.panos_baseline:
        """ Tested on 12 Gb GPU for z_dim in [128, 256, 512] """
        bs = [4, 8 if args.atlas_baseline else 6][min(1, 511 // args.z_dim)]
        factor = args.batch_size // bs
        args.batch_size = bs
        is_baseline = True
        args.no_polar = 1
        print("using batch size of %d, ran %d times" % (bs, factor))
    else:
        factor, is_baseline = 1, False

    # reproducibility is good
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # construct model and ship to GPU
    model = VAE(args).cuda()

    # Logging
    utils.maybe_create_dir(os.path.join(args.base_dir, "samples"))
    utils.maybe_create_dir(os.path.join(args.base_dir, "models"))
    writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.base_dir, "TB"))
    writes = 0
    ns = 16

    # Create the datasets and the loaders.
    dataset_train = KittiDataset(
        dataset_file=os.path.join(root_dir, "kitti_data", "converted", "train.dataset"),
        debug=args.debug,
    )
    dataset_val = KittiDataset(
        dataset_file=os.path.join(root_dir, "kitti_data", "converted", "val.dataset"),
        debug=args.debug,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    print(model)
    model.apply(utils.weights_init)
    optim = optim.Adam(model.parameters(), lr=args.lr)

    # build loss function
    if args.atlas_baseline or args.panos_baseline:
        loss_fn = utils.get_chamfer_dist()
    else:
        loss_fn = lambda a, b: (a - b).abs().sum(-1).sum(-1).sum(-1)

    # VAE training
    # ------------------------------------------------------------------------------

    for epoch in range(150 if args.autoencoder else 300):
        print("##############################")
        print(f"Epoch {epoch:4d} / 1000")
        print("##############################")
        model.train()
        loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]

        # Output is always in polar, however input can be (X,Y,Z) or (D,Z)
        process_input = utils.from_polar if args.no_polar else lambda x: x

        for i, (img, _) in enumerate(train_loader):
            print(f"Iteration {i:3d} / {len(train_loader)}")
            img = img.cuda()
            recon, kl_cost = model(process_input(img))

            loss_recon = loss_fn(recon, img)

            if args.autoencoder:
                kl_obj, kl_cost = [torch.zeros_like(loss_recon)] * 2
            else:
                kl_obj = min(1, float(epoch + 1) / args.kl_warmup_epochs) * torch.clamp(
                    kl_cost, min=5
                )

            loss = (kl_obj + loss_recon).mean(dim=0)
            elbo = (kl_cost + loss_recon).mean(dim=0)

            loss_ += [loss.item()]
            elbo_ += [elbo.item()]
            kl_cost_ += [kl_cost.mean(dim=0).item()]
            kl_obj_ += [kl_obj.mean(dim=0).item()]
            recon_ += [loss_recon.mean(dim=0).item()]

            # baseline loss is very memory heavy
            # we accumulate gradient to simulate a bigger minibatch
            if (i + 1) % factor == 0 or not is_baseline:
                optim.zero_grad()

            loss.backward()
            if (i + 1) % factor == 0 or not is_baseline:
                optim.step()

        writes += 1
        mn = lambda x: np.mean(x)
        utils.print_and_log_scalar(writer, "train/loss", mn(loss_), writes)
        utils.print_and_log_scalar(writer, "train/elbo", mn(elbo_), writes)
        utils.print_and_log_scalar(writer, "train/kl_cost_", mn(kl_cost_), writes)
        utils.print_and_log_scalar(writer, "train/kl_obj_", mn(kl_obj_), writes)
        utils.print_and_log_scalar(writer, "train/recon", mn(recon_), writes)
        loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]

        # save some training reconstructions
        if epoch % 10 == 0:
            recon = recon[:ns].cpu().data.numpy()
            with open(
                os.path.join(args.base_dir, f"samples/train_{epoch}.npz"), "wb"
            ) as f:
                np.save(f, recon)

            print("saved training reconstructions")

        # Testing loop
        # --------------------------------------------------------------------------

        loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
        with torch.no_grad():
            model.eval()
            if epoch % 1 == 0:
                print("test set evaluation")
                for i_val, (img, _) in enumerate(val_loader):
                    print(f"Iteration {i_val:3d} / {len(val_loader)}")
                    img = img.cuda()
                    recon, kl_cost = model(process_input(img))

                    loss_recon = loss_fn(recon, img)

                    if args.autoencoder:
                        kl_obj, kl_cost = [torch.zeros_like(loss_recon)] * 2
                    else:
                        kl_obj = min(
                            1, float(epoch + 1) / args.kl_warmup_epochs
                        ) * torch.clamp(kl_cost, min=5)

                    loss = (kl_obj + loss_recon).mean(dim=0)
                    elbo = (kl_cost + loss_recon).mean(dim=0)

                    loss_ += [loss.item()]
                    elbo_ += [elbo.item()]
                    kl_cost_ += [kl_cost.mean(dim=0).item()]
                    kl_obj_ += [kl_obj.mean(dim=0).item()]
                    recon_ += [loss_recon.mean(dim=0).item()]

                utils.print_and_log_scalar(writer, "valid/loss", mn(loss_), writes)
                utils.print_and_log_scalar(writer, "valid/elbo", mn(elbo_), writes)
                utils.print_and_log_scalar(
                    writer, "valid/kl_cost_", mn(kl_cost_), writes
                )
                utils.print_and_log_scalar(writer, "valid/kl_obj_", mn(kl_obj_), writes)
                utils.print_and_log_scalar(writer, "valid/recon", mn(recon_), writes)
                loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]

                if epoch % 10 == 0:
                    with open(
                        os.path.join(args.base_dir, f"samples/test_{epoch}.npz"),
                        "wb",
                    ) as f:
                        recon = recon[:ns].cpu().data.numpy()
                        np.save(f, recon)
                        print("saved test recons")

                    sample = model.sample()
                    with open(
                        os.path.join(args.base_dir, f"samples/sample_{epoch}.npz"),
                        "wb",
                    ) as f:
                        sample = sample.cpu().data.numpy()
                        np.save(f, recon)

                    print("saved model samples")

                if epoch == 0:
                    with open(
                        os.path.join(args.base_dir, "samples/real.npz"), "wb"
                    ) as f:
                        img = img.cpu().data.numpy()
                        np.save(f, img)

                    print("saved real LiDAR")

        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.base_dir, f"models/gen_{epoch}.pth"),
            )
