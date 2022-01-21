# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
import itertools
import utils
import copy
import vision_transformer as vits


def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    # 输出路径
    from polyaxon_client.tracking import get_outputs_path
    args.output_dir = get_outputs_path()

    # 数据路径
    from polyaxon_client.tracking import get_data_paths
    source_data = 'ImageNet'
    args.data_path = os.path.join(get_data_paths()['ceph'], source_data.strip('/'))

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    model.cuda()
    model.eval()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)

    args.lrs = [base * n for base in [10 ** k for k in range(-4, 1)] for n in range(1, 10)]
    # [0.0001, 0.0002, 0.00030000000000000003, 0.0004, 0.0005, 0.0006000000000000001, 0.0007, 0.0008, 0.0009000000000000001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.0090000000000
    # 00001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if not args.sweep_lr_only:
        args.wds = [0, 1e-6]
        args.optims = ['sgd', 'lars']
    else:
        args.wds = [0]
        args.optims = ['sgd']
    args.permutes = list(itertools.product(args.lrs, args.wds, args.optims))

    linear_classifiers = nn.ModuleList()
    optimizers = []
    schedulers = []
    for pm in args.permutes:
        lr, wd, optim = pm
        linear_classifier = LinearClassifier(model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens)), num_labels=args.num_labels)
        linear_classifier = linear_classifier.cuda()
        linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])
        linear_classifiers.append(linear_classifier)

        # set optimizer
        parameters = linear_classifier.parameters()
        optimizer = torch.optim.SGD if optim == 'sgd' else utils.LARS
        optimizer = optimizer(
            parameters,
            lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
            momentum=0.9,
            weight_decay=wd,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

        optimizers.append(optimizer)
        schedulers.append(scheduler)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            state_dict=linear_classifiers)
        for optimizer, scheduler in zip(optimizers, schedulers):
            utils.restart_from_checkpoint(
                os.path.join(args.output_dir, args.load_from),
                optimizer=optimizer,
                scheduler=scheduler)
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    best_acc_sweep_lr_only = to_restore["best_acc"]


    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        linear_classifiers.train()
        train_stats = train(model, linear_classifiers, optimizers, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens, args.permutes)
        for scheduler in schedulers:
            scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            linear_classifiers.eval()
            test_stats = validate_network(val_loader, model, linear_classifiers, args.n_last_blocks, args.avgpool_patchtokens, args.permutes)

            group_best_acc = 0
            group_best_acc_hidx = 0
            group_best_acc_sweep_lr_only = 0
            for group, pm in enumerate(args.permutes):
                lr, wd, optim = pm
                # print(f"Accuracy at epoch {epoch} with lr {lr:.5f} wd {wd:.0e} optim {optim:4} of the network \
                #         on the {len(dataset_val)} test images: {test_stats['acc{}@1'.format(group)]:.1f}%")
                if group % (len(args.wds) * len(args.optims)) == 0:
                    group_best_acc_sweep_lr_only = max(group_best_acc_sweep_lr_only,
                                                       test_stats['acc{}@1'.format(group)])
                # group_best_acc = max(group_best_acc, test_stats['acc{}@1'.format(group)])
                if test_stats['acc{}@1'.format(group)] >= group_best_acc:
                    group_best_acc_hidx = group
                    group_best_acc = test_stats['acc{}@1'.format(group)]

            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}

            if utils.is_main_process() and (group_best_acc >= best_acc):
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifiers.state_dict(),
                    "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                    "schedulers": [scheduler.state_dict() for scheduler in schedulers],
                    "best_acc": group_best_acc,
                    'best_acc_hidx': group_best_acc_hidx,
                    "best_acc_sweep_lr_only": group_best_acc_sweep_lr_only,
                }
                torch.save(save_dict,
                           os.path.join(args.output_dir, "checkpoint_{}_linear.pth".format(args.checkpoint_key)))

            best_acc = max(best_acc, group_best_acc)
            best_acc_sweep_lr_only = max(best_acc_sweep_lr_only, group_best_acc_sweep_lr_only)
            print(f'Max accuracy so far: {best_acc:.2f}%')
            print(f'Max accuracy with sweeping lr only so far: {best_acc_sweep_lr_only:.2f}%')

    lr, wd, optim = args.permutes[group_best_acc_hidx]
    print("Training of the supervised linear classifier on frozen features completed.\n",
          "Top-1 test accuracy: {acc:.1f}\n".format(acc=best_acc),
          "Top-1 test accuracy with sweeping lr only: {acc:.1f}\n".format(acc=best_acc_sweep_lr_only),
          "Optim configs with top-1 test accuracy: lr {lr:.5f}, wd {wd:.0e}, optim {optim:4}\n".format(lr=lr, wd=wd, optim=optim))


def train(model, linear_classifiers, optimizers, loader, epoch, n, avgpool, permutes):
    metric_logger = utils.MetricLogger(delimiter="  ")
    for group, _ in enumerate(permutes):
        metric_logger.add_meter('lr{}'.format(group), utils.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 100, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model.forward_return_n_last_blocks(inp, n, avgpool)
            # torch.Size([256, 1536])

        losses = []
        for linear_classifier, optimizer in zip(linear_classifiers, optimizers):
            pred = linear_classifier(output)

            # compute cross entropy loss
            loss = nn.CrossEntropyLoss()(pred, target)

            # compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # step
            optimizer.step()

            losses.append(loss)

        # log 
        torch.cuda.synchronize()
        for group, (loss, optimizer) in enumerate(zip(losses, optimizers)):
            metric_logger.update(**{'loss{}'.format(group): loss.item()})
            metric_logger.update(**{'lr{}'.format(group): optimizer.param_groups[0]["lr"]})
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifiers, n, avgpool, permutes):
    linear_classifiers.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model.forward_return_n_last_blocks(inp, n, avgpool)

        losses = []
        acc1s = []
        acc5s = []
        for group, linear_classifier in enumerate(linear_classifiers):

            pred = linear_classifier(output)
            loss = nn.CrossEntropyLoss()(pred, target)
            losses.append(loss)

            if linear_classifier.module.num_labels >= 5:
                acc1, acc5 = utils.accuracy(pred, target, topk=(1, 5))
                acc1s.append(acc1)
                acc5s.append(acc5)
            else:
                acc1, = utils.accuracy(pred, target, topk=(1,))
                acc1s.append(acc1)

            batch_size = inp.shape[0]
            metric_logger.update(**{'loss{}'.format(group): loss.item()})
            metric_logger.meters['acc{}@1'.format(group)].update(acc1.item(), n=batch_size)
            if linear_classifier.module.num_labels >= 5:
                metric_logger.meters['acc{}@5'.format(group)].update(acc5.item(), n=batch_size)

    for group, (pm, linear_classifier) in enumerate(zip(permutes, linear_classifiers)):
        lr, wd, optim = pm
        if linear_classifier.module.num_labels >= 5:
            print(
                '* [Lr {lr:.5f} Wd {wd:.0e} Optim {optim:4}] Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(lr=lr, wd=wd, optim=optim,
                        top1=metric_logger.meters['acc{}@1'.format(group)],
                        top5=metric_logger.meters['acc{}@5'.format(group)],
                        losses=metric_logger.meters['loss{}'.format(group)]))
        else:
            print('* [Lr {lr:.5f} Wd {wd:.0e} Optim {optim:4}] Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
                  .format(lr=lr, wd=wd, optim=optim,
                          top1=metric_logger.meters['acc{}@1'.format(group)],
                          losses=metric_logger.meters['loss{}'.format(group)]))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating DeiT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for DeiT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='deit_small', type=str,
        choices=['deit_tiny', 'deit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--load_from', default=None, help='Path to load checkpoints to resume training')
    parser.add_argument('--sweep_lr_only', default=True, type=bool,
                        help='Wether or not to only sweep over learning rate')

    args = parser.parse_args()
    eval_linear(args)
