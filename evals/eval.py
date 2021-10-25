import time

import numpy as np
import torch

from tools.dataloader import get_loaders
from tools.metapruners import get_pruner
from models.metamodels import MetaWrapper, get_metamodel
from utils import AverageMeter, psnr


def test(args, logger=None, prune=False):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    """ Get Image """
    log_(f"Loading dataset {args.data} with resolution {args.res}")
    train_loader, test_loader = get_loaders(args.data, args.res, args.batch_size, args.use_train_set)

    if args.use_train_set:
        test_loader = train_loader
        log_(f"Using training set for evaluation...")

    """ Get Model """
    log_(f"Generating {args.model} with width {args.width} and {args.depth} hidden layers")
    net = get_metamodel(args.model, dim_in=args.dim_in,
                        dim_hidden=args.width, dim_out=args.dim_out,
                        num_layers=args.depth, w0=args.w0).cuda()

    if prune:
        from tools.metapruners import prune_reparam
        prune_reparam(net)

    return eval(args, net, test_loader, args.optim, args.num_steps_inner, args.alpha, logger)


def eval(args, net, loader, optim='sgd', num_steps_inner=2, alpha=0.01, logger=None):

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    losses = dict()
    losses['mse'] = AverageMeter()
    losses['psnr'] = AverageMeter()
    check = time.time()

    if args.load_path is not None and not args.no_init:
        log_(f"Loading model from {args.load_path}")
        checkpoint = torch.load(args.load_path)

    for n, (data, _) in enumerate(loader):
        if args.num_test is not None:
            if n > args.num_test:
                break

        if args.no_init:
            net = get_metamodel(args.model, dim_in=args.dim_in,
                                dim_hidden=args.width, dim_out=args.dim_out,
                                num_layers=args.depth, w0=args.w0).cuda()
        else:
            net.load_state_dict(checkpoint)

        if optim == 'sgd':
            opt = torch.optim.SGD(net.parameters(), lr=alpha)
        elif optim == 'adam':
            opt = torch.optim.Adam(net.parameters(), lr=alpha)
        else:
            raise NotImplementedError()
        wrapper = MetaWrapper(net, image_width=args.res, image_height=args.res).cuda()

        batch_size = data.size(0)
        assert batch_size == 1

        data = data.cuda()

        for step_inner in range(num_steps_inner):
            loss = wrapper(data)

            opt.zero_grad()
            loss.backward()
            opt.step()

        ### Log losses ###
        mse = wrapper(data)
        losses['mse'].update(mse.item(), batch_size)
        losses['psnr'].update(psnr(mse).item(), batch_size)

    log_('[Time %3d] [MSE %f] [PSNR %f]' %
         (time.time()-check, losses['mse'].average, losses['psnr'].average))
    return losses['mse'].average, losses['psnr'].average


def eval_imp(args, loader, iter=1., logger=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    log_(f"Loading model from {args.load_path}")
    checkpoint = torch.load(args.load_path)

    pruner = get_pruner('MP')

    losses = dict()
    losses['mse'] = AverageMeter()
    losses['psnr'] = AverageMeter()
    check = time.time()

    for n, (data, _) in enumerate(loader):
        if args.num_test is not None:
            if n > args.num_test:
                break

        net = get_metamodel(args.model, dim_in=args.dim_in,
                            dim_hidden=args.width, dim_out=args.dim_out,
                            num_layers=args.depth, w0=args.w0).cuda()
        net.load_state_dict(checkpoint)

        wrapper = MetaWrapper(net, image_width=args.res, image_height=args.res).cuda()

        if args.optim == 'adam':
            opt = torch.optim.Adam(net.parameters(), lr=args.alpha)
        else:
            raise NotImplementedError()

        batch_size = data.size(0)
        assert batch_size == 1

        data = data.cuda()

        prune_step = np.arange(
            0, args.num_steps_inner, float(args.num_steps_inner / (iter + 1))
        )[1:1 + iter].astype(np.int).tolist()

        for step_inner in range(1, args.num_steps_inner+1):
            loss = wrapper(data)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step_inner in prune_step:
                pruner(net, args.amount)

        ### Log losses ###
        mse = wrapper(data)

        losses['mse'].update(mse.item(), batch_size)
        losses['psnr'].update(psnr(mse).item(), batch_size)

    log_('[Time %3d] [MSE %f] [PSNR %f]' %
         (time.time() - check, losses['mse'].average, losses['psnr'].average))
    return losses['mse'].average, losses['psnr'].average


def eval_oneshot(args, loader, iter=1., logger=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    log_(f"Loading model from {args.load_path}")
    checkpoint = torch.load(args.load_path)

    pruner = get_pruner('MP')

    losses = dict()
    losses['mse'] = AverageMeter()
    losses['psnr'] = AverageMeter()
    check = time.time()

    amount_list = [(1 - args.amount)**i for i in range(args.prune_num)]

    for n, (data, _) in enumerate(loader):
        if args.num_test is not None:
            if n > args.num_test:
                break

        net = get_metamodel(args.model, dim_in=args.dim_in,
                            dim_hidden=args.width, dim_out=args.dim_out,
                            num_layers=args.depth, w0=args.w0).cuda()
        net.load_state_dict(checkpoint)

        wrapper = MetaWrapper(net, image_width=args.res, image_height=args.res).cuda()

        if args.optim == 'adam':
            opt = torch.optim.Adam(net.parameters(), lr=args.alpha)
        else:
            raise NotImplementedError()

        batch_size = data.size(0)
        assert batch_size == 1

        data = data.cuda()

        amount = 1.0 - amount_list[int(iter)]

        for step_inner in range(1, args.num_steps_inner+1):
            loss = wrapper(data)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step_inner == int(args.num_steps_inner/2):
                pruner(net, amount)

        ### Log losses ###
        mse = wrapper(data)

        losses['mse'].update(mse.item(), batch_size)
        losses['psnr'].update(psnr(mse).item(), batch_size)

    log_('[Time %3d] [MSE %f] [PSNR %f]' %
         (time.time() - check, losses['mse'].average, losses['psnr'].average))
    return losses['mse'].average, losses['psnr'].average
