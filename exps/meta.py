import json

import torch

from tools.trainer import maml
from tools.dataloader import get_loaders
from models.metamodels import MetaWrapper, get_metamodel


def exp(rootdir, args, logger=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    """ Saving setups """
    argdict = vars(args)
    with open(rootdir + 'args.json', "w") as fp:
        json.dump(argdict, fp)

    """ Get Image """
    log_(f"Loading dataset {args.data} with resolution {args.res}")
    train_loader, test_loader = get_loaders(args.data, args.res, args.batch_size)

    """ Get Model """
    log_(f"Generating {args.model} with width {args.width} and {args.depth} hidden layers")
    log_(f"Model input dim shape with {args.dim_in} and output dim shape with {args.dim_out}")
    net = get_metamodel(args.model, dim_in=args.dim_in,
                        dim_hidden=args.width, dim_out=args.dim_out,
                        num_layers=args.depth, w0=args.w0).cuda()

    torch.save(net.state_dict(), rootdir + f'net_init.pth')

    """ Train """
    wrapper = MetaWrapper(net, image_width=args.res, image_height=args.res).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=args.beta)
    log_(f"Training for {args.epochs} epochs")
    maml(0, wrapper, train_loader, opt, args.epochs, args.num_steps_inner, args.alpha, logger)
    torch.save(net.state_dict(), rootdir + f'net_meta.pth')
