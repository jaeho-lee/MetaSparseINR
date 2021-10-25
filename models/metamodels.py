from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.metasiren import MetaSirenNet


def exists(val):
    return val is not None


class MetaWrapper(nn.Module):
    """
    This is a wrapper, that is used to train the meta-model given an image, e.g.,
    net = MetaSirenNet(...)
    wrapper = MetaWrapper(net)
    img = wrapper() <- makes an inference of the image.
    loss = wrapper(img) <- calculates the MSE loss between the image and model output.
    loss = wrapper(img,mask) <- calculates the MSE loss between the image and a part of the model.
    """
    def __init__(self, net, image_width, image_height):
        super().__init__()
        assert isinstance(net, MetaSirenNet)
        self.net = net

        self.image_width = image_width
        self.image_height = image_height

        tensors = [torch.linspace(-1, 1, steps=image_width), torch.linspace(-1, 1, steps=image_height)]
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')
        self.register_buffer('grid', mgrid)

    def forward(self, img=None, params=None):
        coords = self.grid.clone().detach()
        out = self.net(coords, params=params)
        out = rearrange(out, '(h w) c -> () c h w', h=self.image_height, w=self.image_width)
        if exists(img):
            return F.mse_loss(img, out)
        return out


def get_metamodel(netstr, dim_in, dim_hidden, dim_out, num_layers=4, w0=30.):
    if netstr == 'siren':
        return MetaSirenNet(dim_in, dim_hidden, dim_out, num_layers, w0=w0, w0_initial=w0)
    else:
        raise ValueError("no such model exists, mate.")
