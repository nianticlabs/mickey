
import copy
import math
import torch
import torch.nn as nn
from einops.einops import rearrange
from lib.models.MicKey.modules.att_layers.transformer_utils import EncoderLayer
import torch.nn.functional as F

class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

class Transformer_self_att(nn.Module):
    """This class implement self attention transformer module.
        Arguments:
            d_model: Feature dimension after feature extractor (default: 1024d).
            aggregator_conf: Configuration dictionary containing the parameters for the transformer module.
    """

    def __init__(self, d_model, num_layers, add_posEnc=False):
        super(Transformer_self_att, self).__init__()

        # Define the transformer parameters
        self.d_model = d_model

        # TODO: Expose parameters to config file
        layer_names = ['self'] * num_layers
        attention = 'linear'
        self.nheads = 8
        self.layer_names = layer_names
        encoder_layer = EncoderLayer(d_model, self.nheads, attention)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()
        self.add_posEnc = add_posEnc
        self.posEnc = PositionEncodingSine(d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, feats):
        """
            Runs the common self and cross-attention module.
            Args:
                feats_a: Features from image A (source) ([N, d_model, im_size/down_factor, im_size/down_factor]).
                feats_b: Features from image B (destination) ([N, d_model, im_size/down_factor, im_size/down_factor]).
            Output:
                feats_a: Self and cross-attended features corresponding to image A (source)
                ([N, d_model, im_size/down_factor, im_size/down_factor])
                feats_b: Self and cross-attended features corresponding to image B (destination)
                ([N, d_model, im_size/down_factor, im_size/down_factor]).
        """

        assert self.d_model == feats.size(1), "The feature size and transformer must be equal"

        b, c, h, w = feats.size()

        if self.add_posEnc:
            feats = self.posEnc(feats)

        feats = rearrange(feats, 'n c h w -> n (h w) c')

        # Apply linear self attention to feats
        for layer, name in zip(self.layers, self.layer_names):
            feats = layer(feats, feats)

        feats = feats.transpose(2, 1).reshape((b, c, h, w))

        return feats

class Transformer_att(nn.Module):
    """This class implement self attention transformer module.
        Arguments:
            d_model: Feature dimension after feature extractor (default: 1024d).
            aggregator_conf: Configuration dictionary containing the parameters for the transformer module.
    """

    def __init__(self, d_model, num_layers, add_posEnc=False):
        super(Transformer_att, self).__init__()

        # Define the transformer parameters
        self.d_model = d_model

        # TODO: Expose parameters to config file
        layer_names = ['self', 'cross'] * num_layers
        attention = 'linear'
        self.nheads = 8
        self.layer_names = layer_names
        encoder_layer = EncoderLayer(d_model, self.nheads, attention)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()
        self.add_posEnc = add_posEnc
        self.posEnc = PositionEncodingSine(d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, feats0, feats1):
        """
            Runs the common self and cross-attention module.
            Args:
                feats_a: Features from image A (source) ([N, d_model, im_size/down_factor, im_size/down_factor]).
                feats_b: Features from image B (destination) ([N, d_model, im_size/down_factor, im_size/down_factor]).
            Output:
                feats_a: Self and cross-attended features corresponding to image A (source)
                ([N, d_model, im_size/down_factor, im_size/down_factor])
                feats_b: Self and cross-attended features corresponding to image B (destination)
                ([N, d_model, im_size/down_factor, im_size/down_factor]).
        """

        assert self.d_model == feats0.size(1), "The feature size and transformer must be equal"

        b, c, h, w = feats0.size()

        if self.add_posEnc:
            feats0 = self.posEnc(feats0)
            feats1 = self.posEnc(feats1)

        feats0 = rearrange(feats0, 'n c h w -> n (h w) c')
        feats1 = rearrange(feats1, 'n c h w -> n (h w) c')

        # Apply linear self attention to feats
        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feats0 = layer(feats0, feats0)
                feats1 = layer(feats1, feats1)
            elif name == 'cross':
                feats0, feats1 = layer(feats0, feats1), layer(feats1, feats0)
            else:
                raise KeyError

        feats0 = feats0.transpose(2, 1).reshape((b, c, h, w))
        feats1 = feats1.transpose(2, 1).reshape((b, c, h, w))

        return feats0, feats1
