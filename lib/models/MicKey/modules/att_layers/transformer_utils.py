
import torch
import torch.nn as nn
from lib.models.MicKey.modules.att_layers.attention import Attention

class EncoderLayer(nn.Module):
    """
        Transformer encoder layer containing the linear self and cross-attention, and the epipolar attention.
        Arguments:
            d_model: Feature dimension of the input feature maps (default: 128d).
            nhead: Number of heads in the multi-head attention.
            attention: Type of attention for the common transformer block. Options: linear, full.
    """
    def __init__(self, d_model, nhead, attention='linear'):
        super(EncoderLayer, self).__init__()

        # Transformer encoder layer parameters
        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention definition
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # full_att = False if attention == 'linear' else True
        self.attention = Attention(attention=attention)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source):
        """
        Args:
            x (torch.Tensor): [N, L, C] (L = im_size/down_factor ** 2)
            source (torch.Tensor): [N, S, C]
            if is_epi_att:
                S = (im_size/down_factor/step_grid) ** 2 * sampling_dim
            else:
                S = im_size/down_factor ** 2
            is_epi_att (bool): Indicates whether it applies epipolar cross-attention
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message
