import torch
import torch.nn as nn
import torch.nn.functional as F


def desc_l2norm(desc: torch.Tensor):
    '''descriptors with shape NxC or NxCxHxW'''
    eps_l2_norm = 1e-10
    desc = desc / desc.pow(2).sum(dim=1, keepdim=True).add(eps_l2_norm).pow(0.5)
    return desc

class BasicBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, padding_mode='zeros'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(planes) if bn else nn.Identity()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x, relu=True):
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += shortcut
        if relu:
            out = F.relu(out)
        return out
