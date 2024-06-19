import math

import torch
import torch.nn as nn


def get_mask(source, source_lengths):
    mask = source.new_ones(source.size()[:-1]).unsqueeze(-1).transpose(1, -2)
    B = source.size(-2)
    for i in range(B):
        mask[source_lengths[i]:, i] = 0
    return mask.transpose(-2, 1)


def cal_si_snr(source, estimate_source):
    EPS = 1e-8
    assert source.size() == estimate_source.size()
    device = estimate_source.device.type

    source_lengths = torch.tensor(
        [estimate_source.shape[0]] * estimate_source.shape[-2], device=device
    )
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    num_samples = (
        source_lengths.contiguous().reshape(1, -1, 1).float()
    )  # [1, B, 1]
    mean_target = torch.sum(source, dim=0, keepdim=True) / num_samples
    mean_estimate = (
            torch.sum(estimate_source, dim=0, keepdim=True) / num_samples
    )
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = zero_mean_target  # [T, B, C]
    s_estimate = zero_mean_estimate  # [T, B, C]
    # s_target = <s', s>s / ||s||^2
    dot = torch.sum(s_estimate * s_target, dim=0, keepdim=True)  # [1, B, C]
    s_target_energy = (
            torch.sum(s_target ** 2, dim=0, keepdim=True) + EPS
    )  # [1, B, C]
    proj = dot * s_target / s_target_energy  # [T, B, C]
    # e_noise = s' - s_target
    e_noise = s_estimate - proj  # [T, B, C]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    si_snr_beforelog = torch.sum(proj ** 2, dim=0) / (
            torch.sum(e_noise ** 2, dim=0) + EPS
    )
    si_snr = 10 * torch.log10(si_snr_beforelog + EPS)  # [B, C]

    return -si_snr.unsqueeze(0)



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def downsample_basic_block(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(outplanes),
        nn.MaxPool2d(kernel_size=(1, 1)),
        nn.Dropout(p=0.30, inplace=True) 
    )


def downsample_basic_block_v2(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(outplanes),
        nn.Dropout(p=0.30, inplace=True) 
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type='prelu'):
        super(BasicBlock, self).__init__()

        assert relu_type in ['relu', 'prelu', 'swish']

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(planes)

        # type of ReLU is an input option
        #f relu_type == 'relu':
        #    self.relu1 = nn.ReLU(inplace=True)
        #    self.relu2 = nn.ReLU(inplace=True)
        #elif relu_type == 'prelu':
        #    self.relu1 = nn.PReLU(num_parameters=planes)
        #    self.relu2 = nn.PReLU(num_parameters=planes)
        #elif relu_type == 'swish':
        #    self.relu1 = Swish()
        #    self.relu2 = Swish()
        #else:
        #    raise Exception('relu type not implemented')
        # --------

        self.conv2 = conv3x3(planes, planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride
        self.planes_size = planes

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)
            
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, relu_type='relu', gamma_zero=False, avg_pool_downsample=False):
        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block

        super(ResNet, self).__init__()
        #self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        #self.layer1 = self._make_layer(block, 32,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64,  layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=2)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride >= 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(inplanes=self.inplanes,
                                               outplanes=planes * block.expansion,
                                               stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        #x = self.avgpool1(x)
        #x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool2(x)
        x = x.view(x.size(0), -1)
        return x
