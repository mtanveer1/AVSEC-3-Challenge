import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.dnn import BasicBlock, ResNet, Swish, cal_si_snr


class AudioEncoder(nn.Module):
    def __init__(self, kernel_size=2, out_channels=64):
        super(AudioEncoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=out_channels,
                                kernel_size=kernel_size, stride=kernel_size // 2, groups=1, bias=False)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.conv1d(x)
        x = F.relu(x)
        return x


class AudioDecoder(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super(AudioDecoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class VisualFeatNet(nn.Module):
    def __init__(self, relu_type='relu'):
        super(VisualFeatNet, self).__init__()
        self.frontend_nout = 64
        self.trunk = ResNet(BasicBlock, [1, 1, 1, 1, 1], relu_type=relu_type)
        if relu_type == 'relu':
            frontend_relu = nn.ReLU(True)
        elif relu_type == 'prelu':
            frontend_relu = nn.PReLU(self.frontend_nout)
        elif relu_type == 'swish':
            frontend_relu = Swish()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))

        self.nn_out = nn.Linear(512, 256, bias=False)
        torch.nn.init.xavier_uniform_(self.nn_out.weight)
        self._initialize_weights_randomly()

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]
        n_batch, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2).reshape(n_batch * s_time, n_channels, sx, sy)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        return torch.relu(self.nn_out(x))

    def _initialize_weights_randomly(self):
        f = lambda n: math.sqrt(2.0 / float(n))
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod(m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))


class SeparatorBlock(nn.Module):
    def __init__(self, out_channels, hidden_channels, dropout=0, bidirectional=False):
        super(SeparatorBlock, self).__init__()
        self.intra_rnn = nn.LSTM(out_channels, hidden_channels, 1, batch_first=True, dropout=dropout,
                                 bidirectional=bidirectional)
        self.inter_rnn = nn.LSTM(out_channels, hidden_channels, 1, batch_first=True, dropout=dropout,
                                 bidirectional=bidirectional)
        self.intra_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.inter_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.intra_linear = nn.Linear(hidden_channels * 2 if bidirectional else hidden_channels, out_channels)
        self.inter_linear = nn.Linear(hidden_channels * 2 if bidirectional else hidden_channels, out_channels)

    def forward(self, x):
        B, N, K, S = x.shape
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B * S * K, -1)).view(B * S, K, -1)
        intra_rnn = intra_rnn.view(B, S, K, N)
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)
        intra_rnn = intra_rnn + x
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B * S * K, -1)).view(B * K, S, -1)
        inter_rnn = inter_rnn.view(B, K, S, N)
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        out = inter_rnn + intra_rnn
        return out


class Separator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout=0,
                 bidirectional=False, num_layers=4, K=200):
        super(Separator, self).__init__()
        self.K = K
        self.num_layers = num_layers
        self.input_conv = nn.Sequential(nn.GroupNorm(1, in_channels, eps=1e-8),
                                        nn.Conv1d(in_channels, out_channels, 1, bias=False))
        self.separator_blocks = nn.Sequential(*[SeparatorBlock(out_channels, hidden_channels, dropout=dropout,
                                                               bidirectional=bidirectional) for _ in range(num_layers)])
        self.conv2d = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.end_conv1x1 = nn.Conv1d(out_channels, 256, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Tanh())  
        self.output_gate = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.input_conv(x)
        x, gap = self._segment(x, self.K)
        x = self.separator_blocks(x)
        x = self.prelu(x)
        x = self.conv2d(x)
        B, _, K, S = x.shape
        x = x.view(B, -1, K, S)
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)
        x = self.end_conv1x1(x)
        _, N, L = x.shape
        x = x.view(B, -1, N, L)
        x = self.activation(x)
        return x.transpose(0, 1)[0]

    def _padding(self, input, K):
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)
        return input, gap

    def _segment(self, input, K):
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)
        return input.contiguous(), gap

    def _over_add(self, input, gap):
        B, N, K, S = input.shape
        P = K // 2
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)
        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        if gap > 0:
            input = input[:, :, :-gap]
        return input


class AVSE(nn.Module):
    def __init__(self):
        super(AVSE, self).__init__()
        self.audio_encoder = AudioEncoder(kernel_size=16, out_channels=256)
        self.audio_decoder = AudioDecoder(in_channels=256, out_channels=1, kernel_size=16, stride=8, bias=False)
        self.visual_encoder = VisualFeatNet()
        self.separator = Separator(512, 32, 32, num_layers=1, bidirectional=True, dropout=0.3)
        #self.separator = Separator(512, 256, 256, num_layers=1, bidirectional=True)
        #self.separator = Separator(512, 32, 32, num_layers=1, bidirectional=True, dropout=0.3) # Best
        #self.separator = Separator(512, 64, 128, num_layers=1, bidirectional=True, dropout=0.2)

    def forward(self, input):
        noisy = input["noisy_audio"]
        encoded_audio = self.audio_encoder(noisy)
        video_frames = input["video_frames"]
        encoded_visual = self.visual_encoder(video_frames)
        _, _, time_steps = encoded_audio.shape
        _, _, vis_feat_size = encoded_visual.shape
        upsampled_visual_feat = F.interpolate(encoded_visual.unsqueeze(1), size=(time_steps, vis_feat_size),
                                              mode="bilinear").reshape(-1, time_steps, vis_feat_size).moveaxis(1, 2)
        encoded_av = torch.cat((upsampled_visual_feat, encoded_audio), dim=-2)
        mask = self.separator(encoded_av)
        out = mask * encoded_audio
        audio = self.audio_decoder(out)
        return audio


class AVSEModule(LightningModule):
    def __init__(self, lr=0.001, val_dataset=None):
        super(AVSEModule, self).__init__()
        self.lr = lr
        self.val_dataset = val_dataset
        self.loss = cal_si_snr
        self.model = AVSE()

    def forward(self, data):
        """ Processes the input tensor x and returns an output tensor."""
        est_source = self.model(data)
        return est_source

    def training_step(self, batch_inp, batch_idx):
        loss = self.cal_loss(batch_inp)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch_inp, batch_idx):
        loss = self.cal_loss(batch_inp)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def enhance(self, data):
        inputs = dict(noisy_audio=torch.from_numpy(data["noisy_audio"][np.newaxis, ...]).to(self.device),
                      video_frames=torch.from_numpy(data["video_frames"][np.newaxis, ...]).to(self.device))
        estimated_audio = self(inputs).cpu().numpy()
        estimated_audio /= np.max(np.abs(estimated_audio))
        return estimated_audio

    def training_epoch_end(self, outputs):
        if self.val_dataset is not None:
            with torch.no_grad():
                tensorboard = self.logger.experiment
                for index in range(5):
                    rand_int = random.randint(0, len(self.val_dataset))
                    data = self.val_dataset[rand_int]
                    estimated_audio = self.enhance(data)
                    tensorboard.add_audio("{}/{}_clean".format(self.current_epoch, index),
                                          data["clean"][np.newaxis, ...],
                                          sample_rate=16000)
                    tensorboard.add_audio("{}/{}_noisy".format(self.current_epoch, index),
                                          data["noisy_audio"][np.newaxis, ...],
                                          sample_rate=16000)
                    tensorboard.add_audio("{}/{}_enhanced".format(self.current_epoch, index),
                                          estimated_audio.reshape(-1)[np.newaxis, ...],
                                          sample_rate=16000)

    def cal_loss(self, batch_inp):
        mask = batch_inp["clean"].T
        pred_mask = self(batch_inp).T.reshape(mask.shape)
        loss = self.loss(pred_mask.unsqueeze(2), mask.unsqueeze(2))
        loss[loss < -30] = -30
        return torch.mean(loss)

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.8, patience=5),
                "monitor": "val_loss_epoch",
            },
        }
