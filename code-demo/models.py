import copy
import math

import numpy
import torch
from torch import nn
from torch.nn import functional as F

import attentions_2
import commons
from glow_tts import ActNorm, InvConvNear, CouplingBlock, StochasticPitchPredictor
import modules
import attentions

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
from pqmf import PQMF
from stft import TorchSTFT
import math
class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class encoder_2(nn.Module):
    """Rhythm Encoder
    """

    def __init__(self, ):
        super().__init__()

        self.dim_neck_2 = 1
        self.freq_2 = 8
        self.dim_freq = 80
        self.dim_enc_2 = 128
        self.dim_emb = 82
        self.chs_grp = 16

        convolutions = []
        for i in range(1):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_freq if i == 0 else self.dim_enc_2,
                         self.dim_enc_2,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc_2 // self.chs_grp, self.dim_enc_2))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(self.dim_enc_2, self.dim_neck_2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, mask=None):

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        if mask is not None:
            outputs = outputs * mask
        out_forward = outputs[:, :, :self.dim_neck_2].squeeze(2)
        out_backward = outputs[:, :, self.dim_neck_2:].squeeze(2)

        codes = torch.cat((out_forward[:, self.freq_2 - 1::self.freq_2], out_backward[:, ::self.freq_2]), -1)

        return codes
class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                                              gin_channels=gin_channels, mean_only=True))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, f0,g=None, emotion=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask,f0=f0, g=g, emotion=emotion, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask,f0=f0, g=g, emotion=emotion, reverse=reverse)
        return x


# class ResidualCouplingBlock(nn.Module):
#   def __init__(self,
#       channels,
#       hidden_channels,
#       kernel_size,
#       dilation_rate,
#       n_layers,
#       n_flows=4,
#       gin_channels=0):
#     super().__init__()
#     self.channels = channels
#     self.hidden_channels = hidden_channels
#     self.kernel_size = kernel_size
#     self.dilation_rate = dilation_rate
#     self.n_layers = n_layers
#     self.n_flows = n_flows
#     self.gin_channels = gin_channels
#     self.n_sqz = 2
#     self.n_blocks=12
#     self.sigmoid_scale=True
#     self.p_dropout=0.05
#     self.n_split=4
#     self.in_channels=192
#
#     self.flows = nn.ModuleList()
#     for _ in range(self.n_blocks):
#         self.flows.append(ActNorm(channels=self.in_channels * self.n_sqz))
#         self.flows.append(
#             InvConvNear(channels=self.in_channels * self.n_sqz, n_split=self.n_split)
#         )
#         self.flows.append(
#             CouplingBlock(
#                 self.in_channels * self.n_sqz,
#                 hidden_channels,
#                 kernel_size=kernel_size,
#                 dilation_rate=dilation_rate,
#                 n_layers=n_layers,
#                 p_dropout=self.p_dropout,
#                 sigmoid_scale=self.sigmoid_scale,
#                 gin_channels=gin_channels,
#             )
#         )
#   def squeeze(self, x, x_mask=None, n_sqz=1):
#       b, c, t = x.size()
#       #print("111")
#       #print(x.shape)
#       t = (t // n_sqz) * n_sqz
#       x = x[:, :, :t]
#       #print(x.shape)
#       x_sqz = x.view(b, c, t // n_sqz, n_sqz)
#       #print(x_sqz.shape)
#       x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)
#       #print(x_sqz.shape)
#       if x_mask is not None:
#           x_mask = x_mask[:, :, n_sqz - 1:: n_sqz]
#           #x_mask = x_mask.view(b, c, t // n_sqz, n_sqz)
#           #x_mask=x_mask.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)
#       else:
#           x_mask = torch.ones(b, 1, t // n_sqz).to(device=x.device, dtype=x.dtype)
#
#       #print(x_mask.shape)
#       return x_sqz * x_mask, x_mask
#
#
#   def unsqueeze(self, x, x_mask=None, n_sqz=1):
#       b, c, t = x.size()
#
#       x_unsqz = x.view(b, n_sqz, c // n_sqz, t)
#       x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // n_sqz, t * n_sqz)
#
#       if x_mask is not None:
#           x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, n_sqz).view(b, 1, t * n_sqz)
#       else:
#           x_mask = torch.ones(b, 1, t * n_sqz).to(device=x.device, dtype=x.dtype)
#       return x_unsqz * x_mask, x_mask
#
#   def forward(self,x,x_mask,g=None,pitch=None,reverse=False):
#       logdet=None
#       # print(x.shape)
#       # print(x_mask.shape)
#       if not reverse:
#           flows = self.flows
#           logdet_tot = 0
#       else:
#           flows = reversed(self.flows)
#           logdet_tot = None
#
#       if self.n_sqz > 1:
#           x, x_mask = self.squeeze(x, x_mask, self.n_sqz)
#
#       for f in flows:
#           if not reverse:
#               #print("999")
#               #print(x.shape)
#               #print(x_mask.shape)
#               x, logdet = f(x, x_mask, g=g,
#                              pitch=pitch,reverse=reverse)
#               logdet_tot += logdet
#           else:
#               x, logdet = f(x, x_mask, g=g,
#                              pitch=pitch,reverse=reverse)
#       if self.n_sqz > 1:
#           x, x_mask = self.unsqueeze(x, x_mask, self.n_sqz)
#       # print("fin")
#       # print(x.shape)
#       return x

def padDiff(x):
    return F.pad(F.pad(x, (0,0,-1,1), 'constant', 0) - x, (0,0,0,-1), 'constant', 0)

class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.onnx = False

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], \
                              device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for normal case

            # To prevent torch.cumsum numerical overflow,
            # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
            # Buffer tmp_over_one_idx indicates the time step to add -1.
            # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1)
                              * 2 * numpy.pi)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * numpy.pi)
        return sines

    def forward(self, f0, upp=None):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        if self.onnx:
            with torch.no_grad():
                f0 = f0[:, None].transpose(1, 2)
                f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
                # fundamental component
                f0_buf[:, :, 0] = f0[:, :, 0]
                for idx in numpy.arange(self.harmonic_num):
                    f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (
                        idx + 2
                    )  # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
                rad_values = (f0_buf / self.sampling_rate) % 1  ###%1意味着n_har的乘积无法后处理优化
                rand_ini = torch.rand(
                    f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device
                )
                rand_ini[:, 0] = 0
                rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
                tmp_over_one = torch.cumsum(rad_values, 1)  # % 1  #####%1意味着后面的cumsum无法再优化
                tmp_over_one *= upp
                tmp_over_one = F.interpolate(
                    tmp_over_one.transpose(2, 1),
                    scale_factor=upp,
                    mode="linear",
                    align_corners=True,
                ).transpose(2, 1)
                rad_values = F.interpolate(
                    rad_values.transpose(2, 1), scale_factor=upp, mode="nearest"
                ).transpose(
                    2, 1
                )  #######
                tmp_over_one %= 1
                tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
                cumsum_shift = torch.zeros_like(rad_values)
                cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
                sine_waves = torch.sin(
                    torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * numpy.pi
                )
                sine_waves = sine_waves * self.sine_amp
                uv = self._f02uv(f0)
                uv = F.interpolate(
                    uv.transpose(2, 1), scale_factor=upp, mode="nearest"
                ).transpose(2, 1)
                noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
                noise = noise_amp * torch.randn_like(sine_waves)
                sine_waves = sine_waves * uv + noise
            return sine_waves, uv, noise
        else:
            with torch.no_grad():
                # fundamental component
                fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

                # generate sine waveforms
                sine_waves = self._f02sine(fn) * self.sine_amp

                # generate uv signal
                # uv = torch.ones(f0.shape)
                # uv = uv * (f0 > self.voiced_threshold)
                uv = self._f02uv(f0)

                # noise: for unvoiced should be similar to sine_amp
                #        std = self.sine_amp/3 -> max value ~ self.sine_amp
                # .       for voiced regions is self.noise_std
                noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
                noise = noise_amp * torch.randn_like(sine_waves)

                # first: set the unvoiced part to 0 by uv
                # then: additive noise
                sine_waves = sine_waves * uv + noise
            return sine_waves, uv, noise

class PosteriorEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        # print("888888")
        # print(x.shape)
        x = self.enc(x, x_mask, g=g)
        #print(x.shape)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp=None):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs.to(self.l_linear.weight.dtype)))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv

class iSTFT_Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, gin_channels=0):
        super(iSTFT_Generator, self).__init__()
        # self.h = h
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.post_n_fft = self.gen_istft_n_fft
        self.conv_post = weight_norm(Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.cond = nn.Conv1d(256, 512, 1)
        self.stft = TorchSTFT(filter_length=self.gen_istft_n_fft, hop_length=self.gen_istft_hop_size,
                              win_length=self.gen_istft_n_fft)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=h["sampling_rate"],
            harmonic_num=8)

    def forward(self, x, f0,g=None):
        #print("88")
        if not self.onnx:
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        # print(2,f0.shape)
        har_source, noi_source, uv = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        x = self.conv_pre(x)
        x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            # print(4,x_source.shape,har_source.shape,x.shape)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = math.pi * torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        out = self.stft.inverse(spec, phase).to(x.device)
        return out, None

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class Multiband_iSTFT_Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, subbands,
                 gin_channels=0):
        super(Multiband_iSTFT_Generator, self).__init__()
        # self.h = h
        self.subbands = subbands
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u + 1 - i) // 2, output_padding=1 - i)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.post_n_fft = gen_istft_n_fft
        self.ups.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.reshape_pixelshuffle = []

        self.subband_conv_post = weight_norm(Conv1d(ch, self.subbands * (self.post_n_fft + 2), 7, 1, padding=3))

        self.subband_conv_post.apply(init_weights)
        self.cond = nn.Conv1d(256, 512, 1)
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size

    def forward(self, x, f0,g=None):
        #print("77777777777")
        stft = TorchSTFT(filter_length=self.gen_istft_n_fft, hop_length=self.gen_istft_hop_size,
                         win_length=self.gen_istft_n_fft).to(x.device)
        # print(x.device)
        pqmf = PQMF(x.device)

        x = self.conv_pre(x)  # [B, ch, length]
        x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.subband_conv_post(x)
        x = torch.reshape(x, (x.shape[0], self.subbands, x.shape[1] // self.subbands, x.shape[-1]))

        spec = torch.exp(x[:, :, :self.post_n_fft // 2 + 1, :])
        phase = math.pi * torch.sin(x[:, :, self.post_n_fft // 2 + 1:, :])

        y_mb_hat = stft.inverse(
            torch.reshape(spec, (spec.shape[0] * self.subbands, self.gen_istft_n_fft // 2 + 1, spec.shape[-1])),
            torch.reshape(phase, (phase.shape[0] * self.subbands, self.gen_istft_n_fft // 2 + 1, phase.shape[-1])))
        y_mb_hat = torch.reshape(y_mb_hat, (x.shape[0], self.subbands, 1, y_mb_hat.shape[-1]))
        y_mb_hat = y_mb_hat.squeeze(-2)

        y_g_hat = pqmf.synthesis(y_mb_hat)

        return y_g_hat, y_mb_hat

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class Multistream_iSTFT_Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, subbands,
                 gin_channels=0):
        super(Multistream_iSTFT_Generator, self).__init__()
        # self.h = h
        self.subbands = subbands
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u + 1 - i) // 2,
                                output_padding=1 - i)))  # 这里k和u不是成倍数的关系，对最终结果很有可能是有影响的，会有checkerboard artifacts的现象

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.post_n_fft = gen_istft_n_fft
        self.ups.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.reshape_pixelshuffle = []

        self.subband_conv_post = weight_norm(Conv1d(ch, self.subbands * (self.post_n_fft + 2), 7, 1, padding=3))

        self.subband_conv_post.apply(init_weights)

        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size

        updown_filter = torch.zeros((self.subbands, self.subbands, self.subbands)).float()
        for k in range(self.subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.multistream_conv_post = weight_norm(Conv1d(4, 1, kernel_size=63, bias=False, padding=get_padding(63, 1)))
        self.multistream_conv_post.apply(init_weights)
        self.cond = nn.Conv1d(256, 512, 1)
        self.fond=nn.Conv1d(40,512,1)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=16000,
            harmonic_num=8)
        self.upp = numpy.prod(24000)

    def forward(self, x, g=None):
        #print("666666666666")
        #har_source, noi_source, uv = self.m_source(f0, self.upp)
        #har_source = har_source.transpose(1, 2)
        stft = TorchSTFT(filter_length=self.gen_istft_n_fft, hop_length=self.gen_istft_hop_size,
                         win_length=self.gen_istft_n_fft).to(x.device)
        # pqmf = PQMF(x.device)

        x = self.conv_pre(x)  # [B, ch, length]
        # print(x.size(),g.size())
        x = x + self.cond(g)  # g [b, 256, 1] => cond(g) [b, 512, 1]
        #print(f0.shape)
        #print(x.shape)
        #x = x+ self.fond(f0.unsqueeze(2))

        for i in range(self.num_upsamples):

            # print(x.size(),g.size())
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            # print(x.size(),g.size())
            x = self.ups[i](x)
            #x_source = self.noise_convs[i](har_source)
            # print(4,x_source.shape,har_source.shape,x.shape)
            #x = x + x_source
            xs = None
            # print(x.size(),g.size())
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        # print(x.size(),g.size())
        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.subband_conv_post(x)
        x = torch.reshape(x, (x.shape[0], self.subbands, x.shape[1] // self.subbands, x.shape[-1]))
        # print(x.size(),g.size())
        spec = torch.exp(x[:, :, :self.post_n_fft // 2 + 1, :])
        phase = math.pi * torch.sin(x[:, :, self.post_n_fft // 2 + 1:, :])
        # print(spec.size(),phase.size())
        y_mb_hat = stft.inverse(
            torch.reshape(spec, (spec.shape[0] * self.subbands, self.gen_istft_n_fft // 2 + 1, spec.shape[-1])),
            torch.reshape(phase, (phase.shape[0] * self.subbands, self.gen_istft_n_fft // 2 + 1, phase.shape[-1])))
        # print(y_mb_hat.size())
        y_mb_hat = torch.reshape(y_mb_hat, (x.shape[0], self.subbands, 1, y_mb_hat.shape[-1]))
        # print(y_mb_hat.size())
        y_mb_hat = y_mb_hat.squeeze(-2)
        # print(y_mb_hat.size())
        y_mb_hat = F.conv_transpose1d(y_mb_hat, self.updown_filter * self.subbands,
                                      stride=self.subbands)  # .cuda(x.device) * self.subbands, stride=self.subbands)
        # print(y_mb_hat.size())
        y_g_hat = self.multistream_conv_post(y_mb_hat)
        # print(y_g_hat.size(),y_mb_hat.size())
        return y_g_hat, y_mb_hat

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=80, model_num_layers=3, model_hidden_size=256, model_embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames - partial_frames, partial_hop):
            mel_range = torch.arange(i, i + partial_frames)
            mel_slices.append(mel_range)

        return mel_slices

    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:, -partial_frames:]

        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(mel_len, partial_frames, partial_hop)
            mels = list(mel[:, s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)

            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            # embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)

        return embed


# class predict(nn.Module):
#     def __init__(self,
#                  **kwargs):
#         super().__init__()
#         self.pitch_predictor=StochasticPitchPredictor(in_channels=192 ,filter_channels=256,kernel_size=3,p_dropout=0.1,n_flows= 4,gin_channels=192)
#         self.stoch_pitch_noise_scale=1.0
#         self.pitch_scale=0.0
#     def forward(self,x,x_mask,speaker_embeddings,pitch=None,used=False):
#         if used:
#             pitch=None
#             if self.pitch_predictor is not None:
#                 # expand embeddings if required
#                 _,pitch = self.pitch_predictor(x, x_mask, w=None, g=speaker_embeddings,
#                                                         noise_scale=self.stoch_pitch_noise_scale,
#                                                         reverse=True, )
#                 pitch = pitch.squeeze(1)
#                 pitch = torch.clamp_min(pitch, 0)
#
#                 #if pitch.shape[-1] != z.shape[-1]:
#                     # need to expand predicted pitch to match no of tokens
#                     #durs_predicted = x_mask.squeeze()
#                     #pitch, _ = regulate_len(durs_predicted, pitch.unsqueeze(-1))
#                     #pitch = pitch.squeeze(-1)
#
#                 pitch = pitch + self.pitch_scale
#                 # pitch[pitch_mask] = 0.0
#                 pitch = pitch.squeeze(1)
#             return pitch,None,None
#         else:
#             x_pitch = x.detach()
#             pitch_norm = pitch
#             pitch_mask2 = (pitch_norm == 0.0)
#             pitch_norm = torch.log(torch.clamp(pitch_norm, min=torch.finfo(pitch_norm.dtype).tiny))
#             pitch_norm[pitch_mask2] = 0.0
#             pitch_norm = pitch_norm.squeeze(1)
#             print("888")
#             print(x_pitch.shape)
#             print(x_mask.shape)
#             print(speaker_embeddings.shape)
#             pitch_pred_loss, pitch_pred = self.pitch_predictor(x_pitch, x_mask, w=pitch_norm.unsqueeze(1),
#                                                           g=speaker_embeddings, )
#             if pitch_pred:
#                 pitch_pred = pitch_pred.squeeze(1)
#             return pitch_norm, pitch_pred, pitch_pred_loss

def normalize_f0(f0, x_mask, uv, random_scale=True):
    # calculate means based on x_mask
    uv_sum = torch.sum(uv, dim=1, keepdim=True)
    uv_sum[uv_sum == 0] = 9999
    means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum

    if random_scale:
        factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
    else:
        factor = torch.ones(f0.shape[0], 1).to(f0.device)
    # normalize f0 based on means and factor
    f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
    if torch.isnan(f0_norm).any():
        exit(0)
    return f0_norm * x_mask

class F0Decoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 spk_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.spk_channels = spk_channels

        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = attentions_2.FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(self, x, norm_f0, x_mask, spk_emb=None):
        x = torch.detach(x)
        if (spk_emb is not None):
            # print(x.shape)
            # print(spk_emb.shape)
            x = x + self.cond(spk_emb)
        x += self.f0_prenet(norm_f0)
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x
class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 gen_istft_n_fft,
                 gen_istft_hop_size,
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=False,
                 ms_istft_vits=False,
                 mb_istft_vits=False,
                 subbands=False,
                 istft_vits=False,
                 **kwargs):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.ms_istft_vits = ms_istft_vits
        self.mb_istft_vits = mb_istft_vits
        self.istft_vits = istft_vits
        self.gin_channels = gin_channels

        self.use_sdp = use_sdp

        self.enc_p = PosteriorEncoder(256, inter_channels, hidden_channels, 5, 1,
                                      16)  # 768, inter_channels, hidden_channels, 5, 1, 16)

        if mb_istft_vits == True:
            print('Mutli-band iSTFT VITS')
            self.dec = Multiband_iSTFT_Generator(inter_channels, resblock, resblock_kernel_sizes,
                                                 resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
                                                 upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, subbands,
                                                 gin_channels=gin_channels)
        elif ms_istft_vits == True:
            print('Mutli-stream iSTFT VITS')
            self.dec = Multistream_iSTFT_Generator(inter_channels, resblock, resblock_kernel_sizes,
                                                   resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
                                                   upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, subbands,
                                                   gin_channels=gin_channels)
        elif istft_vits == True:
            print('iSTFT-VITS')
            self.dec = iSTFT_Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                                       upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gen_istft_n_fft,
                                       gen_istft_hop_size, gin_channels=gin_channels)
        else:
            print('Decoder Error in json file')

        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16,
                                      gin_channels=gin_channels)
        # self.enc_k = PosteriorEncoder(80, inter_channels, hidden_channels, 5, 1, 16,
        #                               gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        self.enc_spk = SpeakerEncoder(model_hidden_size=gin_channels, model_embedding_size=gin_channels)
        # self.pitch_predictor = StochasticPitchPredictor(in_channels=192 ,filter_channels=256, kernel_size=3,
        #                                                 p_dropout=0.1, n_flows=4, gin_channels=192)
        self.enc_emo=nn.Conv1d(768,256,kernel_size=1)
        self.use_log_pitch = True
        self.stoch_pitch_noise_scale = 1.0
        # self.rhythm_predictor = encoder_2()
        # self.rhythm_predictor.load_state_dict(torch.load("encoder_rhythm.pt", map_location='cpu'))
        self.f0_decoder = F0Decoder(
            1,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            spk_channels=gin_channels
        )

    def forward(self, c, f0, uv, spec, g=None, emotion=None,mel=None, c_lengths=None, spec_lengths=None):
        mel_lengths=None
        if c_lengths == None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        if spec_lengths == None:
            spec_lengths = (torch.ones(spec.size(0)) * spec.size(-1)).to(spec.device)
        if mel_lengths == None:
            mel_lengths = (torch.ones(mel.size(0)) * mel.size(-1)).to(mel.device)
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        #print(x_mask.shape)
        # emotion = self.enc_emo(emotion)
        emotion=emotion.unsqueeze(2)
        emotion = self.enc_emo(emotion)
        g = self.enc_spk(mel.transpose(1, 2))
        g=g.unsqueeze(2)
        # print(g.shape)
        # print(emotion.shape)

        lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
        norm_lf0 = normalize_f0(lf0, x_mask, uv)
        pred_lf0 = self.f0_decoder(c, norm_lf0, x_mask, spk_emb=g)
        # print(pred_lf0)
        # print(norm_lf0)
        # print(lf0)
        _, m_p, logs_p, _ = self.enc_p(c, c_lengths)  # z, m, logs, x_mask
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
        #z_slice, pitch_slice, ids_slice = commons.rand_slice_segments_with_pitch(z, f0, spec_lengths, self.segment_size)
        # rhythm=self.rhythm_predictor(mel)
        #
        # rhythm=torch.concat([rhythm ,rhythm ,rhythm ,rhythm],dim=1)

        z_p = self.flow(z, spec_mask, g=g, emotion=emotion,f0=f0)


        z_slice, ids_slice = commons.rand_slice_segments(z, spec_lengths, self.segment_size)
        o, o_mb = self.dec(z_slice, g=g)

        return o, o_mb, ids_slice, spec_mask,  (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0  #

    def infer(self, c,f0, uv,g=None,emotion=None,mel1=None, mel=None, c_lengths=None):
        mel_lengths = None
        if c_lengths == None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        if mel_lengths == None:
            mel_lengths = (torch.ones(mel.size(0)) * mel.size(-1)).to(mel.device)

        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        emotion=emotion.unsqueeze(2)
        emotion = self.enc_emo(emotion)
        g = self.enc_spk.embed_utterance(mel.transpose(1, 2))
        g = g.unsqueeze(-1)
        # f0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
        # f0=f0.squeeze(1)
        # norm_lf0 = normalize_f0(lf0, x_mask, uv, random_scale=False)
        # pred_lf0 = self.f0_decoder(c, norm_lf0, x_mask, spk_emb=g)
        # f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)
        # rhythm = self.rhythm_predictor(mel1)
        z_p, m_p, logs_p, c_mask = self.enc_p(c, c_lengths)
        # rhythm = torch.concat([rhythm, rhythm, rhythm, rhythm], dim=1)

        z = self.flow(z_p, c_mask, g=g, emotion=emotion,f0=f0, reverse=True)
        o, o_mb = self.dec(z * c_mask, g=g)

        return o,g
