# FreqTrack: A Frequency-Enhanced Spatiotemporal Network
# Core Implementation of Frequency Domain Convolution (FDConv), KSM, and FBM

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    Efficient activation function for stable feature learning.
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


def get_fft2freq(d1, d2, use_rfft=False):
    """Get the 2D frequency domain coordinates and their distances."""
    freq_h = torch.fft.fftfreq(d1)
    if use_rfft:
        freq_w = torch.fft.rfftfreq(d2)
    else:
        freq_w = torch.fft.fftfreq(d2)

    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w, indexing='ij'), dim=-1)
    dist = torch.norm(freq_hw, dim=-1)
    sorted_dist, indices = torch.sort(dist.view(-1))

    if use_rfft:
        d2 = d2 // 2 + 1
    sorted_coords = torch.stack([indices // d2, indices % d2], dim=-1)

    return sorted_coords.permute(1, 0), freq_hw


class KernelSpatialModulation_Global(nn.Module):
    """
    Kernel Spatial Modulation (KSM) - Global Context
    Extracts global descriptors to adaptively modulate the learnable Fourier weights.
    """

    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16,
                 temp=1.0, kernel_temp=None, kernel_att_init='dyconv_as_extra', att_multi=2.0,
                 ksm_only_kernel_att=False, att_grid=1, stride=1, spatial_freq_decompose=False,
                 act_type='sigmoid'):
        super(KernelSpatialModulation_Global, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = temp
        self.kernel_temp = kernel_temp
        self.ksm_only_kernel_att = ksm_only_kernel_att
        self.att_multi = att_multi
        self.spatial_freq_decompose = spatial_freq_decompose

        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = StarReLU()

        if ksm_only_kernel_att:
            self.func_channel = self.skip
        else:
            if spatial_freq_decompose:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes * 2 if self.kernel_size > 1 else in_planes, 1,
                                            bias=True)
            else:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
            self.func_channel = self.get_channel_attention

        if (in_planes == groups and in_planes == out_planes) or self.ksm_only_kernel_att:
            self.func_filter = self.skip
        else:
            if spatial_freq_decompose:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes * 2, 1, stride=stride, bias=True)
            else:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, stride=stride, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1 or self.ksm_only_kernel_att:
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if hasattr(self, 'spatial_fc'):
            nn.init.normal_(self.spatial_fc.weight, std=1e-6)
        if hasattr(self, 'func_filter') and isinstance(self.func_filter, nn.Conv2d):
            nn.init.normal_(self.func_filter.weight, std=1e-6)
        if hasattr(self, 'kernel_fc') and isinstance(self.kernel_fc, nn.Conv2d):
            nn.init.normal_(self.kernel_fc.weight, std=1e-6)
        if hasattr(self, 'channel_fc') and isinstance(self.channel_fc, nn.Conv2d):
            nn.init.normal_(self.channel_fc.weight, std=1e-6)

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        if self.act_type == 'sigmoid':
            channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2),
                                                                      x.size(-1)) / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            channel_attention = 1 + torch.tanh_(
                self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) / self.temperature)
        else:
            raise NotImplementedError
        return channel_attention

    def get_filter_attention(self, x):
        if self.act_type == 'sigmoid':
            filter_attention = torch.sigmoid(
                self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            filter_attention = 1 + torch.tanh_(
                self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature)
        else:
            raise NotImplementedError
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        if self.act_type == 'sigmoid':
            spatial_attention = torch.sigmoid(spatial_attention / self.temperature) * self.att_multi
        elif self.act_type == 'tanh':
            spatial_attention = 1 + torch.tanh_(spatial_attention / self.temperature)
        else:
            raise NotImplementedError
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        if self.act_type == 'softmax':
            kernel_attention = F.softmax(kernel_attention / self.kernel_temp, dim=1)
        elif self.act_type == 'sigmoid':
            kernel_attention = torch.sigmoid(kernel_attention / self.kernel_temp) * 2 / kernel_attention.size(1)
        elif self.act_type == 'tanh':
            kernel_attention = (1 + torch.tanh(kernel_attention / self.kernel_temp)) / kernel_attention.size(1)
        else:
            raise NotImplementedError
        return kernel_attention

    def forward(self, x):
        avg_x = self.relu(self.bn(self.fc(x)))
        return self.func_channel(avg_x), self.func_filter(avg_x), self.func_spatial(avg_x), self.func_kernel(avg_x)


class FrequencyBandModulation(nn.Module):
    """
    Frequency Band Modulation (FBM)
    Explicitly amplifies critical high-frequency edges and suppresses low-frequency background noise.
    """

    def __init__(self, in_channels, k_list=[2, 4, 8], lowfreq_att=False, act='sigmoid',
                 spatial_group=1, spatial_kernel=3, init='zero', **kwargs):
        super().__init__()
        self.k_list = k_list
        self.in_channels = in_channels
        self.spatial_group = spatial_group if spatial_group <= 64 else in_channels
        self.lowfreq_att = lowfreq_att
        self.act = act

        self.freq_weight_conv_list = nn.ModuleList()
        _n = len(k_list)
        if lowfreq_att: _n += 1

        for i in range(_n):
            freq_weight_conv = nn.Conv2d(in_channels=in_channels, out_channels=self.spatial_group,
                                         stride=1, kernel_size=spatial_kernel,
                                         groups=self.spatial_group, padding=spatial_kernel // 2, bias=True)
            if init == 'zero':
                nn.init.normal_(freq_weight_conv.weight, std=1e-6)
                freq_weight_conv.bias.data.zero_()
            self.freq_weight_conv_list.append(freq_weight_conv)

    def sp_act(self, freq_weight):
        if self.act == 'sigmoid':
            return freq_weight.sigmoid() * 2
        elif self.act == 'tanh':
            return 1 + freq_weight.tanh()
        elif self.act == 'softmax':
            return freq_weight.softmax(dim=1) * freq_weight.shape[1]
        raise NotImplementedError

    def forward(self, x, att_feat=None):
        if att_feat is None: att_feat = x
        x_list = []
        x_dtype = x.dtype
        x = x.to(torch.float32)
        pre_x = x.clone()
        b, _, h, w = x.shape

        x_fft = torch.fft.rfft2(x, norm='ortho')

        for idx, freq in enumerate(self.k_list):
            mask = torch.zeros_like(x_fft[:, 0:1, :, :], device=x.device)
            _, freq_indices = get_fft2freq(d1=h, d2=w, use_rfft=True)
            freq_indices = freq_indices.max(dim=-1)[0]
            mask[:, :, freq_indices < 0.5 / freq] = 1.0

            low_part = torch.fft.irfft2(x_fft * mask, s=(h, w), dim=(-2, -1), norm='ortho').real
            high_part = pre_x - low_part
            pre_x = low_part

            freq_weight = self.sp_act(self.freq_weight_conv_list[idx](att_feat))
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group, -1, h,
                                                                                           w)
            x_list.append(tmp.reshape(b, -1, h, w))

        if self.lowfreq_att:
            freq_weight = self.sp_act(self.freq_weight_conv_list[len(x_list)](att_feat))
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        else:
            x_list.append(pre_x)

        return sum(x_list).to(x_dtype)


class FDConv(nn.Conv2d):
    """
    Standard Frequency Domain Convolution (FDConv)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, reduction=0.0625, kernel_num=16,
                 use_fdconv_if_c_gt=16, use_fdconv_if_k_in=[1, 3], use_fbm_if_k_in=[3],
                 param_ratio=1, param_reduction=1.0, att_multi=2.0, **kwargs):
        p = autopad(kernel_size, None)
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=p,
                         **kwargs)

        self.use_fdconv_if_c_gt = use_fdconv_if_c_gt
        self.use_fdconv_if_k_in = use_fdconv_if_k_in
        self.kernel_num = kernel_num or (out_channels // 2)
        self.param_ratio = param_ratio
        self.param_reduction = param_reduction
        self.att_multi = att_multi

        if min(in_channels, out_channels) <= use_fdconv_if_c_gt or self.kernel_size[0] not in use_fdconv_if_k_in:
            return

        self.KSM_Global = KernelSpatialModulation_Global(
            in_channels, out_channels, self.kernel_size[0], groups=self.groups,
            temp=math.sqrt(self.kernel_num * self.param_ratio), reduction=reduction,
            kernel_num=self.kernel_num * self.param_ratio, att_multi=att_multi
        )

        if self.kernel_size[0] in use_fbm_if_k_in:
            self.FBM = FrequencyBandModulation(in_channels)

    def forward(self, x):
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt or self.kernel_size[
            0] not in self.use_fdconv_if_k_in:
            return super().forward(x)

        if hasattr(self, 'FBM'):
            x = self.FBM(x)
        return super().forward(x)


class CoordAtt(nn.Module):
    """
    Coordinate Attention.
    Replaces GAP to preserve spatial location information for tiny objects.
    """

    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = StarReLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out.mean(dim=[2, 3], keepdim=True)


class KernelSpatialModulation_Global_VisDrone(nn.Module):
    """
    Enhanced KSM optimized for UAV/VisDrone datasets.
    """

    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4,
                 min_channel=16, temp=1.0, kernel_temp=None, att_multi=2.0, stride=1, act_type='sigmoid'):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = temp
        self.kernel_temp = kernel_temp
        self.att_multi = att_multi
        self.act_type = act_type

        # Use Coordinate Attention to retain positional awareness of small objects
        self.coord_att = CoordAtt(in_planes, in_planes, reduction=16)

        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = StarReLU()

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, stride=stride, bias=True)
        if kernel_size > 1:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
        if kernel_num > 1:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)

    def forward(self, x):
        global_context = self.coord_att(x)
        avg_x = self.relu(self.bn(self.fc(global_context)))

        chan_att = torch.sigmoid(self.channel_fc(
            avg_x) / self.temperature) * self.att_multi if self.act_type == 'sigmoid' else self.channel_fc(avg_x)
        filt_att = torch.sigmoid(self.filter_fc(
            avg_x) / self.temperature) * self.att_multi if self.act_type == 'sigmoid' else self.filter_fc(avg_x)

        spat_att = 1.0
        if self.kernel_size > 1:
            spat_att = self.spatial_fc(avg_x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
            spat_att = torch.sigmoid(
                spat_att / self.temperature) * self.att_multi if self.act_type == 'sigmoid' else spat_att

        kern_att = 1.0
        if self.kernel_num > 1:
            kern_att = self.kernel_fc(avg_x).view(x.size(0), -1, 1, 1, 1, 1)
            kern_att = torch.sigmoid(
                kern_att / self.kernel_temp) * 2 / self.kernel_num if self.act_type == 'sigmoid' else F.softmax(
                kern_att / self.kernel_temp, dim=1)

        return chan_att, filt_att, spat_att, kern_att


class FrequencyBandModulation_VisDrone(nn.Module):
    """
    Enhanced FBM optimized for UAV/VisDrone datasets.
    Extracts and actively enhances high-frequency signatures.
    """

    def __init__(self, in_channels, k_list=[2], spatial_group=1, spatial_kernel=3, max_size=(128, 128)):
        super().__init__()
        self.k_list = k_list
        self.spatial_group = spatial_group

        self.high_freq_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            StarReLU(),
            nn.Conv2d(in_channels, spatial_group, kernel_size=1)
        )

        self.register_buffer('cached_masks', self._precompute_masks(max_size, k_list), persistent=False)

    def _precompute_masks(self, max_size, k_list):
        max_h, max_w = max_size
        _, freq_indices = get_fft2freq(d1=max_h, d2=max_w, use_rfft=True)
        freq_indices = freq_indices.abs().max(dim=-1)[0]
        masks = [freq_indices < 0.5 / freq + 1e-8 for freq in k_list]
        return torch.stack(masks, dim=0).unsqueeze(1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_float = x.to(torch.float32)
        x_fft = torch.fft.rfft2(x_float, norm='ortho')

        freq_h, freq_w = h, w // 2 + 1
        current_masks = F.interpolate(self.cached_masks.float(), size=(freq_h, freq_w), mode='nearest')

        mask_lowest = current_masks[0]
        low_part = torch.fft.irfft2(x_fft * mask_lowest, s=(h, w), norm='ortho')
        high_part = x_float - low_part

        high_freq_att = self.high_freq_enhance(high_part).sigmoid() * 2.0
        high_part_enhanced = high_part * high_freq_att

        return low_part + high_part_enhanced


class FDConv_VisDrone(FDConv):
    """
    Our definitive implementation of FDConv for UAV tracking.
    Overrides global KSM and FBM with the improved VisDrone variants.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if hasattr(self, 'KSM_Global'):
            self.KSM_Global = KernelSpatialModulation_Global_VisDrone(
                self.in_channels, self.out_channels, self.kernel_size[0],
                groups=self.groups, reduction=kwargs.get('reduction', 0.0625),
                kernel_num=self.kernel_num * self.param_ratio,
                att_multi=self.att_multi, stride=self.stride
            )

        fbm_cfg = kwargs.get('fbm_cfg', {})
        if hasattr(self, 'FBM'):
            self.FBM = FrequencyBandModulation_VisDrone(
                self.in_channels,
                k_list=fbm_cfg.get('k_list', [2]),
                max_size=(128, 128)
            )