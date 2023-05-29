"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch
"""
import torch
import torch.nn as nn
import math


__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, 3, stride, relu=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostNet(nn.Module):
    def __init__(self, encoder_sets, decoder_sets, cfg, is_train=False, width_mult=1.):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.encoder_sets = encoder_sets
        self.decoder_sets = decoder_sets
        extra = cfg.MODEL.EXTRA
        intermediate_channel = extra.INTERMEDIATE_CHANNELS
        self.use_regress = extra.USE_REGRESS_BRANCH
        self.use_heatmap = extra.USE_HEATMAP_BRANCH
        self.use_gaussion_modulate = extra.USE_DM if "USE_DM" in extra else False
        self.use_brow = False if "USE_BROW" not in cfg.MODEL.EXTRA else cfg.MODEL.EXTRA.USE_BROW

        self.lr_range_test = False
        if 'LR_RANGE_TEST' in cfg:
            self.lr_range_test = True
        use_rgb = cfg.DATASET.COLOR_RGB
        if use_rgb:
            init_channel = 3
        else:
            init_channel = 1

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        layers = [nn.Sequential(
            nn.Conv2d(init_channel, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )]
        input_channel = output_channel

        # building inverted residual blocks
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.encoder_sets:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        layers = []
        for k, exp_size, c, use_se, s in self.decoder_sets:
            layers.append(nn.Upsample(scale_factor=s, mode='bilinear'))
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, 1, use_se))
            input_channel = output_channel
        self.head = nn.Sequential(*layers)

        if self.use_brow:
            final_channel = cfg.MODEL.NUM_EYE_JOINTS + 9
        else:
            final_channel = cfg.MODEL.NUM_EYE_JOINTS
        # building last several layers
        self.final_layer = nn.Conv2d(
                in_channels=input_channel,
                out_channels=final_channel,
                kernel_size=1,
                stride=1,
                padding=0
        )

        if self.use_gaussion_modulate:
            sigma = cfg.MODEL.FACE_SIGMA
            temp_size = sigma * 3
            self.num_joints = cfg.MODEL.NUM_EYE_JOINTS
            size = int(2 * temp_size + 1)
            x = torch.arange(0, size, 1, dtype=torch.float32, requires_grad=False)  #.cuda()  #.to(device)
            y = x[:, None]
            x0 = size // 2
            y0 = size // 2

            g = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
            g = g.unsqueeze(0).unsqueeze(0).expand(self.num_joints, -1, -1, -1)
            if size % 2 == 0:
                self.DM_padding = (size // 2, size // 2 - 1, size // 2, size // 2 - 1)
            else:
                self.DM_padding = (size // 2, size // 2, size // 2, size // 2)
            self.DM_kernel = g
        # self.squeeze = nn.Sequential(
        #     nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(output_channel),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        # )
        # input_channel = output_channel
        #
        # output_channel = intermediate_channel
        # self.classifier = nn.Sequential(
        #     nn.Linear(input_channel, output_channel, bias=False),
        #     nn.BatchNorm1d(output_channel),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(output_channel, cfg.MODEL.NUM_EYE_JOINTS*2),
        # )

        if is_train:
            self._initialize_weights()

    def forward(self, x):
        output = {}
        x = self.features(x)
        x = self.head(x)
        x = self.final_layer(x)
        if self.lr_range_test:
                x_hm_find_max = x.view(x.size(0), x.size(1), -1)
                max_val, max_pos = x_hm_find_max.max(axis=2)
                max_pos = max_pos.int()
        if self.use_gaussion_modulate:
            x = torch.nn.functional.pad(x, self.DM_padding)
            x = torch.nn.functional.conv2d(x, self.DM_kernel,
                                           groups=self.num_joints)
        if self.lr_range_test:
            return x, max_val, max_pos
        output['heatmap'] = x
        return output


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def ghost_net(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    # cfgs = [
    #     # k, t, c, SE, s
    #     [3,  16,  16, 0, 1],
    #     [3,  48,  24, 0, 2],
    #     [3,  72,  24, 0, 1],
    #     [5,  72,  40, 1, 2],
    #     [5, 120,  40, 1, 1],
    #     [3, 240,  80, 0, 2],
    #     [3, 200,  80, 0, 1],
    #     [3, 184,  80, 0, 1],
    #     [3, 184,  80, 0, 1],
    #     [3, 480, 112, 1, 1],
    #     [3, 672, 112, 1, 1],
    #     [5, 672, 160, 1, 2],
    #     [5, 960, 160, 0, 1],
    #     [5, 960, 160, 1, 1],
    #     [5, 960, 160, 0, 1],
    #     [5, 960, 160, 1, 1]
    # ]
    cfgs = [
        # k, t, c, SE, s
        [3,  16,  16, 0, 1],
        [3,  48,  24, 0, 2],
        [3,  72,  24, 0, 1],
        [5,  72,  40, 1, 2],
        [5, 120,  40, 1, 1],
        [3, 240,  80, 0, 2],
        [3, 200,  80, 0, 1],
        [3, 184,  80, 0, 1],
    ]
    return GhostNet(cfgs, **kwargs)


def get_eye_net(cfg, is_train=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        is_train (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    encoder_sets = [
        # k, t,   c,  SE, s
        [3,  16,  16, 0, 1],
        [3,  48,  24, 0, 2],
        [3,  72,  24, 0, 1],
        [5,  72,  40, 1, 2],
        [5, 120,  40, 1, 1],
        [3, 240,  80, 0, 2],
    ]

    encoder_add_sets = [
        # k, t, c, SE, s
        [3, 200,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
    ]

    addtion_encoder_layer = cfg.MODEL.EXTRA.ADDITION_ENCODER_LAYER if "ADDITION_ENCODER_LAYER" in cfg.MODEL.EXTRA else None
    addtion_decoder_layer = cfg.MODEL.EXTRA.ADDITION_DECODER_LAYER if "ADDITION_DECODER_LAYER" in cfg.MODEL.EXTRA else None
    if addtion_encoder_layer:
        addition_layer = cfg.MODEL.EXTRA.ADDITION_ENCODER_LAYER
        assert addition_layer < 6, "Addition layer must less than 6."
        encoder_sets += encoder_add_sets[:addition_layer]

    if addtion_decoder_layer:
        decoder_sets = [
            [4,  184, 80, 1, 2],
            [4,  120, 80, 1, 2],
            [4,  72,  40, 1, 2],
        ]
    else:
        decoder_sets = [
            [4,  120, 40, 1, 2],
            [4,  72,  40, 1, 2],
            [4,  72,  24, 1, 2],
        ]

    model = GhostNet(encoder_sets, decoder_sets, cfg, is_train, **kwargs)
    # ghost_net(cfg, is_train, progress, **kwargs)

    return model

if __name__=='__main__':
    model = ghost_net()
    model.eval()
    print(model)
    input = torch.randn(32,3,224,224)
    y = model(input)
    print(y)
