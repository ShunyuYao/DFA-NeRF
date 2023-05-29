import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn
logger = logging.getLogger(__name__)

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

BN_MOMENTUM = 0.1
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DReLU(nn.Module):

    def __init__(self, input_channel, k=2, type='B',
                 init_param_alpha=[1.0, 0.0],
                 init_param_beta=[0.0, 0.0],
                 init_param_gamma=[1.0, 0.5], ratio=8):

        super(DReLU, self).__init__()
        assert k == len(init_param_alpha), 'length of init param alpha should equal {}'.format(k)
        assert k == len(init_param_beta), 'length of init param beta should equal {}'.format(k)
        assert len(init_param_gamma) == 2, 'length of init param gamma should equal 2'

        self.hidden_dim = input_channel // ratio
        self.hidden_dim = 8 if (self.hidden_dim < 8) else self.hidden_dim

        self.type = type
        self.input_channel = input_channel
        self.k = k  # only support k=2
        self.init_param_alpha = nn.Parameter(torch.tensor(init_param_alpha, dtype=torch.float32).view(1, 1, k), requires_grad = False)
        self.init_param_beta = nn.Parameter(torch.tensor(init_param_beta, dtype=torch.float32).view(1, 1, k), requires_grad = False)
        self.init_param_gamma = nn.Parameter(torch.tensor(init_param_gamma, dtype=torch.float32), requires_grad = False)

        if self.type == 'A':
            output_channel = 2 * k
        elif self.type in ['B', 'C']:
            output_channel = 2 * k * input_channel
        else:
            raise Exception('type shoulf be in A,B,C')

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                    nn.Linear(input_channel, input_channel // ratio),
                    nn.ReLU(inplace=True),
                    nn.Linear(input_channel // ratio, output_channel),
                    )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.size()
        assert c == self.input_channel, "input channel {} should match the setted channel {}.".format(c, input_channel)
        output = 0

        if self.type == 'B':
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y)  # shape (b, 2 * c * k)
            y = torch.tensor(2.0, dtype=torch.float32) * torch.sigmoid(y) - torch.tensor(1.0, dtype=torch.float32)  # normailze

            y = y.view(b, 2, c, self.k)
            alpha = y[:, 0, :, :]  # (b, c, k)
            beta = y[:, 1, :, :]
            # self.init_param_alpha and self.init_param_beta: (1, 1, k)
            alpha = self.init_param_alpha + self.init_param_gamma[0] * alpha
            beta = self.init_param_beta + self.init_param_gamma[1] * beta

            #  (b, c, k, 1)  (b, c, 1, h*w) -> (b, c, k, h*w)
            alpha = alpha.unsqueeze(3)
            beta = beta.unsqueeze(3)
            x_view = x.view(b, c, 1, h*w)
            output = x_view * alpha + beta

            # maximum
            output, _ = torch.max(output, dim=2)
            output = output.view(b, c, h, w)

        else:
            raise NotImplementedError

        return output


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            DReLU(out_planes)
            # nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, cfg, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        # # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(self.last_channel, num_classes),
        # )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        # x = self.intermediate_layers(x)
        # x = x.flatten(start_dim=1)
        # x = self.fc_layers(x)
        return x

    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.mean([2, 3])
    #     x = self.classifier(x)
    #     return x

class FaceMobilenet(nn.Module):

    def __init__(self, cfg, is_train=False, progress=True, **kwargs):
        super(FaceMobilenet, self).__init__()

        backbone = MobileNetV2(cfg, **kwargs)
        extra = cfg.MODEL.EXTRA
        intermediate_channel = extra.INTERMEDIATE_CHANNELS
        self.use_regress = extra.USE_REGRESS_BRANCH
        self.use_heatmap = extra.USE_HEATMAP_BRANCH
        self.last_channel = backbone.last_channel
        self.inplanes = self.last_channel
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.use_aux_head = extra.USE_AUX_HEAD

        self.c3_last_channel = 96
        input_size = cfg.MODEL.IMAGE_SIZE[0]
        down_sample_rate = extra.DOWN_SAMPLE
        assert input_size % down_sample_rate == 0, 'Input size must be divisible by {}'.format(down_sample_rate)
        self.output_feats_size = input_size // down_sample_rate

        if is_train:
            state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                                  progress=progress)
            backbone.load_state_dict(state_dict, strict=False)
        self.before_layer3 = backbone.features[:14]
        self.layer4 = backbone.features[14:]

        if self.use_regress:
            # self.intermediate_layers = nn.Sequential(
            #     ConvBNReLU(self.last_channel, 320, kernel_size=1),
            #     ConvBNReLU(320, 64, kernel_size=1)
            #     )
            self.fc_layers = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, cfg.MODEL.NUM_FACE_JOINTS*2),
                )

        if self.use_heatmap:
            # used for deconv layers
            self.deconv_layers = self._make_deconv_layer(
                extra.NUM_DECONV_LAYERS,
                extra.NUM_DECONV_FILTERS,
                extra.NUM_DECONV_KERNELS,
            )
            self.deconv_final_layer = nn.Conv2d(
                in_channels=extra.NUM_DECONV_FILTERS[-1],
                out_channels=cfg.MODEL.NUM_FACE_JOINTS,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
            )

        self.inplanes = self.c3_last_channel
        if self.use_aux_head:
            # used for deconv layers
            self.aux_deconv_layers = self._make_deconv_layer(
                extra.NUM_DECONV_LAYERS - 1,
                extra.NUM_DECONV_FILTERS[1:],
                extra.NUM_DECONV_KERNELS[1:],
            )
            self.aux_deconv_final_layer = nn.Conv2d(
                in_channels=extra.NUM_DECONV_FILTERS[-1],
                out_channels=cfg.MODEL.NUM_FACE_JOINTS,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
            )

    def forward(self, x):
        c3 = self.before_layer3(x)
        c4 = self.layer4(c3)
        outputs = {}

        if self.use_regress:
            # x_fc = self.intermediate_layers(c4)
            x_fc = c4.mean([2, 3])
            x_fc = self.fc_layers(x_fc)
            outputs['regress'] = x_fc

        if self.use_heatmap:
            x_hm = self.deconv_layers(c4)
            x_hm = self.deconv_final_layer(x_hm)
            outputs['heatmap'] = x_hm

        if self.use_aux_head:
            x_aux = self.aux_deconv_layers(c3)
            x_aux = self.aux_deconv_final_layer(x_aux)
            outputs['heatmap_aux'] = x_aux

        return outputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def init_heatmap_weights(self):
        logger.info('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                logger.info('=> init {}.weight as 1'.format(name))
                logger.info('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        logger.info('=> init final conv weights from normal distribution')
        for m in self.deconv_final_layer.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


def get_face_net(cfg, is_train=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        is_train (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = FaceMobilenet(cfg, is_train, progress, **kwargs)

    return model
