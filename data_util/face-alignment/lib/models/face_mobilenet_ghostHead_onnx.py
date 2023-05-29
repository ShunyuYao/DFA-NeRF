import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn
from .mobile_head import MobileHead
logger = logging.getLogger(__name__)

from .eye_ghostnet import GhostBottleneck
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


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
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
        img_channel = cfg.MODEL.EXTRA.IMG_CHANNEL if "IMG_CHANNEL" in cfg.MODEL.EXTRA else 3

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
        features = [ConvBNReLU(img_channel, input_channel, stride=2)]
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


class FaceMobilenet(nn.Module):

    def __init__(self, cfg, is_train=False, progress=True, **kwargs):
        super(FaceMobilenet, self).__init__()

        backbone = MobileNetV2(cfg, **kwargs)
        extra = cfg.MODEL.EXTRA
        self.use_regress = extra.USE_REGRESS_BRANCH
        self.use_heatmap = extra.USE_HEATMAP_BRANCH
        self.last_channel = backbone.last_channel
        self.inplanes = self.last_channel
        self.deconv_with_bias = extra.DECONV_WITH_BIAS if "DECONV_WITH_BIAS" in extra else False
        self.use_aux_head = extra.USE_AUX_HEAD
        self.use_boundary_map = extra.USE_BOUNDARY_MAP if "USE_BOUNDARY_MAP" in cfg.MODEL.EXTRA else False
        self.use_stage4 = extra.USE_STAGE4 if "USE_STAGE4" in cfg.MODEL.EXTRA else False

        self.c3_last_channel = 96
        input_size = cfg.MODEL.IMAGE_SIZE[0]
        self.negative_example = cfg.FACE_DATASET.NEGATIVE_EXAMPLE
        self.use_gaussion_modulate = extra.USE_DM if "USE_DM" in extra else False
        self.use_background_hm = cfg.MODEL.EXTRA.USE_BACKGROUND_HM if "USE_BACKGROUND_HM" in cfg.MODEL.EXTRA else False
        self.use_brow = False if "USE_BROW" not in cfg.MODEL.EXTRA else cfg.MODEL.EXTRA.USE_BROW
        # down_sample_rate = extra.DOWN_SAMPLE
        # assert input_size % down_sample_rate == 0, 'Input size must be divisible by {}'.format(down_sample_rate)
        # self.output_feats_size = input_size // down_sample_rate
        if "DENSE_REGRESSION" in cfg.MODEL.EXTRA:
            self.dense_regression = cfg.MODEL.EXTRA.DENSE_REGRESSION
        else:
            self.dense_regression = None

        self.lr_range_test = False
        if 'LR_RANGE_TEST' in cfg:
            self.lr_range_test = True

        use_pretrain = extra.USE_PRETRAIN if "USE_PRETRAIN" in extra else True
        if is_train and use_pretrain:
            state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                                  progress=progress)
            backbone.load_state_dict(state_dict, strict=False)
        self.before_layer3 = backbone.features[:14]
        if self.use_stage4:
            self.layer4 = backbone.features[14:]

        final_fc_channel = cfg.MODEL.NUM_FACE_JOINTS*2

        if self.negative_example:
            final_fc_channel += 1

        if self.use_regress:
            # self.intermediate_layers = nn.Sequential(
            #     ConvBNReLU(self.last_channel, 320, kernel_size=1),
            #     ConvBNReLU(320, 64, kernel_size=1)
            #     )
            if self.dense_regression:
                self.last_regress_layer = nn.Conv2d(self.last_channel, final_fc_channel, 1)
            else:
                self.fc_layers = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(self.last_channel, final_fc_channel),
                    )

        if self.use_heatmap:
            output_channel = cfg.MODEL.NUM_FACE_JOINTS
            if self.use_boundary_map:
                output_channel = output_channel + 1
            if self.use_background_hm:
                output_channel = output_channel + 1

            if self.use_brow:
                output_channel += 9
            if not self.use_stage4:
                self.inplanes = self.c3_last_channel
            # _make_ghostModule_head(self, num_layers, exp_sizes, num_filters, num_kernels, use_uses, strides)
            dilations = extra.DILATION if "DILATION" in extra else None
            self.head_layers = self._make_ghostModule_head(
                extra.NUM_LAYERS,
                extra.EXP_SIZE,
                extra.NUM_FILTERS,
                extra.NUM_KERNELS,
                extra.USE_SE,
                extra.STRIDES,
                dilations
            )

            self.num_head_layers = len(extra.NUM_LAYERS)
            self.head_final_layer = nn.Conv2d(
                in_channels=extra.NUM_FILTERS[-1],
                out_channels=output_channel,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
            )

        if self.use_aux_head:
            # used for deconv layers
            self.aux_head_layers = self._make_deconv_layer(
                extra.NUM_DECONV_LAYERS - 1,
                extra.NUM_DECONV_FILTERS[1:],
                extra.NUM_DECONV_KERNELS[1:],
            )
            self.aux_head_final_layer = nn.Conv2d(
                in_channels=extra.NUM_DECONV_FILTERS[-1],
                out_channels=cfg.MODEL.NUM_FACE_JOINTS,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
            )

        if self.use_gaussion_modulate:
            sigma = cfg.MODEL.FACE_SIGMA
            temp_size = sigma * 3
            self.num_joints = cfg.MODEL.NUM_FACE_JOINTS
            size = int(2 * temp_size + 1)
            x = torch.arange(0, size, 1, dtype=torch.float32, requires_grad=False) #.cuda()  #.to(device)
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

        # if self.negative_example:
        #     self.negative_example_layter = nn.Linear()

    def forward(self, x):
        out = self.before_layer3(x)
        if self.use_stage4:
            out = self.layer4(out)
        outputs = {}

        # if self.use_regress:
        #     # x_fc = self.intermediate_layers(c4)
        #     if self.dense_regression:
        #         x_fc = self.last_regress_layer(c4)
        #     else:
        #         x_fc = c4.mean([2, 3])
        #         x_fc = self.fc_layers(x_fc)
        #     if self.lr_range_test:
        #         return x_fc
        #     outputs['regress'] = x_fc

        output_range_test = []
        if self.use_heatmap:
            for i in range(self.num_head_layers):
                out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
                out = self.head_layers[i+1](out)
            x_hm = self.head_final_layer(out)
            if self.lr_range_test:
                if self.use_boundary_map:
                    x_hm = x_hm[:, :-1, ...]
                x_hm_find_max = x_hm.view(x_hm.size(0), x_hm.size(1), -1)
                max_val, max_pos = x_hm_find_max.max(axis=2)
                max_pos = max_pos.int()
                output_range_test.append(max_val)
                output_range_test.append(max_pos)
                # print("x_hm_max: ", max_val, max_arg)
            if self.use_gaussion_modulate:
                heatmaps_reshaped = x_hm.view(x_hm.size(0), self.num_joints, -1)
                heatmaps_max = heatmaps_reshaped.max(2)[0].unsqueeze(-1).unsqueeze(-1)
                x_hm = torch.nn.functional.pad(x_hm, self.DM_padding)
                x_hm = torch.nn.functional.conv2d(x_hm, self.DM_kernel,
                                                  groups=self.num_joints)
                output_range_test.append(x_hm)
                # x_hm = x_hm.view(x_hm.size(0), self.num_joints, -1)
            if self.lr_range_test:
                return output_range_test
            outputs['heatmap'] = x_hm

        if self.use_aux_head:
            x_aux = self.aux_head_layers(out)
            x_aux = self.aux_head_final_layer(x_aux)
            outputs['heatmap_aux'] = x_aux

        return outputs

    def _make_ghostModule_head(self, num_layers, exp_sizes, num_filters, num_kernels, use_uses, strides, dilations=None):
        assert len(num_layers) == len(exp_sizes), \
            'ERROR: len(num_layers)  is different len(exp_sizes)'
        assert len(num_layers) == len(num_filters), \
            'ERROR: len(num_layers)  is different len(num_filters)'
        assert len(num_layers) == len(num_kernels), \
            'ERROR: len(num_layers)  is different len(num_kernels)'
        assert len(num_layers) == len(use_uses), \
            'ERROR: len(num_layers)  is different len(use_use)'
        assert len(num_layers) == len(strides), \
            'ERROR: len(num_layers)  is different len(strides)'
        # assert len(num_layers) == len(dilations), \
        #     'ERROR: len(num_layers)  is different len(dilations)'
        layers = []
        use_dilations = True if dilations else False

        for i in range(len(num_layers)):
            num_layer = num_layers[i]
            exp_size = exp_sizes[i]
            out_dim = num_filters[i]
            kernel_size = num_kernels[i]
            use_se = use_uses[i]
            stride = strides[i]
            if use_dilations:
                dilation = dilations[i]
            else:
                dilation = 1

            # if stride == 2:
            #     layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            if stride == 2:
                layers.append(nn.Sequential())
            for j in range(num_layer):
                layers.append(GhostBottleneck(self.inplanes, exp_size, out_dim, kernel_size, 1, use_se, dilation=dilation))
                self.inplanes = out_dim

        return nn.Sequential(*layers)

    def init_heatmap_weights(self):
        logger.info('=> init deconv weights from normal distribution')
        # for name, m in self.head_layers.named_modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        logger.info('=> init final conv weights from normal distribution')
        for m in self.head_final_layer.modules():
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

    if is_train and cfg.MODEL.INIT_WEIGHTS and cfg.MODEL.EXTRA.USE_HEATMAP_BRANCH:
        model.init_heatmap_weights()

    return model
