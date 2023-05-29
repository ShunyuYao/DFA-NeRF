import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

import logging
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


from .mobile_block import InvertedResidual

class MobileHead(nn.Module):
    def __init__(self, cfg, inp, out, settings, upsample_scale=2, Relu='relu6'):
        super(MobileHead, self).__init__()
        extra = cfg.MODEL.EXTRA
        if not settings:
            self.setting = [[3, 256, 3, 2],
                            [3, 128, 3, 2],
                            [3, 64, 3, 1]]
        else:
            self.setting = settings
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        self.Relu = Relu
        self.mobile_module = self._mobile_layer(inp, upsample_scale)

        self.final_layer = nn.Conv2d(
                in_channels=self.setting[-1][1],
                out_channels=out,
                kernel_size=1,
                stride=1,
                padding=0
        )

        # self._init_weights()
        logger.info('mobilev2 head module init weight')
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

    def forward(self, input):
        x = self.mobile_module(input)
        output = self.final_layer(x)

        return output

    def _mobile_layer(self, inp, upsample_scale):
        layers = []
        '''
        setting = [[3, 256, 5, 2],
                    [3, 128, 5, 1],
                    [3, 128, 5, 1]]'''

        inplanes = inp
        for expand_ratio, outplanes, kernel_size, nums in self.setting:
            layers.append(nn.Upsample(scale_factor=upsample_scale, mode='bilinear'))
            for i in range(nums):
                layers.append(InvertedResidual(inplanes, outplanes, kernel_size, 1, 1, expand_ratio, nn.BatchNorm2d, self.Relu))
                inplanes = outplanes

        return nn.Sequential(*layers)

    # def _init_weights(self):
    #     '''
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             # m.weight.data.normal_(0, math.sqrt(2. / n))
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, SynchronizedBatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #     '''
    #     for name, m in self.named_modules():
    #         if isinstance(m, nn.ConvTranspose2d):
    #             logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
    #             logger.info('=> init {}.bias as 0'.format(name))
    #             nn.init.normal_(m.weight, std=0.001)
    #             if self.deconv_with_bias:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             logger.info('=> init {}.weight as 1'.format(name))
    #             logger.info('=> init {}.bias as 0'.format(name))
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #     logger.info('=> init final conv weights from normal distribution')
