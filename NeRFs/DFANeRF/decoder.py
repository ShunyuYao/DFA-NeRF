
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from numpy import pi



class DeformationField(nn.Module):
    def __init__(self, dim_embed, dim_signal, hidden_size=64, n_blocks=7,skips=[4],residual=True):
        super().__init__()
        self.dim_embed = dim_embed
        self.dim_signal = dim_signal
        self.skips = skips
        self.residual = residual

        self.blocks_embed = nn.ModuleList([
            nn.Linear(dim_embed + dim_signal, hidden_size)
        ])
        for i in range(n_blocks - 3):
            self.blocks_embed.append(nn.Linear(hidden_size, hidden_size))
        self.out_embed = nn.Linear(hidden_size, dim_embed)

        ### keep signal to be not changed
        # self.blocks_signal = nn.ModuleList([
        #     nn.Linear(dim_embed + dim_signal, hidden_size)
        # ])
        # for i in range(n_blocks - 3):
        #     self.blocks_signal.append(nn.Linear(hidden_size, hidden_size))
        # self.out_signal = nn.Linear(hidden_size, dim_signal)

        n_skips = sum([i in skips for i in range(n_blocks - 1)])
        if n_skips > 0:
            self.fc_embed_skips = nn.ModuleList(
                [nn.Linear(dim_embed, hidden_size) for i in range(n_skips)]
            )
            # self.fc_signal_skips = nn.ModuleList(
            #     [nn.Linear(dim_signal, hidden_size) for i in range(n_skips)]
            # )

        self.act_fn = F.relu

    def forward(self, input):
        embed = input[..., :self.dim_embed]
        signal = input[..., -self.dim_signal:]

        # embed
        skip_idx = 0
        net_embed = input
        for idx, layer in enumerate(self.blocks_embed):
            net_embed = self.act_fn(layer(net_embed))
            if (idx + 1) in self.skips and (idx < len(self.blocks_embed) - 1):
                net_embed = net_embed + self.fc_embed_skips[skip_idx](embed)
                skip_idx += 1
        embed_deformed = self.out_embed(net_embed)
        
        ### keep signal to be not changed
        # signal
        # skip_idx = 0
        # net_signal = input
        # for idx, layer in enumerate(self.blocks_signal):
        #     net_signal = self.act_fn(layer(net_signal))
        #     if (idx + 1) in self.skips and (idx < len(self.blocks_signal) - 1):
        #         net_signal = net_signal + self.fc_signal_skips[skip_idx](signal)
        #         skip_idx += 1
        # signal_deformed = self.out_signal(net_signal)
        if self.residual:
            signal_deformed = torch.zeros_like(signal)
        else:
            signal_deformed = signal

        output = torch.cat((embed_deformed, signal_deformed), -1)
        return output


class DeformationField_ori(nn.Module):
    def __init__(self, dim_embed, dim_signal, hidden_size=64, n_blocks=7,skips=[4]):
        super().__init__()
        self.dim_embed = dim_embed
        self.dim_signal = dim_signal
        self.skips = skips

        self.blocks_embed = nn.ModuleList([
            nn.Linear(dim_embed + dim_signal, hidden_size)
        ])
        for i in range(n_blocks - 3):
            self.blocks_embed.append(nn.Linear(hidden_size, hidden_size))
        self.out_embed = nn.Linear(hidden_size, dim_embed)

        self.blocks_signal = nn.ModuleList([
            nn.Linear(dim_embed + dim_signal, hidden_size)
        ])
        for i in range(n_blocks - 3):
            self.blocks_signal.append(nn.Linear(hidden_size, hidden_size))
        self.out_signal = nn.Linear(hidden_size, dim_signal)

        n_skips = sum([i in skips for i in range(n_blocks - 1)])
        if n_skips > 0:
            self.fc_embed_skips = nn.ModuleList(
                [nn.Linear(dim_embed, hidden_size) for i in range(n_skips)]
            )
            self.fc_signal_skips = nn.ModuleList(
                [nn.Linear(dim_signal, hidden_size) for i in range(n_skips)]
            )

        self.act_fn = F.relu

    def forward(self, input):
        embed = input[..., :self.dim_embed]
        signal = input[..., -self.dim_signal:]

        # embed
        skip_idx = 0
        net_embed = input
        for idx, layer in enumerate(self.blocks_embed):
            net_embed = self.act_fn(layer(net_embed))
            if (idx + 1) in self.skips and (idx < len(self.blocks_embed) - 1):
                net_embed = net_embed + self.fc_embed_skips[skip_idx](embed)
                skip_idx += 1
        embed_deformed = self.out_embed(net_embed)
        
        # signal
        skip_idx = 0
        net_signal = input
        for idx, layer in enumerate(self.blocks_signal):
            net_signal = self.act_fn(layer(net_signal))
            if (idx + 1) in self.skips and (idx < len(self.blocks_signal) - 1):
                net_signal = net_signal + self.fc_signal_skips[skip_idx](signal)
                skip_idx += 1
        signal_deformed = self.out_signal(net_signal)

        output = torch.cat((embed_deformed, signal_deformed), -1)
        return output


class Decoder(nn.Module):
    ''' Decoder class.

    Predicts volume density and color from 3D location, viewing
    direction, and latent code z.

    Args:
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of layers
        n_blocks_view (int): number of view-dep layers
        skips (list): where to add a skip connection
        use_viewdirs: (bool): whether to use viewing directions
        n_freq_posenc (int), max freq for positional encoding (3D location)
        n_freq_posenc_views (int), max freq for positional encoding (
            viewing direction)
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        rgb_out_dim (int): output dimension of feature / rgb prediction
        final_sigmoid_activation (bool): whether to apply a sigmoid activation
            to the feature / rgb output
        downscale_by (float): downscale factor for input points before applying
            the positional encoding
        positional_encoding (str): type of positional encoding
        gauss_dim_pos (int): dim for Gauss. positional encoding (position)
        gauss_dim_view (int): dim for Gauss. positional encoding (
            viewing direction)
        gauss_std (int): std for Gauss. positional encoding
    '''

    def __init__(self, hidden_size=128, n_blocks=8, n_blocks_view=1,dim_signal=64,
                 skips=[4], use_viewdirs=True, n_freq_posenc=10,dim_exp=256,dim_et_embed=42,
                 n_freq_posenc_views=4, use_aud_net=False, dim_aud=64,
                 z_dim=64, rgb_out_dim=3, final_sigmoid_activation=True,
                 downscale_p_by=2., positional_encoding="normal", use_wav2lip=False,dim_w2lfeature=512,
                 gauss_dim_pos=10, gauss_dim_view=4, gauss_std=4., use_deformation_field=False,
                 use_expression=False,
                 **kwargs):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.n_freq_posenc = n_freq_posenc
        self.n_freq_posenc_views = n_freq_posenc_views
        self.skips = skips
        self.downscale_p_by = downscale_p_by
        self.z_dim = z_dim
        self.final_sigmoid_activation = final_sigmoid_activation
        self.n_blocks = n_blocks
        self.n_blocks_view = n_blocks_view
        self.dim_signal = dim_signal
        self.use_deformation_field = use_deformation_field
        self.use_expression = use_expression
        self.use_wav2lip = use_wav2lip

        assert(positional_encoding in ('normal', 'gauss'))
        self.positional_encoding = positional_encoding
        if positional_encoding == 'gauss':
            np.random.seed(42)
            # remove * 2 because of cos and sin
            self.B_pos = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_pos * 3, 3)).float().cuda()
            self.B_view = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_view * 3, 3)).float().cuda()
            dim_embed = 3 * gauss_dim_pos * 2
            dim_embed_view = 3 * gauss_dim_view * 2
        else:
            dim_embed = 3 * self.n_freq_posenc * 2
            dim_embed_view = 3 * self.n_freq_posenc_views * 2


        if use_deformation_field:
            self.deform_net = DeformationField_ori(dim_embed, dim_et_embed)


        if use_expression:
            self.expnet = nn.Linear(dim_exp, hidden_size)
            # self.expnet = nn.Sequential(
            #     nn.Linear(dim_exp, hidden_size),
            #     nn.ReLU(),
            #     nn.Linear(dim_exp, hidden_size),
            # )
        if use_wav2lip:
            self.w2lnet = nn.Linear(dim_w2lfeature, hidden_size)

        # Density Prediction Layers
        self.fc_in = nn.Linear(dim_embed + dim_signal, hidden_size)
        self.fc_in_listener = nn.Linear(dim_embed, hidden_size)
        self.fc_in_torso = nn.Linear(dim_embed + dim_et_embed, hidden_size)

        if z_dim > 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        self.blocks = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)
        ])
        n_skips = sum([i in skips for i in range(n_blocks - 1)])
        if n_skips > 0:
            self.fc_z_skips = nn.ModuleList(
                [nn.Linear(z_dim, hidden_size) for i in range(n_skips)]
            )
            self.fc_p_skips = nn.ModuleList([
                nn.Linear(dim_embed + dim_signal, hidden_size) for i in range(n_skips)
            ])
            self.fc_p_skips_listener = nn.ModuleList([
                nn.Linear(dim_embed, hidden_size) for i in range(n_skips)
            ])
            self.fc_p_skips_torso = nn.ModuleList([
                nn.Linear(dim_embed +  dim_et_embed, hidden_size) for i in range(n_skips)
            ])
        self.sigma_out = nn.Linear(hidden_size, 1)

        # Feature Prediction Layers
        self.fc_z_view = nn.Linear(z_dim, hidden_size)
        self.feat_view = nn.Linear(hidden_size, hidden_size)
        self.fc_view = nn.Linear(dim_embed_view, hidden_size)
        self.feat_out = nn.Linear(hidden_size, rgb_out_dim)
        if use_viewdirs and n_blocks_view > 1:
            self.blocks_view = nn.ModuleList(
                [nn.Linear(dim_embed_view + hidden_size, hidden_size)
                 for i in range(n_blocks_view - 1)])

    def transform_points(self, p, views=False):
        # Positional encoding
        # normalize p between [-1, 1]
        p = p / self.downscale_p_by

        # we consider points up to [-1, 1]
        # so no scaling required here
        if self.positional_encoding == 'gauss':
            B = self.B_view if views else self.B_pos
            p_transformed = (B @ (pi * p.permute(0, 2, 1))).permute(0, 2, 1)
            p_transformed = torch.cat(
                [torch.sin(p_transformed), torch.cos(p_transformed)], dim=-1)
        else:
            L = self.n_freq_posenc_views if views else self.n_freq_posenc
            p_transformed = torch.cat([torch.cat(
                [torch.sin((2 ** i) * pi * p),
                 torch.cos((2 ** i) * pi * p)],
                dim=-1) for i in range(L)], dim=-1)
        return p_transformed

    def forward(self, p_in, ray_d, z_shape=None, z_app=None, signal=None, head_or_torso=None):
        expression = None
        if head_or_torso == 'head':
            if self.use_expression and signal[1] is not None:
                expression = self.expnet(signal[1])
            signal = signal[0]
            
        a = F.relu
        if self.z_dim > 0:
            batch_size = p_in.shape[0]
            if z_shape is None:
                z_shape = torch.randn(batch_size, self.z_dim).to(p_in.device)
            if z_app is None:
                z_app = torch.randn(batch_size, self.z_dim).to(p_in.device)
        p = self.transform_points(p_in)
        # concate signal
        if signal is not None:
            signal = signal.expand(p.shape[1], -1).unsqueeze(0)
            p = torch.cat((p, signal), -1)

        if self.use_deformation_field and head_or_torso == 'torso':
            if True:
                p = self.deform_net(p) + p
            else:
                p = self.deform_net(p)

        if head_or_torso == 'head':
            if signal is not None:
                net = self.fc_in(p)
            else:
                net = self.fc_in_listener(p)
        if head_or_torso == 'torso':
            net = self.fc_in_torso(p)
        if z_shape is not None:
            net = net + self.fc_z(z_shape).unsqueeze(1)
        net = a(net)

        skip_idx = 0
        for idx, layer in enumerate(self.blocks):
            net = a(layer(net))
            if (idx + 1) in self.skips and (idx < len(self.blocks) - 1):
                net = net + self.fc_z_skips[skip_idx](z_shape).unsqueeze(1)
                if head_or_torso == 'head':
                    if signal is not None:
                        net = net + self.fc_p_skips[skip_idx](p)
                    else:
                        net = net + self.fc_p_skips_listener[skip_idx](p)
                if head_or_torso == 'torso':
                    net = net + self.fc_p_skips_torso[skip_idx](p)
                if head_or_torso is None:
                    raise Exception('Do not give head or torso!!')
                skip_idx += 1
        sigma_out = self.sigma_out(net).squeeze(-1)

        net = self.feat_view(net)
        net = net + self.fc_z_view(z_app).unsqueeze(1)
        if expression is not None:
            net = net + expression

        if self.use_viewdirs and ray_d is not None:
            ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            ray_d = self.transform_points(ray_d, views=True)
            net = net + self.fc_view(ray_d)
            net = a(net)
            if self.n_blocks_view > 1:
                for layer in self.blocks_view:
                    net = a(layer(net))
        feat_out = self.feat_out(net)

        if self.final_sigmoid_activation:
            feat_out = torch.sigmoid(feat_out)

        return feat_out, sigma_out
