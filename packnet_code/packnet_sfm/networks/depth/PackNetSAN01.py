# CC BY-NC-SA 4.0 License
# Copyright 2024 Samsung Israel R&D Center (SIRC), based on:
# https://github.com/TRI-ML/packnet-sfm - Toyota Research Institute

import torch
import torch.nn as nn
import cv2
# import matplotlib.pyplot as plt

from packnet_code.packnet_sfm.networks.layers.minkowski_encoder import MinkowskiEncoder
from packnet_code.packnet_sfm.networks.layers.packnet.layers01 import PackLayerConv3d, UnpackLayerConv3d, Conv2D, ResidualBlock, InvDepth

output_feature_maps = False

def save_channel_to_image(channel, name):
    channel_norm = channel + channel.min()
    channel_norm = (channel_norm / channel_norm.max()) * 255
    channel_numpy = channel_norm.detach().cpu().numpy()[0, 0, :, :]
    # plt.imshow(channel_numpy)
    cv2.imwrite(name + '.png', channel_numpy)

class PackNetSlimEnc01(nn.Module):
    def __init__(self, version, in_channels, ni, n1, n2, n3, n4, n5,
                 pack_kernel, num_blocks, num_3d_feat, dropout):
        super().__init__()
        # Encoder
        self.version = version

        self.pre_calc = Conv2D(in_channels, ni, 5, 1)

        self.pack1 = PackLayerConv3d(n1, pack_kernel[0], d=num_3d_feat)
        self.pack2 = PackLayerConv3d(n2, pack_kernel[1], d=num_3d_feat)
        self.pack3 = PackLayerConv3d(n3, pack_kernel[2], d=num_3d_feat)
        self.pack4 = PackLayerConv3d(n4, pack_kernel[3], d=num_3d_feat)
        self.pack5 = PackLayerConv3d(n5, pack_kernel[4], d=num_3d_feat)

        self.conv1 = Conv2D(ni, n1, 7, 1)
        self.conv2 = ResidualBlock(n1, n2, num_blocks[0], 1, dropout=dropout)
        self.conv3 = ResidualBlock(n2, n3, num_blocks[1], 1, dropout=dropout)
        self.conv4 = ResidualBlock(n3, n4, num_blocks[2], 1, dropout=dropout)
        self.conv5 = ResidualBlock(n4, n5, num_blocks[3], 1, dropout=dropout)

    def forward(self, rgb):

        x = self.pre_calc(rgb)

        # Encoder

        x1 = self.conv1(x)
        x1p = self.pack1(x1)
        x2 = self.conv2(x1p)
        x2p = self.pack2(x2)
        x3 = self.conv3(x2p)
        x3p = self.pack3(x3)
        x4 = self.conv4(x3p)
        x4p = self.pack4(x4)
        x5 = self.conv5(x4p)
        x5p = self.pack5(x5)

        # Skips
        return x5p, [x, x1p, x2p, x3p, x4p]


class Decoder(nn.Module):
    def __init__(self, version, out_channels, ni, n1, n2, n3, n4, n5,
                 unpack_kernel, iconv_kernel, num_3d_feat):
        super().__init__()
        # Decoder
        self.version = version

        n1o, n1i = n1, n1 + ni + out_channels
        n2o, n2i = n2, n2 + n1 + out_channels
        n3o, n3i = n3, n3 + n2 + out_channels
        n4o, n4i = n4, n4 + n3
        n5o, n5i = n5, n5 + n4

        self.unpack5 = UnpackLayerConv3d(n5, n5o, unpack_kernel[0], d=num_3d_feat)
        self.unpack4 = UnpackLayerConv3d(n5, n4o, unpack_kernel[1], d=num_3d_feat)
        self.unpack3 = UnpackLayerConv3d(n4, n3o, unpack_kernel[2], d=num_3d_feat)
        self.unpack2 = UnpackLayerConv3d(n3, n2o, unpack_kernel[3], d=num_3d_feat)
        self.unpack1 = UnpackLayerConv3d(n2, n1o, unpack_kernel[4], d=num_3d_feat)

        self.iconv5 = Conv2D(n5i, n5, iconv_kernel[0], 1)
        self.iconv4 = Conv2D(n4i, n4, iconv_kernel[1], 1)
        self.iconv3 = Conv2D(n3i, n3, iconv_kernel[2], 1)
        self.iconv2 = Conv2D(n2i, n2, iconv_kernel[3], 1)
        self.iconv1 = Conv2D(n1i, n1, iconv_kernel[4], 1)

        # Depth Layers

        self.unpack_disps = nn.PixelShuffle(2)
        self.unpack_disp4 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        self.disp4_layer = InvDepth(n4, out_channels=out_channels)
        self.disp3_layer = InvDepth(n3, out_channels=out_channels)
        self.disp2_layer = InvDepth(n2, out_channels=out_channels)
        self.disp1_layer = InvDepth(n1, out_channels=out_channels)

    def forward(self, x5p, skips):
        skip1, skip2, skip3, skip4, skip5 = skips

        unpack5 = self.unpack5(x5p)
        if self.version == 'A':
            concat5 = torch.cat((unpack5, skip5), 1)
        else:
            concat5 = unpack5 + skip5
        iconv5 = self.iconv5(concat5)

        unpack4 = self.unpack4(iconv5)
        if self.version == 'A':
            concat4 = torch.cat((unpack4, skip4), 1)
        else:
            concat4 = unpack4 + skip4
        iconv4 = self.iconv4(concat4)
        inv_depth4 = self.disp4_layer(iconv4)
        up_inv_depth4 = self.unpack_disp4(inv_depth4)

        unpack3 = self.unpack3(iconv4)
        if self.version == 'A':
            concat3 = torch.cat((unpack3, skip3, up_inv_depth4), 1)
        else:
            concat3 = torch.cat((unpack3 + skip3, up_inv_depth4), 1)
        iconv3 = self.iconv3(concat3)
        inv_depth3 = self.disp3_layer(iconv3)
        up_inv_depth3 = self.unpack_disp3(inv_depth3)

        unpack2 = self.unpack2(iconv3)
        if self.version == 'A':
            concat2 = torch.cat((unpack2, skip2, up_inv_depth3), 1)
        else:
            concat2 = torch.cat((unpack2 + skip2, up_inv_depth3), 1)
        iconv2 = self.iconv2(concat2)
        inv_depth2 = self.disp2_layer(iconv2)
        up_inv_depth2 = self.unpack_disp2(inv_depth2)


        unpack1 = self.unpack1(iconv2)
        if self.version == 'A':
            concat1 = torch.cat((unpack1, skip1, up_inv_depth2), 1)
        else:
            concat1 = torch.cat((unpack1 + skip1, up_inv_depth2), 1)
        iconv1 = self.iconv1(concat1)
        inv_depth1 = self.disp1_layer(iconv1)

        # if self.training:
        inv_depths = [inv_depth1, inv_depth2, inv_depth3, inv_depth4]
        # else:
        #     inv_depths = [inv_depth1]

        return inv_depths


class PackNetSAN01(nn.Module):
    """
    PackNet-SAN network, from the paper (https://arxiv.org/abs/2103.16690)

    Parameters
    ----------
    dropout : float
        Dropout value to use
    version : str
        Has a XY format, where:
        X controls upsampling variations (not used at the moment).
        Y controls feature stacking (A for concatenation and B for addition)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, dropout=None, version=None, freeze_encoder=False,
                 freeze_decoder=False, freeze_san=False, input_channels=3,
                 is_depth_aux_net=False, output_channels=1, **kwargs):
        super().__init__()
        self.version = version[1:]
        # Input/output channels
        self.in_channels = input_channels
        out_channels = output_channels
        # Hyper-parameters
        ni, n1, n2, n3, n4, n5 = 32, 32, 64, 128, 256, 512
        num_blocks = [2, 2, 3, 3]
        pack_kernel = [5, 3, 3, 3, 3]
        unpack_kernel = [3, 3, 3, 3, 3]
        iconv_kernel = [3, 3, 3, 3, 3]
        num_3d_feat = 4
        # freeze_encoder = True
        # freeze_decoder = True
        # freeze_san = True

        self.encoder = PackNetSlimEnc01(self.version, self.in_channels, ni, n1, n2, n3, n4, n5,
                                        pack_kernel, num_blocks, num_3d_feat, dropout)
        if freeze_encoder:
            self.freeze_weights(self.encoder.parameters())

        self.decoder = Decoder(self.version, out_channels, ni, n1, n2, n3, n4, n5,
                               unpack_kernel, iconv_kernel, num_3d_feat)
        if freeze_decoder:
            self.freeze_weights(self.decoder.parameters())

        self.mconvs = MinkowskiEncoder([n1, n2, n3, n4, n5], with_uncertainty=False)
        # self.weight = torch.nn.parameter.Parameter(torch.ones(5), requires_grad=True)
        # self.bias = torch.nn.parameter.Parameter(torch.zeros(5), requires_grad=True)
        if freeze_san:
            self.freeze_weights(self.mconvs.parameters())
            # self.weight = self.weight.detach()
            # self.bias = self.bias.detach()
            # self.freeze_weights(self.weight)
            # self.freeze_weights(self.bias)

        self.weight = torch.nn.parameter.Parameter(torch.ones(5), requires_grad=not freeze_san)
        self.bias = torch.nn.parameter.Parameter(torch.zeros(5), requires_grad=not freeze_san)

        self.init_weights()

    def init_weights(self):
        """Initializes network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def freeze_weights(self, net_to_freeze):

        for param in net_to_freeze:
            param.requires_grad = False


    def run_network(self, rgb, input_depth=None, output_features=False, return_encoded_rgb=False, encoded_rgb=None):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        if encoded_rgb is None:
            x5p, skips = self.encoder(rgb)
            encoded_rgb_data = (x5p, skips)
        else:
            x5p, skips = encoded_rgb


        # self.save_channel_to_image(skips[0][:,0:1,:,:], 'new_chan')

        #
        # channel = skips[4][:, 20:21, :, :]
        # channel_norm = channel + channel.min()
        # channel_norm = (channel_norm/channel_norm.max())*255
        # self.save_channel_to_image(channel_norm, 'new_chan_small_3')

        if input_depth is not None:
            self.mconvs.prep(input_depth)

            mconvs_1 = self.mconvs(skips[1])
            mconvs_2 = self.mconvs(skips[2])

            skips[1] = skips[1] * self.weight[0].view(1, 1, 1, 1) + mconvs_1 + self.bias[0].view(1, 1, 1, 1)
            skips[2] = skips[2] * self.weight[1].view(1, 1, 1, 1) + mconvs_2 + self.bias[1].view(1, 1, 1, 1)
            skips[3] = skips[3] * self.weight[2].view(1, 1, 1, 1) + self.mconvs(skips[3]) + self.bias[2].view(1, 1, 1, 1)
            skips[4] = skips[4] * self.weight[3].view(1, 1, 1, 1) + self.mconvs(skips[4]) + self.bias[3].view(1, 1, 1, 1)
            x5p      = x5p      * self.weight[4].view(1, 1, 1, 1) + self.mconvs(x5p)      + self.bias[4].view(1, 1, 1, 1)

        decoder_output = self.decoder(x5p, skips)
        skips_x5p = skips + [x5p]
        output_var = [decoder_output, skips_x5p]

        if return_encoded_rgb: # For backward compatibility
            output_var.append(encoded_rgb_data)

        if self.is_depth_aux_net:
            input_depth_aux_tensor = torch.cat((decoder_output[0], skips_x5p[0]), axis=1)
            depth_aux_output = self.depth_aux_net(input_depth_aux_tensor)
            output_var.append(depth_aux_output)

        return output_var

    def forward(self, rgb, input_depth=None, rgb_edge = None, output_features=False, **kwargs):

        # edge classification = 4 channels
        if self.in_channels == 4:
            rgb_encoder_input = torch.cat((rgb,rgb_edge), dim=1) # same but more readable and common to use
        elif self.in_channels == 3:
            rgb_encoder_input = rgb

        if not self.training:
            return_arr = self.run_network(rgb_encoder_input, input_depth)
            # inv_depths = return_arr[0]
            inv_depths = return_arr
            if self.in_channels == 4:
                # should it be like this? seems to be only on the first image before
                # (previously it was  inv_depths = return_arr[0])
                inv_depths[0] = (inv_depths[0] * rgb_edge)

            return {
                'inv_depths': inv_depths,
            }

        output = {}

        return_arr = self.run_network(rgb_encoder_input)
        inv_depths_rgb = return_arr[0]
        skip_feat_rgb = return_arr[1]
        # encoded_rgb_data = return_arr[2]
        if self.is_depth_aux_net:
            depth_rgb_aux_output = return_arr[2]
            output['depth_rgb_aux_output'] = depth_rgb_aux_output

        # if self.in_channels==4:
        #     inv_depths_rgb[0] = (inv_depths_rgb[0] * rgb_edge)
        output['inv_depths'] = inv_depths_rgb

        if output_features:
            if 'target_rgb' in kwargs:
                return_arr = self.run_network(kwargs['target_rgb'])
                target_inv_depths_rgb = return_arr[0]
                target_skip_feat_rgb = return_arr[1]

                output['target_skip_feat_rgb'] = target_skip_feat_rgb
                output['target_inv_depths_rgb'] = target_inv_depths_rgb
            output['skip_feat_rgb'] = skip_feat_rgb

        if input_depth is None:
            return {
                'inv_depths': inv_depths_rgb,
            }

        return_arr = self.run_network(rgb_encoder_input, input_depth)
        inv_depths_rgbd = return_arr[0]
        skip_feat_rgbd = return_arr[1]
        if self.is_depth_aux_net:
            depth_rgbd_aux_output = return_arr[2]
            output['depth_rgbd_aux_output'] = depth_rgbd_aux_output

        # if self.in_channels==4:
        #     inv_depths_rgbd[0] = (inv_depths_rgbd[0] * rgb_edge)
        output['inv_depths_rgbd'] = inv_depths_rgbd

        loss = sum([((srgbd.detach() - srgb) ** 2).mean()
                    for srgbd, srgb in zip(skip_feat_rgbd, skip_feat_rgb)]) / len(skip_feat_rgbd)
        # Its L2 loss between the skip features of the unet (with and without the lidar for same samples)
        output['depth_loss'] = loss

        if output_features:
            if 'target_rgb' in kwargs and 'target_input_depth' in kwargs:
                return_arr = self.run_network(kwargs['target_rgb'], kwargs['target_input_depth'])
                target_inv_depths_rgbd = return_arr[0]
                target_skip_feat_rgbd = return_arr[1]
                output['target_skip_feat_rgbd'] = target_skip_feat_rgbd
                output['target_inv_depths_rgbd'] = target_inv_depths_rgbd
            output['skip_feat_rgbd'] = skip_feat_rgbd

        return output
