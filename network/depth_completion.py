import torch
from torch import nn
import torch.nn.functional as F


class LightBranch(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = config['in_channels']
        base_channels = config['base_channels']
        use_bias = config.get('use_bias', False)

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.encode_layer1 = GeoEncoder(base_channels, base_channels * 2, 2, use_bias)
        self.encode_layer2 = GeoEncoder(base_channels * 2, base_channels * 2, 1, use_bias)
        self.encode_layer3 = GeoEncoder(base_channels * 2, base_channels * 4, 2, use_bias)
        self.encode_layer4 = GeoEncoder(base_channels * 4, base_channels * 4, 1, use_bias)
        self.encode_layer5 = GeoEncoder(base_channels * 4, base_channels * 8, 2, use_bias)
        self.encode_layer6 = GeoEncoder(base_channels * 8, base_channels * 8, 1, use_bias)
        self.encode_layer7 = GeoEncoder(base_channels * 8, base_channels * 16, 2, use_bias)
        self.encode_layer8 = GeoEncoder(base_channels * 16, base_channels * 16, 1, use_bias)
        self.encode_layer9 = GeoEncoder(base_channels * 16, base_channels * 32, 2, use_bias)
        self.encode_layer10 = GeoEncoder(base_channels * 32, base_channels * 32, 1, use_bias)

        self.decode_layer8 = Decoder(base_channels * 32, base_channels * 16, 5, 2, 2, 1, use_bias)
        self.decode_layer6 = Decoder(base_channels * 16, base_channels * 8, 5, 2, 2, 1, use_bias)
        self.decode_layer4 = Decoder(base_channels * 8, base_channels * 4, 5, 2, 2, 1, use_bias)
        self.decode_layer2 = Decoder(base_channels * 4, base_channels * 2, 5, 2, 2, 1, use_bias)
        self.decode_layer0 = Decoder(base_channels * 2, base_channels, 5, 2, 2, 1, use_bias)

        filter_channels = config['filter_channels']
        output_channels = config['output_channels']
        output_layer = []
        for i in range(len(filter_channels) - 1):
            output_layer.append(
                nn.Sequential(
                    nn.Conv2d(filter_channels[i], filter_channels[i + 1],
                              kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(filter_channels[i + 1]),
                    nn.ReLU(inplace=True)
                )
            )
        output_layer.append(
            nn.Conv2d(filter_channels[-1], output_channels,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.output_layer = nn.Sequential(*output_layer)

        self.depth_min = config['depth_min']
        self.depth_max = config['depth_max']

    @staticmethod
    def get_geometry_feature(img, d, K, stride):
        b, _, h, w = img.shape
        device = img.device

        v, u = torch.meshgrid(
            [torch.arange(h, device=device), torch.arange(w, device=device)],
            indexing='ij'
        )

        u = u.float().unsqueeze(0).expand(b, -1, -1).unsqueeze(1)
        v = v.float().unsqueeze(0).expand(b, -1, -1).unsqueeze(1)

        u_sampled = F.avg_pool2d(u, kernel_size=stride, stride=stride)
        v_sampled = F.avg_pool2d(v, kernel_size=stride, stride=stride)

        mask = (d > 0).float()
        mask_sampled = F.max_pool2d(mask, kernel_size=stride, stride=stride)

        v_inf = 1e3
        d_sampled = d + (1 - mask) * v_inf  # set invalid depth to a large value
        d_sampled = - F.max_pool2d(-d_sampled, kernel_size=stride, stride=stride)  # min pooling
        d_sampled = d_sampled * mask_sampled  # set invalid depth to 0

        fx = K[:, 0, 0].reshape(b, 1, 1, 1)
        cx = K[:, 0, 2].reshape(b, 1, 1, 1)
        fy = K[:, 1, 1].reshape(b, 1, 1, 1)
        cy = K[:, 1, 2].reshape(b, 1, 1, 1)

        x_sampled = d_sampled * (u_sampled - cx) / fx
        y_sampled = d_sampled * (v_sampled - cy) / fy
        z_sampled = d_sampled

        geo_feature = torch.cat([x_sampled, y_sampled, z_sampled], dim=1)

        return geo_feature

    def forward(self, img, d, K):
        '''
        :param img: (B, C, H, W)
        :param d: (B, 1, H, W)
        :param K: (B, 3, 3)
        :return:
        '''
        geo1_s1 = self.get_geometry_feature(img, d, K, 1)
        geo2_s2 = self.get_geometry_feature(img, d, K, 2)
        geo3_s4 = self.get_geometry_feature(img, d, K, 4)
        geo4_s8 = self.get_geometry_feature(img, d, K, 8)
        geo5_s16 = self.get_geometry_feature(img, d, K, 16)
        geo6_s32 = self.get_geometry_feature(img, d, K, 32)

        enfeat0_s1 = self.input_layer(torch.cat([img, d], dim=1))

        enfeat1_s2 = self.encode_layer1(enfeat0_s1, geo1_s1, geo2_s2)
        enfeat2_s2 = self.encode_layer2(enfeat1_s2, geo2_s2, geo2_s2)
        enfeat3_s4 = self.encode_layer3(enfeat2_s2, geo2_s2, geo3_s4)
        enfeat4_s4 = self.encode_layer4(enfeat3_s4, geo3_s4, geo3_s4)
        enfeat5_s8 = self.encode_layer5(enfeat4_s4, geo3_s4, geo4_s8)
        enfeat6_s8 = self.encode_layer6(enfeat5_s8, geo4_s8, geo4_s8)
        enfeat7_s16 = self.encode_layer7(enfeat6_s8, geo4_s8, geo5_s16)
        enfeat8_s16 = self.encode_layer8(enfeat7_s16, geo5_s16, geo5_s16)
        enfeat9_s32 = self.encode_layer9(enfeat8_s16, geo5_s16, geo6_s32)
        enfeat10_s32 = self.encode_layer10(enfeat9_s32, geo6_s32, geo6_s32)

        defeat8_s16 = self.decode_layer8(enfeat10_s32)
        feature8_s16 = enfeat8_s16 + defeat8_s16

        defeat6_s8 = self.decode_layer6(feature8_s16)
        feature6_s8 = enfeat6_s8 + defeat6_s8

        defeat4_s4 = self.decode_layer4(feature6_s8)
        feature4_s4 = enfeat4_s4 + defeat4_s4

        defeat2_s2 = self.decode_layer2(feature4_s4)
        feature2_s2 = enfeat2_s2 + defeat2_s2

        defeat0_s1 = self.decode_layer0(feature2_s2)
        feature0_s1 = enfeat0_s1 + defeat0_s1

        mf = self.output_layer(feature0_s1)

        ddp = F.softmax(mf, dim=1)
        samples = torch.linspace(self.depth_min, self.depth_max, ddp.shape[1], device=ddp.device)
        samples = samples.view(1, -1, 1, 1)
        depth = torch.sum(ddp * samples, dim=1, keepdim=True)

        return depth, mf, ddp


class GeoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_bias=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.geo_channels = 3

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels + self.geo_channels, out_channels,
                      kernel_size=3, stride=stride, padding=1, bias=use_bias),
            nn.BatchNorm2d(out_channels)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels + self.geo_channels, out_channels,
                      kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(out_channels)
        )

        self.fuse_layer = None
        if stride > 1 or in_channels != out_channels:
            self.fuse_layer = nn.Sequential(
                nn.Conv2d(in_channels + self.geo_channels, out_channels,
                          kernel_size=1, stride=stride, padding=0, bias=use_bias),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature, geo1, geo2):
        x = torch.cat([feature, geo1], dim=1)

        out = self.layer1(x)

        out = self.relu(out)

        out = torch.cat([out, geo2], dim=1)

        out = self.layer2(out)

        if self.fuse_layer is not None:
            y = self.fuse_layer(x)
            out += y
        else:
            out += feature

        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=5, stride=2, padding=2, output_padding=1, use_bias=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, output_padding=output_padding, bias=use_bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
