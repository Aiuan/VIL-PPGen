from torch import nn
import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp


class SparseSpatialEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.spatial_shape = config['spatial_shape']

        in_channels = config['in_channels']
        encode_channels = config['encode_channels']
        decode_channels = config['decode_channels']
        use_bias = config.get('use_bias', False)

        self.encode1 = spconv.SparseSequential(
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, encode_channels[0],
                    kernel_size=3, stride=1, padding=1, bias=use_bias,
                    indice_key='su1'
                ),
                nn.BatchNorm1d(encode_channels[0], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    encode_channels[0], encode_channels[0],
                    kernel_size=3, stride=1, padding=1, bias=use_bias,
                    indice_key='su1'
                ),
                nn.BatchNorm1d(encode_channels[0], eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )

        self.encode2 = spconv.SparseSequential(
            # [256, 2048, 2048] -> [128, 1024, 1024]
            spconv.SparseSequential(
                spconv.SparseConv3d(
                    encode_channels[0], encode_channels[1],
                    kernel_size=3, stride=2, padding=1, bias=use_bias,
                    indice_key='sp2'
                ),
                nn.BatchNorm1d(encode_channels[1], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    encode_channels[1], encode_channels[1],
                    kernel_size=3, stride=1, padding=1, bias=use_bias,
                    indice_key='su2'
                ),
                nn.BatchNorm1d(encode_channels[1], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    encode_channels[1], encode_channels[1],
                    kernel_size=3, stride=1, padding=1, bias=use_bias,
                    indice_key='su2'
                ),
                nn.BatchNorm1d(encode_channels[1], eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )

        self.encode3 = spconv.SparseSequential(
            # [128, 1024, 1024] -> [64, 512, 512]
            spconv.SparseSequential(
                spconv.SparseConv3d(
                    encode_channels[1], encode_channels[2],
                    kernel_size=3, stride=2, padding=1, bias=use_bias,
                    indice_key='sp3'
                ),
                nn.BatchNorm1d(encode_channels[2], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    encode_channels[2], encode_channels[2],
                    kernel_size=3, stride=1, padding=1, bias=use_bias,
                    indice_key='su3'
                ),
                nn.BatchNorm1d(encode_channels[2], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    encode_channels[2], encode_channels[2],
                    kernel_size=3, stride=1, padding=1, bias=use_bias,
                    indice_key='su3'
                ),
                nn.BatchNorm1d(encode_channels[2], eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )

        self.encode4 = spconv.SparseSequential(
            # [64, 512, 512] -> [32, 256, 256]
            spconv.SparseSequential(
                spconv.SparseConv3d(
                    encode_channels[2], encode_channels[3],
                    kernel_size=3, stride=2, padding=1, bias=use_bias,
                    indice_key='sp4'
                ),
                nn.BatchNorm1d(encode_channels[3], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    encode_channels[3], encode_channels[3],
                    kernel_size=3, stride=1, padding=1, bias=use_bias,
                    indice_key='su4'
                ),
                nn.BatchNorm1d(encode_channels[3], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    encode_channels[3], encode_channels[3],
                    kernel_size=3, stride=1, padding=1, bias=use_bias,
                    indice_key='su4'
                ),
                nn.BatchNorm1d(encode_channels[3], eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )

        self.conv4 = spconv.SparseSequential(
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    encode_channels[3], decode_channels[3],
                    kernel_size=1, stride=1, padding=0, bias=use_bias,
                    indice_key='su4'
                ),
                nn.BatchNorm1d(decode_channels[3], eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )

        self.decode3 = spconv.SparseSequential(
            spconv.SparseSequential(
                spconv.SparseInverseConv3d(
                    decode_channels[3], decode_channels[2],
                    kernel_size=3, bias=use_bias, indice_key='sp4'
                ),
                nn.BatchNorm1d(decode_channels[2], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    decode_channels[2], decode_channels[2],
                    kernel_size=3, stride=1, padding=1, bias=use_bias,
                    indice_key='su3'
                ),
                nn.BatchNorm1d(decode_channels[2], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    decode_channels[2], decode_channels[2],
                    kernel_size=3, stride=1, padding=1, bias=use_bias,
                    indice_key='su3'
                ),
                nn.BatchNorm1d(decode_channels[2], eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )

        self.conv3 = spconv.SparseSequential(
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    encode_channels[2], decode_channels[2],
                    kernel_size=1, stride=1, padding=0, bias=use_bias,
                    indice_key='su3'
                ),
                nn.BatchNorm1d(decode_channels[2], eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )

        self.decode2 = spconv.SparseSequential(
            spconv.SparseSequential(
                spconv.SparseInverseConv3d(
                    decode_channels[2], decode_channels[1],
                    kernel_size=3, bias=use_bias, indice_key='sp3'
                ),
                nn.BatchNorm1d(decode_channels[1], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    decode_channels[1], decode_channels[1],
                    kernel_size=3, stride=1, padding=1, bias=use_bias,
                    indice_key='su2'
                ),
                nn.BatchNorm1d(decode_channels[1], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    decode_channels[1], decode_channels[1],
                    kernel_size=3, stride=1, padding=1, bias=use_bias,
                    indice_key='su2'
                ),
                nn.BatchNorm1d(decode_channels[1], eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )

        self.conv2 = spconv.SparseSequential(
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    encode_channels[1], decode_channels[1],
                    kernel_size=1, stride=1, padding=0, bias=use_bias,
                    indice_key='su2'
                ),
                nn.BatchNorm1d(decode_channels[1], eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )

        self.decode1 = spconv.SparseSequential(
            spconv.SparseSequential(
                spconv.SparseInverseConv3d(
                    decode_channels[1], decode_channels[0],
                    kernel_size=3, bias=use_bias, indice_key='sp2'
                ),
                nn.BatchNorm1d(decode_channels[0], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    decode_channels[0], decode_channels[0],
                    kernel_size=3, stride=1, padding=1, bias=use_bias,
                    indice_key='su1'
                ),
                nn.BatchNorm1d(decode_channels[0], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    decode_channels[0], decode_channels[0],
                    kernel_size=3, stride=1, padding=1, bias=use_bias,
                    indice_key='su1'
                ),
                nn.BatchNorm1d(decode_channels[0], eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )

        self.conv1 = spconv.SparseSequential(
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    encode_channels[0], decode_channels[0],
                    kernel_size=1, stride=1, padding=0, bias=use_bias,
                    indice_key='su1'
                ),
                nn.BatchNorm1d(decode_channels[0], eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        )

    def forward(self, voxels, voxel_coordinates, voxel_num_points):
        # calculate features mean
        voxel_features = voxels.sum(dim=1, keepdim=False) / voxel_num_points.reshape((-1, 1))

        batch_size = voxel_coordinates[:, 0].max() + 1

        sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coordinates,
            spatial_shape=self.spatial_shape,
            batch_size=batch_size
        )

        x1 = self.encode1(sp_tensor)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x4 = self.encode4(x3)

        y4 = x4
        f4 = self.conv4(y4)

        y3 = self.decode3(f4)
        f3 = Fsp.sparse_add(y3, self.conv3(x3))

        y2 = self.decode2(f3)
        f2 = Fsp.sparse_add(y2, self.conv2(x2))

        y1 = self.decode1(f2)
        f1 = Fsp.sparse_add(y1, self.conv1(x1))

        return f1
