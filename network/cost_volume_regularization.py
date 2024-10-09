from torch import nn
import spconv.pytorch as spconv


class SparseCostVolumeRegulator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        in_channels = config['in_channels']
        filter_channels = config['filter_channels']
        out_channels = config['out_channels']
        use_bias = config.get('use_bias', False)

        channels = [in_channels] + filter_channels
        layers = []
        for i in range(len(channels) - 1):
            layers.append(
                spconv.SparseSequential(
                    spconv.SubMConv3d(
                        channels[i], channels[i + 1],
                        kernel_size=3, stride=1, padding=1, bias=use_bias,
                        indice_key='su1'
                    ),
                    nn.BatchNorm1d(channels[i + 1], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                )
            )

        # output layer
        layers.append(
            spconv.SparseSequential(
                spconv.SubMConv3d(
                    channels[-1], out_channels,
                    kernel_size=1, stride=1, padding=0, bias=use_bias,
                    indice_key='su1'
                )
            )
        )

        self.layers = spconv.SparseSequential(*layers)

    def forward(self, sp_tensor):
        out = self.layers(sp_tensor)
        return out
