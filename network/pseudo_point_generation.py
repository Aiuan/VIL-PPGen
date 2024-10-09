import torch
from torch import nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from spconv.pytorch.hash import HashTable
from spconv.pytorch import functional as Fsp

from network.depth_completion import LightBranch
from network.depth2point import D2PConverter
from network.shuffle import PointShuffler
from network.voxelization import Voxelizer
from network.spatial_encode import SparseSpatialEncoder
from network.cost_volume_regularization import SparseCostVolumeRegulator
from network.loss import SmoothL1DepthLoss


class VILPPGen(nn.Module):
    def __init__(self, config, devices):
        super().__init__()

        self.config = config

        self.devices = devices

        self.vis_branch = LightBranch(config=config['vis_branch']).to(devices[0])
        self.inf_branch = LightBranch(config=config['inf_branch']).to(devices[0])

        self.d2p_converter = D2PConverter()

        self.shuffler = PointShuffler()

        self.voxelizer = Voxelizer(config=config['voxelizer'], device=devices[0])

        self.spatial_encoder = SparseSpatialEncoder(config=config['spatial_encoder']).to(devices[0])

        self.cost_volume_regulator = SparseCostVolumeRegulator(config=config['cost_volume_regulator']).to(devices[0])

        self.loss_func = SmoothL1DepthLoss()

    @staticmethod
    def compute_sparse_cost_volume(base: spconv.SparseConvTensor, others: [spconv.SparseConvTensor]):
        # create hash table for base
        batch_size = base.batch_size
        device = base.features.device
        spatial_shape = base.spatial_shape
        table_size = base.features.shape[0] * 2

        table = HashTable(device, torch.int64, torch.int32, table_size)
        scalar_base = Fsp._indice_to_scalar(
            indices=base.indices.long(), shape=[batch_size.item(), *spatial_shape])
        table.insert(
            keys=scalar_base, values=torch.arange(base.indices.shape[0], device=device, dtype=torch.int32))

        outs = []
        feat_base = base.features
        for other in others:
            assert base.batch_size == other.batch_size
            assert base.spatial_shape == other.spatial_shape
            assert base.features.shape[1] == other.features.shape[1]

            # filter other by base, create feat_filtered
            scalar = Fsp._indice_to_scalar(indices=other.indices.long(), shape=[batch_size.item(), *spatial_shape])
            out_indices, flag_error = table.query(scalar)
            out_indices = out_indices.long()
            flag_right = torch.logical_not(flag_error)
            # base.indices[out_indices[flag_right]] == other.indices[flag_right]
            feat_filtered = torch.zeros_like(base.features)
            feat_filtered[out_indices[flag_right], :] += other.features[flag_right, :]

            # calculate variance
            sq_mean = torch.div(torch.add(torch.pow(feat_base, 2), torch.pow(feat_filtered, 2)), 2)
            mean_sq = torch.pow(torch.div(torch.add(feat_base, feat_filtered), 2), 2)
            variance = torch.sub(sq_mean, mean_sq)

            # construct variance sparse tensor
            out = spconv.SparseConvTensor(
                features=variance,
                indices=base.indices,
                spatial_shape=base.spatial_shape,
                batch_size=base.batch_size,
                benchmark=base.benchmark
            )
            out.indice_dict = base.indice_dict
            out.benchmark_record = base.benchmark_record
            out._timer = base._timer
            out.thrust_allocator = base.thrust_allocator

            outs.append(out)

        return tuple(outs)

    @staticmethod
    def extract_confidence_and_offset(feat, vcoords, point2voxel, pcoords, point2pixel,
                                      image_height, image_width, batch_size):
        '''
        :param feat: sparse tensor, note: feat.indices is as same as vcoords, feat.features is (M, 2)
        :param vcoords: voxel coordinates
        :param point2voxel: point to voxel mapping  (N)
        :param pcoords: point coordinates
        :param point2pixel: point to pixel mapping  (N, 2)
        :param image_height:
        :param image_width:
        :param batch_size:
        :return:
        '''
        assert torch.equal(feat.indices, vcoords)

        feat_points = feat.features[point2voxel]

        sp_feat_pixels = spconv.SparseConvTensor(
            features=feat_points,
            indices=torch.cat([pcoords[:, 0:1], point2pixel], dim=1).int(),
            spatial_shape=[image_height, image_width],
            batch_size=batch_size
        )
        feat_pixels = sp_feat_pixels.dense()

        feat_image = F.interpolate(feat_pixels, size=(image_height, image_width), mode='bilinear', align_corners=False)

        confidence, offset = feat_image.chunk(2, dim=1)
        confidence = F.sigmoid(confidence)

        return confidence, offset

    def forward(self, batch_dict, training, need_loss):
        vis_image = batch_dict['vis_image']
        vis_sparse_depth = batch_dict['vis_sparse_depth']
        vis_intrinsic = batch_dict['vis_intrinsic']
        vis_coarse_depth, vis_mf, vis_ddp = self.vis_branch(vis_image, vis_sparse_depth, vis_intrinsic)
        batch_dict['vis_coarse_depth'] = vis_coarse_depth
        batch_dict['vis_mf'] = vis_mf
        batch_dict['vis_ddp'] = vis_ddp

        inf_image = batch_dict['inf_image']
        inf_sparse_depth = batch_dict['inf_sparse_depth']
        inf_intrinsic = batch_dict['inf_intrinsic']
        inf_coarse_depth, inf_mf, inf_ddp = self.inf_branch(inf_image, inf_sparse_depth, inf_intrinsic)
        batch_dict['inf_coarse_depth'] = inf_coarse_depth
        batch_dict['inf_mf'] = inf_mf
        batch_dict['inf_ddp'] = inf_ddp

        vis_extrinsic = batch_dict['vis_extrinsic']
        vis_geopoints, vis_point2pixel = self.d2p_converter(
            depth=vis_coarse_depth,
            intrinsic=vis_intrinsic,
            extrinsic=vis_extrinsic,
            return_point2pixel=True
        )
        batch_dict['vis_geopoints'] = vis_geopoints

        inf_extrinsic = batch_dict['inf_extrinsic']
        inf_geopoints, inf_point2pixel = self.d2p_converter(
            depth=inf_coarse_depth,
            intrinsic=inf_intrinsic,
            extrinsic=inf_extrinsic,
            return_point2pixel=True
        )
        batch_dict['inf_geopoints'] = inf_geopoints

        lid_points = batch_dict['lid_points']
        lid_geopoints = lid_points[:, :4]
        batch_dict['lid_geopoints'] = lid_geopoints

        if training:
            vis_geopoints, vis_point2pixel = self.shuffler(vis_geopoints, vis_point2pixel)
            inf_geopoints, inf_point2pixel = self.shuffler(inf_geopoints, inf_point2pixel)
            lid_geopoints = self.shuffler(lid_geopoints)

        vis_voxels, vis_coords, vis_npoints, vis_point2voxel = self.voxelizer(vis_geopoints, return_point2voxel=True)
        inf_voxels, inf_coords, inf_npoints, inf_point2voxel = self.voxelizer(inf_geopoints, return_point2voxel=True)
        lid_voxels, lid_coords, lid_npoints = self.voxelizer(lid_geopoints)

        vis_geofeatures = self.spatial_encoder(vis_voxels, vis_coords, vis_npoints)
        inf_geofeatures = self.spatial_encoder(inf_voxels, inf_coords, inf_npoints)
        lid_geofeatures = self.spatial_encoder(lid_voxels, lid_coords, lid_npoints)

        vis_inf_cost_volume, vis_lid_cost_volume = self.compute_sparse_cost_volume(
            base=vis_geofeatures, others=[inf_geofeatures, lid_geofeatures]
        )
        inf_vis_cost_volume, inf_lid_cost_volume = self.compute_sparse_cost_volume(
            base=inf_geofeatures, others=[vis_geofeatures, lid_geofeatures]
        )

        vis_inf_cost_volume_reg = self.cost_volume_regulator(vis_inf_cost_volume)
        vis_lid_cost_volume_reg = self.cost_volume_regulator(vis_lid_cost_volume)
        inf_vis_cost_volume_reg = self.cost_volume_regulator(inf_vis_cost_volume)
        inf_lid_cost_volume_reg = self.cost_volume_regulator(inf_lid_cost_volume)

        vis_inf_conf, vis_inf_off = self.extract_confidence_and_offset(
            feat=vis_inf_cost_volume_reg,
            vcoords=vis_coords,
            point2voxel=vis_point2voxel,
            pcoords=vis_geopoints[:, :4],
            point2pixel=vis_point2pixel,
            image_height=vis_image.shape[2],
            image_width=vis_image.shape[3],
            batch_size=vis_image.shape[0]
        )
        batch_dict['vis_inf_conf'] = vis_inf_conf
        batch_dict['vis_inf_off'] = vis_inf_off

        vis_lid_conf, vis_lid_off = self.extract_confidence_and_offset(
            feat=vis_lid_cost_volume_reg,
            vcoords=vis_coords,
            point2voxel=vis_point2voxel,
            pcoords=vis_geopoints[:, :4],
            point2pixel=vis_point2pixel,
            image_height=vis_image.shape[2],
            image_width=vis_image.shape[3],
            batch_size=vis_image.shape[0]
        )
        batch_dict['vis_lid_conf'] = vis_lid_conf
        batch_dict['vis_lid_off'] = vis_lid_off

        inf_vis_conf, inf_vis_off = self.extract_confidence_and_offset(
            feat=inf_vis_cost_volume_reg,
            vcoords=inf_coords,
            point2voxel=inf_point2voxel,
            pcoords=inf_geopoints[:, :4],
            point2pixel=inf_point2pixel,
            image_height=inf_image.shape[2],
            image_width=inf_image.shape[3],
            batch_size=inf_image.shape[0]
        )
        batch_dict['inf_vis_conf'] = inf_vis_conf
        batch_dict['inf_vis_off'] = inf_vis_off

        inf_lid_conf, inf_lid_off = self.extract_confidence_and_offset(
            feat=inf_lid_cost_volume_reg,
            vcoords=inf_coords,
            point2voxel=inf_point2voxel,
            pcoords=inf_geopoints[:, :4],
            point2pixel=inf_point2pixel,
            image_height=inf_image.shape[2],
            image_width=inf_image.shape[3],
            batch_size=inf_image.shape[0]
        )
        batch_dict['inf_lid_conf'] = inf_lid_conf
        batch_dict['inf_lid_off'] = inf_lid_off

        vis_refined_depth = vis_coarse_depth + (1 - vis_inf_conf) * vis_inf_off + (1 - vis_lid_conf) * vis_lid_off
        batch_dict['vis_refined_depth'] = vis_refined_depth

        inf_refined_depth = inf_coarse_depth + (1 - inf_vis_conf) * inf_vis_off + (1 - inf_lid_conf) * inf_lid_off
        batch_dict['inf_refined_depth'] = inf_refined_depth

        vis_points = self.d2p_converter(
            depth=vis_refined_depth,
            intrinsic=vis_intrinsic,
            extrinsic=vis_extrinsic,
            feature=vis_image
        )
        batch_dict['vis_points'] = vis_points

        inf_points = self.d2p_converter(
            depth=inf_refined_depth,
            intrinsic=inf_intrinsic,
            extrinsic=inf_extrinsic,
            feature=inf_image
        )
        batch_dict['inf_points'] = inf_points

        return batch_dict

    def get_loss(self, batch_dict, loss_dict=None):
        if loss_dict is None:
            loss_dict = {}

        loss_dict['loss_vis_coarse_depth'] = self.loss_func(
            preds=batch_dict['vis_coarse_depth'], targets=batch_dict['vis_gt_depth'])

        loss_dict['loss_inf_coarse_depth'] = self.loss_func(
            preds=batch_dict['inf_coarse_depth'], targets=batch_dict['inf_gt_depth'])

        loss_dict['loss_vis_refined_depth'] = self.loss_func(
            preds=batch_dict['vis_refined_depth'], targets=batch_dict['vis_gt_depth'])

        loss_dict['loss_inf_refined_depth'] = self.loss_func(
            preds=batch_dict['inf_refined_depth'], targets=batch_dict['inf_gt_depth'])

        return loss_dict


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from dataset.CustomDataset import CustomDataset
    from utils import read_config
    from utils.load_data import load_batch_dict_to_gpu

    config = read_config('../config/VIL-PPGen.yaml')
    devices = [torch.device('cuda:0')]

    dataset = CustomDataset(config=config['dataset_train_cfg'])

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=dataset.collate_batch
    )

    net = VILPPGen(config=config['network_cfg'], devices=devices)

    batch_dict = next(iter(dataloader))
    batch_dict = load_batch_dict_to_gpu(batch_dict, devices[0])

    batch_dict = net(batch_dict, training=True, need_loss=True)

    loss_dict = net.get_loss(batch_dict)

    print(f'{__file__} done.')
