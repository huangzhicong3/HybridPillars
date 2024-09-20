import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import torch.nn.functional as F
from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_utils
from pcdet.utils.spconv_utils import spconv
from functools import partial
from pcdet.models.backbones_3d.spconv_backbone import SparseBasicBlock, post_act_block


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part])
                               for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x, inplace=True)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PALayer(nn.Module):
    def __init__(self, dim_pa, reduction_pa):
        super(PALayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_pa, dim_pa // reduction_pa),
            nn.ReLU(),
            nn.Linear(dim_pa // reduction_pa, dim_pa)
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(1 + 3, 1)
        # )

    def forward(self, x):
        batch, _, b, w = x.size()
        y = torch.max(x, dim=1, keepdim=True)[0].view(batch, 1, b, w)
        out1 = self.fc(y)  # .view(batch, 1, b, w)
        return out1

class CALayer(nn.Module):
    def __init__(self, dim_ca, reduction_ca):
        super(CALayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_ca, dim_ca // reduction_ca),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ca // reduction_ca, dim_ca)
        )

    def forward(self, x):
        batch, c, b, _ = x.size()
        y = torch.max(x, dim=3, keepdim=True)[0].view(batch, c, b)
        y = self.fc(y.permute(0, 2, 1)).permute(0, 2, 1).view(batch, c, b, 1)
        return y

# Point-wise attention for each voxel
class PACALayer_1(nn.Module):
    def __init__(self, dim_ca, dim_pa, reduction_r, mode="both"):
        super(PACALayer_1, self).__init__()
        self.mode = mode
        if mode == "both":
            self.pa = PALayer(dim_pa,  reduction_r)
            self.ca = CALayer(dim_ca,  reduction_r)
        elif mode == "pa":
            self.pa = PALayer(dim_pa,  reduction_r)
        elif mode == "ca":
            self.ca = CALayer(dim_ca,  reduction_r)
        self.sig = nn.Sigmoid()
        self.fc = nn.Conv2d(2*dim_ca, dim_ca, kernel_size=1, bias=False)

    def forward(self, x):
        if self.mode == "both":
            pa_weight = self.sig(self.pa(x))
            ca_weight = self.sig(self.ca(x))
            paca_weight = torch.mul(pa_weight, ca_weight)
        elif self.mode == "pa":
            pa_weight = self.sig(self.pa(x))
            paca_weight = pa_weight
        elif self.mode == "ca":
            ca_weight = self.sig(self.ca(x))
            paca_weight = ca_weight
        paca_normal_weight = paca_weight
        out = torch.mul(x, paca_normal_weight)
        voxel_features = torch.cat([x, out], dim=1)
        out = self.fc(voxel_features)
        return out, paca_normal_weight

class PACALayer_2(nn.Module):
    def __init__(self, dim_ca, dim_pa, reduction_r, mode="both"):
        super(PACALayer_2, self).__init__()
        self.mode = mode
        if mode == "both":
            self.pa = PALayer(dim_pa,  reduction_r)
            self.ca = CALayer(dim_ca,  reduction_r)
        elif mode == "pa":
            self.pa = PALayer(dim_pa,  reduction_r)
        elif mode == "ca":
            self.ca = CALayer(dim_ca,  reduction_r)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        if self.mode == "both":
            pa_weight = self.sig(self.pa(x))
            ca_weight = self.sig(self.ca(x))
            paca_weight = torch.mul(pa_weight, ca_weight)
        elif self.mode == "pa":
            pa_weight = self.sig(self.pa(x))
            paca_weight = pa_weight
        elif self.mode == "ca":
            ca_weight = self.sig(self.ca(x))
            paca_weight = ca_weight
        paca_normal_weight = paca_weight
        out = torch.mul(x, paca_normal_weight)
        return out, paca_normal_weight


class PFA_Mapper(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features_in = model_cfg.NUM_BEV_FEATURES_IN
        self.point_cloud_range = np.array(model_cfg.POINT_CLOUD_RANGE)
        self.voxel_size = np.array(model_cfg.VOXEL_SIZE)
        self.max_points_per_voxel = model_cfg.MAX_POINTS_PER_VOXEL
        self.feature_idx = model_cfg.FEATURES_IDX
        self.groupers = nn.ModuleList()
        self.pfn_layers = nn.ModuleList()

        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)
        self.voxel_x = self.voxel_size[0]
        self.voxel_y = self.voxel_size[1]
        self.voxel_z = self.voxel_size[2]
        self.x_offset = self.voxel_x / 2 + self.point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + self.point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + self.point_cloud_range[2]

        self.projection = 'pfn'

        radius = self.voxel_size[0]
        assert (self.voxel_size[0] == self.voxel_size[1])
        self.num_bev_features_sp = 0
        for k in range(len(self.feature_idx)):
            nsample = self.max_points_per_voxel[k]
            self.groupers.append(pointnet2_utils.PillarQueryAndGroup_voxel(radius, nsample, use_xyz=True))
            self.paca_mode = model_cfg.get('PACA_MODE', 'both')

            grouper_channel_out = self.num_bev_features_in[k] + 6
            if self.paca_mode != 'none':
                self.paca_layer_1 = PACALayer_1(grouper_channel_out, nsample, 1, mode=self.paca_mode)
                self.paca_layer_2 = PACALayer_2(grouper_channel_out, nsample, 1, mode=self.paca_mode)
            
            # grouper_channel_out *= 2
            if self.projection == 'pfn':
                self.num_filters = self.model_cfg.NUM_FILTERS[k]
                num_filters = [grouper_channel_out] + list(self.num_filters)
                pfn_layers = nn.ModuleList()
                for i in range(len(num_filters) - 1):
                    in_filters = num_filters[i]
                    out_filters = num_filters[i + 1]
                    pfn_layers.append(
                        PFNLayer(in_filters, out_filters, True, last_layer=(i >= len(num_filters) - 2))
                    )
                self.pfn_layers.append(pfn_layers)
                

                if 'HEIGHT_MAP' in self.model_cfg.keys():
                    self.height_map_channel = self.model_cfg.HEIGHT_MAP
                else:
                    self.height_map_channel = 0
                self.num_bev_features_sp += out_filters + self.height_map_channel

            elif self.projection == 'psa':
                mlp_spec = self.model_cfg.NUM_FILTERS
                mlp_spec[0] += 3

                shared_mlps = []
                for k in range(len(mlp_spec) - 1):
                    shared_mlps.extend([
                        nn.Conv2d(mlp_spec[k], mlp_spec[k + 1],
                                  kernel_size=1, bias=False),
                        nn.BatchNorm2d(mlp_spec[k + 1]),
                        nn.ReLU()
                    ])
                self.mlps = (nn.Sequential(*shared_mlps))
                self.num_bev_features_sp = mlp_spec[-1]

        self.nx, self.ny, self.nz = self.grid_size

        nf = self.num_bev_features_sp

        sp_nf = model_cfg.PRE_CONV.NUM_FILTERS
        self.sparse_shape = self.grid_size[::-1]
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.use_sparse = model_cfg.PRE_CONV.get('SPARSE', True)
        if self.use_sparse is True:
            self.conv_input = spconv.SparseSequential(
                spconv.SubMConv3d(nf, nf, 3, padding=1, bias=False, indice_key='subm1'),
                norm_fn(nf),
                nn.ReLU(inplace=True),
            )
            block = post_act_block
            self.conv1 = spconv.SparseSequential(
                block(nf, nf, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
            )

            self.conv2 = spconv.SparseSequential(
                block(nf, sp_nf[0], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
                block(sp_nf[0], sp_nf[0], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
                block(sp_nf[0], sp_nf[0], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            )
        else:

            self.conv_input = nn.Sequential(
                nn.Conv2d(nf, nf//4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(nf//4, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            )
            self.conv1 = nn.Sequential(
                nn.Conv2d(nf//4, nf//2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(nf//2, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(nf//2, sp_nf[0], kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(sp_nf[0], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
                nn.Conv2d(sp_nf[0], sp_nf[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(sp_nf[0], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            )
        self.num_bev_features = sp_nf[0]
        # self.pillar_to_bev = pillar_to_bev.apply
        # self.pillar_query_op = pillar_query_op.apply



    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def locate_features_occupation(self, coords, grid_size):
        nx, ny, nz = grid_size
        batch_spatial_occupation = []
        batch_size = coords.shape[0]

        for batch_idx in range(batch_size):
            this_coords = coords[batch_idx].type(torch.int)
            indices = this_coords[:, 2] * nx + this_coords[:, 1]
            indices = indices.unique()
            # occupation[:, indices] += 1
            # batch_spatial_occupation.append(occupation.squeeze(0))
            if indices.shape[0] < this_coords.shape[0]:
                l = this_coords.shape[0] - indices.shape[0]
                zeros = torch.zeros([l,], dtype=indices.dtype, device=indices.device)
            batch_spatial_occupation.append(torch.cat([indices, zeros], dim=0))
        return batch_spatial_occupation

    def forward(self, batch_dict, **kwargs):
        feature_idx = self.feature_idx[-1]
        nx, ny, nz = self.grid_size
        coords = batch_dict['encoder_coords'][feature_idx].clone()
        coords[:, :, 1] += -self.point_cloud_range[0]
        coords[:, :, 1] *= nx / (self.point_cloud_range[3] - self.point_cloud_range[0])  # x
        coords[:, :, 2] += -self.point_cloud_range[1]
        coords[:, :, 2] *= ny / (self.point_cloud_range[4] - self.point_cloud_range[1])  # y
        coords[:, :, 1] = torch.clamp(coords[:, :, 1], max=nx-1)
        coords[:, :, 2] = torch.clamp(coords[:, :, 2], max=ny-1)

        batch_occupation = self.locate_features_occupation(coords, grid_size=[nx, ny, nz])

        bev_pixel_coords_l = []
        bev_cartesian_coords_l = []
        batch_size = batch_dict['batch_size']

        for b in range(batch_size):
            occupied_idx = batch_occupation[b].unsqueeze(1)
            occupied_idx_y = occupied_idx // nx
            occupied_idx_x = occupied_idx - occupied_idx_y * nx

            occupied_idx_z = torch.zeros_like(occupied_idx_x)
            occupied_idx_b = torch.ones_like(occupied_idx_x) * b

            occupied_coords = torch.cat([occupied_idx_b, occupied_idx_z, occupied_idx_y, occupied_idx_x],
                                        dim=1)  # b, z, y, x
            bev_pixel_coords_l.append(occupied_coords)

            cartesian_idx_x = occupied_idx_x * self.voxel_size[0] + self.voxel_size[0] / 2 + self.point_cloud_range[0]
            cartesian_idx_y = occupied_idx_y * self.voxel_size[1] + self.voxel_size[1] / 2 + self.point_cloud_range[1]
            cartesian_idx_z = occupied_idx_z * self.voxel_size[2] + self.voxel_size[2] / 2 + self.point_cloud_range[2]
            cartesian_idx = torch.cat([cartesian_idx_x, cartesian_idx_y, cartesian_idx_z], dim=1)  # x, y, z
            bev_cartesian_coords_l.append(cartesian_idx)

        # implement the voxelization with query
        pillar_features = []
        for k in range(len(self.groupers)):
            feature_idx = self.feature_idx[k]
            batch_xyz = batch_dict['encoder_coords'][feature_idx][:, :, 1:4].contiguous()  # B, N, 3
            batch_features = batch_dict['encoder_features'][feature_idx]  # B, C, N

            batch_voxel_features = []
            batch_pt_count = []
            for b in range(batch_size):
                xyz = batch_xyz[b].unsqueeze(0)
                new_xyz = bev_cartesian_coords_l[b].unsqueeze(0)  # B, M, 3
                features = batch_features[b].unsqueeze(0)
                voxel_features, pt_count = self.groupers[k](xyz, new_xyz, features)
                # self attention
                if self.paca_mode != 'none':
                    out_1, _ = self.paca_layer_1(voxel_features)
                    out_2, _ = self.paca_layer_2(out_1)
                    voxel_features = out_1 + out_2
                batch_voxel_features.append(voxel_features.squeeze(0).permute(1, 2, 0))
                batch_pt_count.append(pt_count.squeeze(0))

            bev_pixel_coords = torch.cat(bev_pixel_coords_l, 0)
            batch_voxel_features = torch.cat(batch_voxel_features, 0)
            batch_pt_count = torch.cat(batch_pt_count, 0)
            # print(batch_pt_count.max())

            voxel_num_points = batch_pt_count
            coords = bev_pixel_coords
            # pillar VFE
            if self.projection == 'pfn':
                features = batch_voxel_features
                voxel_count = features.shape[1]
                mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
                mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
                features *= mask
                features = self.pfn_layers[k][0](features)

                pillar_features.append(features.squeeze())

        # sparse conv net
        if self.projection == 'psa':
            pillar_features = batch_voxel_features
        elif self.projection == 'pfn':
            pillar_features = torch.cat(pillar_features, dim=1)
        voxel_coords = coords

        if self.use_sparse is True:

            input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
            )
            x = self.conv_input(input_sp_tensor)  # [1, 496, 432]
            x = self.conv1(x)
            x = self.conv2(x)  # [1, 248, 216]
            out = x.dense()


            N, C, D, H, W = out.shape
            spatial_features = out.view(N, C * D, H, W)
        else:
            batch_dict['pillar_features'] = pillar_features
            input_tensor = torch.zeros([batch_size * self.sparse_shape[1] * self.sparse_shape[2], pillar_features.shape[1]],
                                       device=pillar_features.device)
            ind = (voxel_coords[:,0] * self.sparse_shape[1] * self.sparse_shape[2] +
                   voxel_coords[:,2] * self.sparse_shape[2] +
                   voxel_coords[:,3]
                   ).unsqueeze(-1).repeat(1, 128)
            input_dense_tensor = torch.scatter(input_tensor, 0, ind.long(), pillar_features).view(
                batch_size, self.sparse_shape[1], self.sparse_shape[2], pillar_features.shape[1])
            input_dense_tensor = input_dense_tensor.permute(0, 3, 1, 2)
            batch_dict['input_dense_tensor'] = input_dense_tensor
            x = input_dense_tensor
            x = self.conv_input(x)

            x = self.conv1(x)
            spatial_features = self.conv2(x)

        if 'export' in batch_dict.keys():
            return spatial_features
        else:
            batch_dict['spatial_features'] = spatial_features
            batch_dict['voxel_size'] = self.voxel_size
            return batch_dict
