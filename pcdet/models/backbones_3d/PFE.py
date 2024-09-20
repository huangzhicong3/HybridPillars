import torch
import torch.nn as nn
from ...utils.common_utils import gather
from .pfe.ops.builder import grouper
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
import os
from easydict import EasyDict

class PFE(nn.Module):

    def __init__(self, model_cfg, num_class, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class

        self.SA_modules = nn.ModuleList()
        # channel_in = input_channels - 3
        channel_in = 1
        channel_out_list = [channel_in]

        self.num_points_each_layer = []

        sa_config = self.model_cfg.SA_CONFIG
        self.layer_types = sa_config.LAYER_TYPE
        self.ctr_idx_list = sa_config.CTR_INDEX
        self.layer_inputs = sa_config.LAYER_INPUT
        self.aggregation_mlps = sa_config.get('AGGREGATION_MLPS', None)
        self.confidence_mlps = sa_config.get('CONFIDENCE_MLPS', None)
        self.max_translate_range = sa_config.get('MAX_TRANSLATE_RANGE', None)


        for k in range(sa_config.NSAMPLE_LIST.__len__()):
            if isinstance(self.layer_inputs[k], list): ###
                channel_in = channel_out_list[self.layer_inputs[k][-1]]
            else:
                channel_in = channel_out_list[self.layer_inputs[k]]

            if self.layer_types[k] == 'SA_Layer':
                mlps = sa_config.MLPS[k].copy()
                channel_out = 0
                for idx in range(mlps.__len__()):
                    mlps[idx] = [channel_in] + mlps[idx]
                    channel_out += mlps[idx][-1]

                if self.aggregation_mlps and self.aggregation_mlps[k]:
                    aggregation_mlp = self.aggregation_mlps[k].copy()
                    if aggregation_mlp.__len__() == 0:
                        aggregation_mlp = None
                    else:
                        channel_out = aggregation_mlp[-1]
                else:
                    aggregation_mlp = None

                if self.confidence_mlps and self.confidence_mlps[k]:
                    confidence_mlp = self.confidence_mlps[k].copy()
                    if confidence_mlp.__len__() == 0:
                        confidence_mlp = None
                else:
                    confidence_mlp = None

                if sa_config.PILLAR_AGGREGATION[k]:
                    self.SA_modules.append(
                        pointnet2_modules.PointnetSAModuleMSG_WithSampling_pillar(
                            npoint_list=sa_config.NPOINT_LIST[k],
                            sample_range_list=sa_config.SAMPLE_RANGE_LIST[k],
                            sample_type_list=sa_config.SAMPLE_METHOD_LIST[k],
                            radii=sa_config.RADIUS_LIST[k],
                            nsamples=sa_config.NSAMPLE_LIST[k],
                            mlps=mlps,
                            use_xyz=True,
                            dilated_group=sa_config.DILATED_GROUP[k],
                            aggregation_mlp=aggregation_mlp,
                            confidence_mlp=confidence_mlp,
                            num_class=self.num_class
                        )
                    )
                else:
                    self.SA_modules.append(
                        pointnet2_modules.PointnetSAModuleMSG_WithSampling(
                            npoint_list=sa_config.NPOINT_LIST[k],
                            sample_range_list=sa_config.SAMPLE_RANGE_LIST[k],
                            sample_type_list=sa_config.SAMPLE_METHOD_LIST[k],
                            radii=sa_config.RADIUS_LIST[k],
                            nsamples=sa_config.NSAMPLE_LIST[k],
                            mlps=mlps,
                            use_xyz=True,
                            dilated_group=sa_config.DILATED_GROUP[k],
                            aggregation_mlp=aggregation_mlp,
                            confidence_mlp=confidence_mlp,
                            num_class=self.num_class
                        )
                    )

            elif self.layer_types[k] == 'Vote_Layer':
                self.SA_modules.append(pointnet2_modules.Vote_layer(mlp_list=sa_config.MLPS[k],
                                                                    pre_channel=channel_out_list[self.layer_inputs[k]],
                                                                    max_translate_range=self.max_translate_range
                                                                    )
                                       )

            channel_out_list.append(channel_out)

        self.num_point_features = channel_out


    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None  ###
        batch_dict['xyz'] = xyz
        batch_dict['features'] = features

        # xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        # for bs_idx in range(batch_size):
        #     xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
        #
        # assert xyz_batch_cnt.min() == xyz_batch_cnt.max()

        encoder_xyz, encoder_features, sa_ins_preds = [xyz], [features], []
        encoder_coords = [torch.cat([batch_idx.view(batch_size, -1, 1), xyz], dim=-1)]

        li_cls_pred = None
        ctr_offsets = None

        for i in range(len(self.SA_modules)):
            xyz_input = encoder_xyz[self.layer_inputs[i]]
            feature_input = encoder_features[self.layer_inputs[i]]

            if self.layer_types[i] == 'SA_Layer':
                ctr_xyz = encoder_xyz[self.ctr_idx_list[i]] if self.ctr_idx_list[i] != -1 else None
                li_xyz, li_features, li_cls_pred = self.SA_modules[i](xyz_input, feature_input, li_cls_pred, ctr_xyz=ctr_xyz)
            encoder_xyz.append(li_xyz)
            li_batch_idx = batch_idx.view(batch_size, -1)[:, :li_xyz.shape[1]]
            encoder_coords.append(torch.cat([li_batch_idx[..., None].float(),li_xyz.view(batch_size, -1, 3)],dim =-1))
            encoder_features.append(li_features)

            if li_cls_pred is not None:
                li_cls_batch_idx = batch_idx.view(batch_size, -1)[:, :li_cls_pred.shape[1]]
                sa_ins_preds.append(torch.cat([li_cls_batch_idx[..., None].float(),li_cls_pred.view(batch_size, -1, li_cls_pred.shape[-1])],dim =-1))
            else:
                sa_ins_preds.append([])


        batch_dict['encoder_xyz'] = encoder_xyz
        batch_dict['encoder_coords'] = encoder_coords
        batch_dict['sa_ins_preds'] = sa_ins_preds
        batch_dict['encoder_features'] = encoder_features
        
        return batch_dict
