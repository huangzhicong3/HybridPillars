import numpy as np
import torch.nn as nn
import torch
from .anchor_head_template import AnchorHeadTemplate
from ...utils import box_coder_utils, box_utils, loss_utils, common_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
import torch.nn.functional as F

class HybridPillars_AnchorHead(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder2 = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()


    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def assign_stack_targets_IASSD(self, points, gt_boxes, extend_gt_boxes=None, weighted_labels=False,
                                   ret_box_labels=False, ret_offset_labels=True,
                                   set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0,
                                   use_query_assign=False, central_radii=2.0, use_ex_gt_assign=False,
                                   fg_pc_ignore=False,
                                   binary_label=False):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        box_idxs_labels = points.new_zeros(points.shape[0]).long()
        gt_boxes_of_fg_points = []
        gt_box_of_points = gt_boxes.new_zeros((points.shape[0], 8))

        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)

            if use_query_assign:  ##
                centers = gt_boxes[k:k + 1, :, 0:3]
                query_idxs_of_pts = roiaware_pool3d_utils.points_in_ball_query_gpu(
                    points_single.unsqueeze(dim=0), centers.contiguous(), central_radii
                ).long().squeeze(dim=0)
                query_fg_flag = (query_idxs_of_pts >= 0)
                if fg_pc_ignore:
                    fg_flag = query_fg_flag ^ box_fg_flag
                    extend_box_idxs_of_pts[box_idxs_of_pts != -1] = -1
                    box_idxs_of_pts = extend_box_idxs_of_pts
                else:
                    fg_flag = query_fg_flag
                    box_idxs_of_pts = query_idxs_of_pts
            elif use_ex_gt_assign:  ##
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                extend_fg_flag = (extend_box_idxs_of_pts >= 0)

                extend_box_idxs_of_pts[box_fg_flag] = box_idxs_of_pts[
                    box_fg_flag]  # instance points should keep unchanged

                if fg_pc_ignore:
                    fg_flag = extend_fg_flag ^ box_fg_flag
                    extend_box_idxs_of_pts[box_idxs_of_pts != -1] = -1
                    box_idxs_of_pts = extend_box_idxs_of_pts
                else:
                    fg_flag = extend_fg_flag
                    box_idxs_of_pts = extend_box_idxs_of_pts

            elif set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1

            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag

            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 or binary_label else gt_box_of_fg_points[:,
                                                                                             -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single
            bg_flag = (point_cls_labels_single == 0)  # except ignore_id
            # box_bg_flag
            fg_flag = fg_flag ^ (fg_flag & bg_flag)
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]

            gt_boxes_of_fg_points.append(gt_box_of_fg_points)
            box_idxs_labels[bs_mask] = box_idxs_of_pts
            gt_box_of_points[bs_mask] = gt_boxes[k][box_idxs_of_pts]

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder2.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

        gt_boxes_of_fg_points = torch.cat(gt_boxes_of_fg_points, dim=0)
        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'gt_box_of_fg_points': gt_boxes_of_fg_points,
            'box_idxs_labels': box_idxs_labels,
            'gt_box_of_points': gt_box_of_points,
        }
        return targets_dict

    def forward(self, data_dict):
        if 'export' in data_dict.keys():
            pass
        elif 'ctr_offsets' in data_dict.keys():
            ret_dict = {
                        'ctr_offsets': data_dict['ctr_offsets'],
                        'centers': data_dict['centers'],
                        'centers_origin': data_dict['centers_origin'],
                        'sa_ins_preds': data_dict['sa_ins_preds']
                        }
        else:
            ret_dict = {
                'sa_ins_preds': data_dict['sa_ins_preds']
            }

        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        if 'sa_ins_preds' in data_dict.keys():
            self.forward_ret_dict['sa_ins_preds'] = data_dict['sa_ins_preds']

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes'],
                input_dict=data_dict
            )
            self.forward_ret_dict.update(targets_dict)
            self.forward_ret_dict.update(ret_dict)

        if 'export' in data_dict.keys():
            return cls_preds, box_preds, dir_cls_preds
        elif not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

    def assign_targets(self, gt_boxes, input_dict=None):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        target_cfg = self.model_cfg.TARGET_CONFIG
        gt_boxes = input_dict['gt_boxes']
        if gt_boxes.shape[-1] == 10:  # nscence
            gt_boxes = torch.cat((gt_boxes[..., 0:7], gt_boxes[..., -1:]), dim=-1)

        targets_dict_center = {}
        # assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        # assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)
        batch_size = input_dict['batch_size']
        if target_cfg.get('EXTRA_WIDTH', False):  # multi class extension
            extend_gt = box_utils.enlarge_box3d_for_class(
                gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=target_cfg.EXTRA_WIDTH
            ).view(batch_size, -1, gt_boxes.shape[-1])
        else:
            extend_gt = gt_boxes

        extend_gt_boxes = box_utils.enlarge_box3d(
            extend_gt.view(-1, extend_gt.shape[-1]), extra_width=target_cfg.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        if 'centers' in input_dict.keys():
            center_targets_dict = self.assign_stack_targets_IASSD(
                points=input_dict['centers'].detach(),
                gt_boxes=extend_gt, extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True, use_ball_constraint=False,
                ret_box_labels=True
            )
            targets_dict_center['center_gt_box_of_fg_points'] = center_targets_dict['gt_box_of_fg_points']
            targets_dict_center['center_cls_labels'] = center_targets_dict['point_cls_labels']
            targets_dict_center['center_box_labels'] = center_targets_dict['point_box_labels']  # only center assign
            targets_dict_center['center_gt_box_of_points'] = center_targets_dict['gt_box_of_points']
            extra_method = target_cfg.get('ASSIGN_METHOD', None)
            extend_gt_boxes = box_utils.enlarge_box3d(
                gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=extra_method.EXTRA_WIDTH
            ).view(batch_size, -1, gt_boxes.shape[-1])

            if extra_method.get('ASSIGN_TYPE', 'centers') == 'centers_origin':
                points = input_dict['centers_origin'].detach()
            else:
                points = input_dict['centers'].detach()  # default setting

            targets_dict_iassd = self.assign_stack_targets_IASSD(
                points=points, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True, use_ball_constraint=False,
                ret_box_labels=True,
                use_ex_gt_assign=True, fg_pc_ignore=extra_method.FG_PC_IGNORE,
            )
            targets_dict['center_origin_gt_box_of_fg_points'] = targets_dict_iassd['gt_box_of_fg_points']
            targets_dict['center_origin_cls_labels'] = targets_dict_iassd['point_cls_labels']
            targets_dict['center_origin_box_idxs_of_pts'] = targets_dict_iassd['box_idxs_labels']
            targets_dict['gt_box_of_center_origin'] = targets_dict_iassd['gt_box_of_points']
            targets_dict.update(targets_dict_center)

        if 'sa_ins_preds' in input_dict.keys():
            target_cfg = self.model_cfg.TARGET_CONFIG
            if target_cfg.get('INS_AWARE_ASSIGN', False):
                sa_ins_labels, sa_gt_box_of_fg_points, sa_xyz_coords, sa_gt_box_of_points, sa_box_idxs_labels = [], [], [], [], []
                sa_ins_preds = input_dict['sa_ins_preds']
                batch_size = input_dict['batch_size']
                # for i in range(1, len(sa_ins_preds)):  # valid when i = 1,2 for IA-SSD
                if len(sa_ins_preds) == 1:
                    for_num = 2
                else:
                    for_num = len(sa_ins_preds)
                for i in range(1, for_num):  # valid when i = 1,2 for IA-SSD
                    # if sa_ins_preds[i].__len__() == 0:
                    #     continue
                    sa_xyz = input_dict['encoder_coords'][i]
                    if i == 1:
                        extend_gt_boxes = box_utils.enlarge_box3d(
                            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]  # [0.2, 0.2, 0.2]
                        ).view(batch_size, -1, gt_boxes.shape[-1])
                        sa_targets_dict = self.assign_stack_targets_IASSD(
                            points=sa_xyz.view(-1, sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes,
                            extend_gt_boxes=extend_gt_boxes,
                            set_ignore_flag=True, use_ex_gt_assign=False
                        )
                    if i >= 2:
                        # if False:
                        extend_gt_boxes = box_utils.enlarge_box3d(
                            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]
                        ).view(batch_size, -1, gt_boxes.shape[-1])
                        sa_targets_dict = self.assign_stack_targets_IASSD(
                            points=sa_xyz.view(-1, sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes,
                            extend_gt_boxes=extend_gt_boxes,
                            set_ignore_flag=False, use_ex_gt_assign=True
                        )
                    # else:
                    #     extend_gt_boxes = box_utils.enlarge_box3d(
                    #         gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.5, 0.5, 0.5]
                    #     ).view(batch_size, -1, gt_boxes.shape[-1])
                    #     sa_targets_dict = self.assign_stack_targets_IASSD(
                    #         points=sa_xyz.view(-1,sa_xyz.shape[-1]).detach(), gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
                    #         set_ignore_flag=False, use_ex_gt_assign= True
                    #     )
                    sa_xyz_coords.append(sa_xyz)
                    sa_ins_labels.append(sa_targets_dict['point_cls_labels'])
                    sa_gt_box_of_fg_points.append(sa_targets_dict['gt_box_of_fg_points'])
                    sa_gt_box_of_points.append(sa_targets_dict['gt_box_of_points'])
                    sa_box_idxs_labels.append(sa_targets_dict['box_idxs_labels'])

                targets_dict['sa_ins_labels'] = sa_ins_labels
                targets_dict['sa_gt_box_of_fg_points'] = sa_gt_box_of_fg_points
                targets_dict['sa_xyz_coords'] = sa_xyz_coords
                targets_dict['sa_gt_box_of_points'] = sa_gt_box_of_points
                targets_dict['sa_box_idxs_labels'] = sa_box_idxs_labels

        return targets_dict

    def generate_sa_center_ness_mask(self):
        sa_pos_mask = self.forward_ret_dict['sa_ins_labels']
        sa_gt_boxes = self.forward_ret_dict['sa_gt_box_of_fg_points']
        sa_xyz_coords = self.forward_ret_dict['sa_xyz_coords']
        sa_centerness_mask = []
        for i in range(len(sa_pos_mask)):
            pos_mask = sa_pos_mask[i] > 0
            gt_boxes = sa_gt_boxes[i]
            xyz_coords = sa_xyz_coords[i].view(-1,sa_xyz_coords[i].shape[-1])[:,1:]
            xyz_coords = xyz_coords[pos_mask].clone().detach()
            offset_xyz = xyz_coords[:, 0:3] - gt_boxes[:, 0:3]
            offset_xyz_canical = common_utils.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -gt_boxes[:, 6]).squeeze(dim=1)

            template = gt_boxes.new_tensor(([1, 1, 1], [-1, -1, -1])) / 2
            margin = gt_boxes[:, None, 3:6].repeat(1, 2, 1) * template[None, :, :]
            distance = margin - offset_xyz_canical[:, None, :].repeat(1, 2, 1)
            distance[:, 1, :] = -1 * distance[:, 1, :]
            distance_min = torch.where(distance[:, 0, :] < distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])
            distance_max = torch.where(distance[:, 0, :] > distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])

            centerness = distance_min / distance_max
            centerness = centerness[:, 0] * centerness[:, 1] * centerness[:, 2]
            centerness = torch.clamp(centerness, min=1e-6)
            centerness = torch.pow(centerness, 1/3)

            centerness_mask = pos_mask.new_zeros(pos_mask.shape).float()
            centerness_mask[pos_mask] = centerness

            sa_centerness_mask.append(centerness_mask)
        return sa_centerness_mask

    def get_sa_ins_layer_loss(self, tb_dict=None):
        sa_ins_labels = self.forward_ret_dict['sa_ins_labels']
        sa_ins_preds = self.forward_ret_dict['sa_ins_preds']
        if self.model_cfg.LOSS_CONFIG.USE_CENTERNESS_LOSS:
            sa_centerness_mask = self.generate_sa_center_ness_mask()
        sa_ins_loss, ignore = 0, 0
        for i in range(len(sa_ins_preds)): # valid when i =1, 2
            if len(sa_ins_preds[i]) != 0:
                try:
                    point_cls_preds = sa_ins_preds[i][...,1:].view(-1, self.num_class)
                except:
                    point_cls_preds = sa_ins_preds[i][...,1:].view(-1, 1)
            else:
                ignore += 1
                continue
            point_cls_labels = sa_ins_labels[i].view(-1)
            positives = (point_cls_labels > 0)
            negative_cls_weights = (point_cls_labels == 0) * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            pos_normalizer = positives.sum(dim=0).float()
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)

            one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
            one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
            one_hot_targets = one_hot_targets[..., 1:]

            if self.model_cfg.LOSS_CONFIG.USE_CENTERNESS_LOSS:
                if ('ctr' in self.model_cfg.LOSS_CONFIG.SAMPLE_METHOD_LIST[i+1][0]):
                    centerness_mask = sa_centerness_mask[i]
                    one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1).repeat(1, one_hot_targets.shape[1])

            point_loss_ins = self.ins_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights).mean(dim=-1).sum()
            loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
            point_loss_ins = point_loss_ins * loss_weights_dict.get('ins_aware_weight',[1]*len(sa_ins_labels))[i]

            sa_ins_loss += point_loss_ins
            if tb_dict is None:
                tb_dict = {}
            tb_dict.update({
                'sa%s_loss_ins' % str(i): point_loss_ins.item(),
                'sa%s_pos_num' % str(i): pos_normalizer.item()
            })
        # print(ignore)
        sa_ins_loss = sa_ins_loss / (len(sa_ins_preds) - ignore)
        tb_dict.update({
                'sa_loss_ins': sa_ins_loss.item(),
            })
        return sa_ins_loss, tb_dict

    def get_vote_loss_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        center_box_labels = self.forward_ret_dict['center_gt_box_of_fg_points'][:, 0:3]
        centers_origin = self.forward_ret_dict['centers_origin']
        ctr_offsets = self.forward_ret_dict['ctr_offsets']
        centers_pred = centers_origin + ctr_offsets
        centers_pred = centers_pred[pos_mask][:, 1:4]

        vote_loss = F.smooth_l1_loss(centers_pred, center_box_labels, reduction='mean')
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'vote_loss': vote_loss.item()})
        return vote_loss, tb_dict

    def get_contextual_vote_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_origin_cls_labels'] > 0
        center_origin_loss_box = []
        for i in self.forward_ret_dict['center_origin_cls_labels'].unique():
            if i <= 0: continue
            simple_pos_mask = self.forward_ret_dict['center_origin_cls_labels'] == i
            center_box_labels = self.forward_ret_dict['center_origin_gt_box_of_fg_points'][:, 0:3][(pos_mask & simple_pos_mask)[pos_mask==1]]
            centers_origin = self.forward_ret_dict['centers_origin']
            ctr_offsets = self.forward_ret_dict['ctr_offsets']
            centers_pred = centers_origin + ctr_offsets
            centers_pred = centers_pred[simple_pos_mask][:, 1:4]
            simple_center_origin_loss_box = F.smooth_l1_loss(centers_pred, center_box_labels)
            center_origin_loss_box.append(simple_center_origin_loss_box.unsqueeze(-1))
        center_origin_loss_box = torch.cat(center_origin_loss_box, dim=-1).mean()
        center_origin_loss_box = center_origin_loss_box * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('vote_weight')
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_origin_loss_reg': center_origin_loss_box.item()})
        return center_origin_loss_box, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)

        # vote loss
        if 'center_origin_cls_labels' in self.forward_ret_dict.keys():
            if self.model_cfg.TARGET_CONFIG.get('ASSIGN_METHOD') is not None and \
                    self.model_cfg.TARGET_CONFIG.ASSIGN_METHOD.get('ASSIGN_TYPE') == 'centers_origin':
                if self.model_cfg.LOSS_CONFIG.get('LOSS_VOTE_TYPE', 'none') == 'ver1':
                    center_loss_reg, tb_dict_3 = self.get_contextual_vote_loss_ver1()
                elif self.model_cfg.LOSS_CONFIG.get('LOSS_VOTE_TYPE', 'none') == 'ver2':
                    center_loss_reg, tb_dict_3 = self.get_contextual_vote_loss_ver2()
                else:  # 'none'
                    center_loss_reg, tb_dict_3 = self.get_contextual_vote_loss()
            else:
                center_loss_reg, tb_dict_3 = self.get_vote_loss_loss()  # center assign
            tb_dict.update(tb_dict_3)
        else:
            center_loss_reg = 0

        # semantic loss in SA layers
        if self.model_cfg.LOSS_CONFIG.get('LOSS_INS', None) is not None:
            assert ('sa_ins_preds' in self.forward_ret_dict) and ('sa_ins_labels' in self.forward_ret_dict)
            sa_loss_cls, tb_dict_0 = self.get_sa_ins_layer_loss()
            tb_dict.update(tb_dict_0)
        else:
            sa_loss_cls = 0

        rpn_loss = cls_loss + box_loss + sa_loss_cls + center_loss_reg
        # rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict