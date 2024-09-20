from .detector3d_template import Detector3DTemplate
import time
import torch
import numpy as np
class HybridPillars(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        if 'ROI_HEAD' in model_cfg.keys():
            self.rcnn_only = model_cfg.ROI_HEAD.RCNN_ONLY
        else:
            self.rcnn_only = None

    def forward(self, batch_dict):
        torch.cuda.synchronize()
        batch_dict['dataset_cfg'] = self.dataset.dataset_cfg
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            if self.rcnn_only is None:
                loss, tb_dict, disp_dict = self.get_training_loss_stage_one()
                ret_dict = {
                    'loss': loss
                }
            elif self.rcnn_only is True:
                loss, tb_dict, disp_dict = self.get_training_loss_rcnn_only()
                ret_dict = {
                    'loss': loss
                }
            else:
                loss, tb_dict, disp_dict = self.get_training_loss()
                ret_dict = {
                    'loss': loss
                }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss_rcnn_only(self):
        disp_dict = {}

        # loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss()
        tb_dict = {
            # 'loss_rpn': loss_rpn.item(),
            'loss_rcnn': loss_rcnn.item(),
            **tb_dict
        }

        loss = loss_rcnn

        return loss, tb_dict, disp_dict

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_rcnn': loss_rcnn.item(),
            **tb_dict
        }

        loss = loss_rpn + loss_rcnn

        return loss, tb_dict, disp_dict

    def get_training_loss_stage_one(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        # loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            # 'loss_rcnn': loss_rcnn.item(),
            **tb_dict
        }

        loss = loss_rpn

        return loss, tb_dict, disp_dict