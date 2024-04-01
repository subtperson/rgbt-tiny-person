# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_backbone, build_head
from .single_stage import SingleStageDetector
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import bbox2result, multi_apply
import cv2


@DETECTORS.register_module()
class QFDet(SingleStageDetector):
    """Implementation of `QFDet`."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_prehead,
                 base_fusion='cat',
                 quality_attention=True,
                 poolupsample=None,
                 reweight=False,
                 esg=False,
                 qce=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(QFDet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
        self.backbone_t = build_backbone(backbone)
        self.base_fusion = base_fusion
        self.fuse = Fusion_strategy(neck['out_channels'])
        bbox_prehead.update(train_cfg=train_cfg)
        bbox_prehead.update(test_cfg=test_cfg)
        self.bbox_prehead = build_head(bbox_prehead)
        self.iter = 0
        if poolupsample is not None:
            self.poolupsample = PoolingUpsample(neck['out_channels'])
        else:
            self.poolupsample = None
        self.quality_attention = quality_attention

        self.reweight = reweight
        self.esg = esg
        self.qce = qce
        # self.stn = STN()
        # self.stnforno = STNforNodetach()
        # self.stnfor = STNfor()
        self.esg0 = BasicIRNet(in_plane=256, upscale=4)
        self.esg1 = BasicIRNet(in_plane=256, upscale=8)
        self.esg2 = BasicIRNet(in_plane=256, upscale=16)
        self.esg3 = BasicIRNet(in_plane=256, upscale=32)
        self.esg4 = BasicIRNet(in_plane=256, upscale=64)

        self.conv1x1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        self.bceloss = nn.BCELoss()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        # self.iter = self.iter + 1
        v_img, t_img = img
        v_feats = self.backbone(v_img)
        t_feats = self.backbone_t(t_img)
        if self.with_neck:
            v_feats = self.neck(v_feats)
            t_feats = self.neck(t_feats)

        return (v_feats, t_feats)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        x_vs, x_ts = x
        num_level = len(x_vs)
        quality_preds_t, quality_preds_v = self.bbox_prehead.forward_test(x)
        fused_x = []
        for i in range(num_level):
            quality_pred_t, quality_pred_v = my_norm(quality_preds_t[i], quality_preds_v[i], type='minmax')
            
            x_t = (1 + quality_pred_t) * x_ts[i]
            x_v = (1 + quality_pred_v) * x_vs[i]

            if self.poolupsample is not None and i < num_level-1:
                x_v = self.poolupsample(x_v)
            
            fused_x_ = self.fuse(x_t, x_v, 'cat')
            fused_x.append(fused_x_)
        outs = self.bbox_head(fused_x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        
        

        # super(SingleStageDetector, self).forward_train(img, img_metas)
        losses = dict()
        maskmap = self.build_target_esg(gt_bboxes, img_metas)

        x = self.extract_feat(img)
        quality_preds_t, quality_preds_v = None, None

        if self.qce:
            cls_scores_t, bbox_preds_t, centernesses_t, quality_preds_t, \
            cls_scores_v, bbox_preds_v, centernesses_v, quality_preds_v = self.bbox_prehead(x)
        
            pre_loss = self.bbox_prehead.loss(
                cls_scores_t, 
                bbox_preds_t, 
                centernesses_t, 
                quality_preds_t,
                cls_scores_v, 
                bbox_preds_v, 
                centernesses_v, 
                quality_preds_v,
                gt_bboxes,
                gt_labels,
                img_metas,
                gt_bboxes_ignore=None)
            losses.update(pre_loss)
        fused_x = self.qce_fusion(x, quality_preds_t, quality_preds_t)

        if self.esg:
            maskmap = maskmap.unsqueeze(1).float()
            map0 = self.esg0(fused_x[0])
            lossesg0 = self.bceloss(map0, maskmap)
            weight0 = F.max_pool2d(map0, kernel_size=4)
            fused_x[0] = fused_x[0] * weight0
            

            map1 = self.esg1(fused_x[1])
            lossesg1 = self.bceloss(map1, maskmap)
            weight1 = F.max_pool2d(map1, kernel_size=8)
            fused_x[1] = fused_x[1] * weight1

            map2 = self.esg2(fused_x[2])
            lossesg2 = self.bceloss(map2, maskmap)
            weight2 = F.max_pool2d(map2, kernel_size=16)
            fused_x[2] = fused_x[2] * weight2 

            map3 = self.esg3(fused_x[3])
            lossesg3 = self.bceloss(map3, maskmap)
            weight3 = F.max_pool2d(map3, kernel_size=32)
            fused_x[3] = fused_x[3] * weight3

            map4 = self.esg4(fused_x[4])
            lossesg4 = self.bceloss(map4, maskmap)
            weight4 = F.max_pool2d(map4, kernel_size=64)
            fused_x[4] = fused_x[4] * weight4

            # import time
            # if int(time.time()) % 10 == 0:
            #     cv2.imwrite('/media/vision/lzy/mm_person/mmdet-rgbtdroneperson/vis/'+ '2-3.png', maskmap[0, 0,:,:].cpu().numpy()*255)
            #     cv2.imwrite('/media/vision/lzy/mm_person/mmdet-rgbtdroneperson/vis/'+ '2-4.png', map0[0, 0, :, :].detach().cpu().numpy()*255)

            wei = 0.1
            lossesg = (lossesg0 + lossesg1 + lossesg2 + lossesg3 + lossesg4) * wei

            losses.update({'esg': lossesg})

        loss = self.bbox_head.forward_train(fused_x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        losses.update(loss)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
65
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        self.iter += 1
        x = self.extract_feat(img)
        quality_preds_t, quality_preds_v = None, None
        if self.qce:
            quality_preds_t, quality_preds_v = self.bbox_prehead.forward_test(x)
        
        fused_x = self.qce_fusion(x, quality_preds_t, quality_preds_v)

        if self.esg:
            # maskmap = maskmap.unsqueeze(1)
            map0 = self.esg0(fused_x[0])
            weight0 = F.max_pool2d(map0, kernel_size=4)
            fused_x[0] = fused_x[0] * weight0
            

            map1 = self.esg1(fused_x[1])
            weight1 = F.max_pool2d(map1, kernel_size=8)
            fused_x[1] = fused_x[1] * weight1

            map2 = self.esg2(fused_x[2])
            weight2 = F.max_pool2d(map2, kernel_size=16)
            fused_x[2] = fused_x[2] * weight2 

            map3 = self.esg3(fused_x[3])
            weight3 = F.max_pool2d(map3, kernel_size=32)
            fused_x[3] = fused_x[3] * weight3

            map4 = self.esg4(fused_x[4])
            weight4 = F.max_pool2d(map4, kernel_size=64)
            fused_x[4] = fused_x[4] * weight4

        results_list = self.bbox_head.simple_test(
            fused_x, img_metas, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
    
    def qce_fusion(self, x, quality_t, quality_v):
        x_vs, x_ts = x

        num_level = len(x_vs)

        fused_x = []
        for i in range(num_level):
            x_t = x_ts[i]
            x_v = x_vs[i]
            if self.quality_attention and self.qce and quality_t is not None and quality_v is not None:
                quality_pred_t = torch.max(quality_t[i], dim=1, keepdim=True)[0]
                quality_pred_v = torch.max(quality_v[i], dim=1, keepdim=True)[0]

                quality_pred_t, quality_pred_v = my_norm(quality_pred_t, quality_pred_v, type='minmax')

                # if self.lzystn and i < 3:
                #     quality_pred_v, x_v = self.stn(quality_pred_v, quality_pred_t, x_v)


                x_t = (1 + quality_pred_t) * x_t
                x_v = (1 + quality_pred_v) * x_v            
            
            if self.poolupsample is not None and i < num_level-1:
                x_v = self.poolupsample(x_v)
                # x_t = self.poolupsample(x_t)

            fused_x_ = self.fuse(x_t, x_v, self.base_fusion)

            fused_x.append(fused_x_)

        return fused_x

    def build_target_esg(self, gt_bboxes, img_metas):
        # build object map
        list_object_maps = []
        for i, gt_bbox in enumerate(gt_bboxes):
            object_map = torch.zeros(img_metas[0]["batch_input_shape"], device=gt_bboxes[0].device)
            for index in range(gt_bbox.shape[0]):
                gt = gt_bbox[index]
                # 扩大gt
                # gt[0] = gt[0] - 1 if gt[0] - 1 > 0 else 0
                # gt[1] = gt[1] - 1 if gt[1] - 1 > 0 else 0
                # gt[2] = gt[2] + 1 if gt[2] + 1 < object_map.shape[1] else object_map.shape[1]
                # gt[3] = gt[3] + 1 if gt[3] + 1 < object_map.shape[0] else object_map.shape[0]
                # 宽和高都小于32为条件
                w = int(gt[2])-int(gt[0])
                h = int(gt[3])-int(gt[1])

                # if (int(gt[2])-int(gt[0])) <= 32 and (int(gt[3]) - int(gt[1])) <= 32:
                if w*h <= 32*32:
                    object_map[int(gt[1]):(int(gt[3])+1), int(gt[0]):(int(gt[2])+1)] = 1

            list_object_maps.append(object_map[None])

        object_maps = torch.cat(list_object_maps, dim=0)
        return object_maps.long()


def save_feature_to_img(features, name, timestamp=None, method='cv2', channel=None, output_dir=None, maxmin=None):
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import os

    if output_dir is None:
        output_dir = '/media/coding/zyl/mm_person/mmdet-rgbtdroneperson/vis'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not isinstance(timestamp, str):
        timestamp = str(timestamp)

    # for i in range(len(features)):
    upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    if isinstance(features, list) or isinstance(features, tuple):
        for i in range(3):
            features_ = features[i]
            for j in range(features_.shape[0]):
                
                features_ = upsample(features_)

                feature = features_[j, :, :, :]
                if channel is None:
                    feature = torch.sum(feature, 0)
                else:
                    feature = feature[channel, :, :]
                feature = feature.detach().cpu().numpy() # 转为numpy

                dist_dir = os.path.join(output_dir, timestamp)
                if not os.path.exists(dist_dir):
                    os.mkdir(dist_dir)

                if method == 'cv2':
                    if maxmin is not None:
                        img = (feature - maxmin[1])/(maxmin[0] - maxmin[1] + 1e-5) * 255
                    else:
                        img = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) * 255 # 注意要防止分母为0！ 
                    img = img.astype(np.uint8)
                    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    plt.imshow(feature)
                    plt.axis('off')
                    cv2.imwrite(os.path.join(dist_dir, name + str(i) + '.jpg'), img)

                elif method == 'matshow':
                    plt.matshow(feature, interpolation='nearest')
                    plt.colorbar()
                    plt.axis('off')

                    plt.savefig(os.path.join(dist_dir, name + str(i) + '.png'))
                    plt.close()
                else:
                    NotImplementedError()
    
    else:
        for j in range(features.shape[0]):
            
            # features = upsample(features)

            feature = features[j, :, :, :]
            if channel is None:
                feature = torch.sum(feature, 0)
            else:
                feature = feature[channel, :, :]
            feature = feature.detach().cpu().numpy() # 转为numpy

            dist_dir = os.path.join(output_dir, timestamp)
            if not os.path.exists(dist_dir):
                os.mkdir(dist_dir)

            if method == 'cv2':
                if maxmin is not None:
                    img = (feature - maxmin[1])/(maxmin[0] - maxmin[1] + 1e-5) * 255
                else:
                    img = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) * 255 # 注意要防止分母为0！ 
                img = img.astype(np.uint8)
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                plt.imshow(feature)
                plt.axis('off')
                cv2.imwrite(os.path.join(dist_dir, name + '.jpg'), img)

            elif method == 'matshow':
                plt.matshow(feature, interpolation='nearest')
                plt.colorbar()
                plt.axis('off')

                plt.savefig(os.path.join(dist_dir, name + '.png'))
                plt.close()
            else:
                NotImplementedError()


def feature_map_norm(x):
    # input (B, C, H, W)
    bs, c, h, w = x.shape
    x = x.view(bs, -1, h*w)
    x_mean = torch.mean(x, dim=2, keepdim=True)
    x_std = torch.std(x, dim=2, keepdim=True)
    x = (x - x_mean) / x_std
    x = x.view(bs, c, h, w)
    return x

def my_norm(x1, x2, type='standard'):
    assert type in ['standard', 'minmax']
    bs, _ , H, W = x1.size()
    _, _, h, w = x2.size()
    x1 = x1.view(bs, -1, H*W)
    x2 = x2.view(bs, -1, h*w)
    concat = torch.cat((x1, x2), dim=2)
    if type == 'standard':
        x_mean = torch.mean(concat, dim=2, keepdim=True)
        x_std = torch.std(concat, dim=2, keepdim=True)
        x1 = (x1 - x_mean) / x_std
        x2 = (x2 - x_mean) / x_std
    elif type == 'minmax':
        x_min = torch.min(concat, dim=2, keepdim=True)[0]
        x_max = torch.max(concat, dim=2, keepdim=True)[0]
        x1 = (x1 - x_min) / x_max
        x2 = (x2 - x_min) / x_max
    x1 = x1.view(bs, -1, H, W)
    x2 = x2.view(bs, -1, h, w)
    return [x1, x2]


class Fusion_ADD(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = en_ir + en_vi
        return temp

class Fusion_AVG(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = (en_ir + en_vi) / 2
        return temp

class Fusion_MAX(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = torch.max(en_ir, en_vi)
        return temp

class Fusion_CAT(torch.nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(2*in_channels, in_channels, 1)
    
    def forward(self, en_ir, en_vi):
        temp = torch.cat((en_ir, en_vi), 1)
        temp = self.conv1x1(temp)
        return temp

# Fusion strategy, two type
class Fusion_strategy(nn.Module):
    def __init__(self, in_channels):
        super(Fusion_strategy, self).__init__()
        self.fusion_add = Fusion_ADD()
        self.fusion_avg = Fusion_AVG()
        self.fusion_max = Fusion_MAX()
        self.fusion_cat = Fusion_CAT(in_channels=in_channels)

    def forward(self, v_feat, t_feat, fs_type):
        self.fs_type = fs_type
        if self.fs_type == 'add':
            fusion_operation = self.fusion_add
        elif self.fs_type == 'avg':
            fusion_operation = self.fusion_avg
        elif self.fs_type == 'max':
            fusion_operation = self.fusion_max
        elif self.fs_type == 'cat':
            fusion_operation = self.fusion_cat
        if isinstance(v_feat, tuple) or isinstance(v_feat, list):
            fused_feat = []
            for i in range(len(v_feat)):
                fused_feat.append(fusion_operation(v_feat[i], t_feat[i]))
        else:
            fused_feat = fusion_operation(v_feat, t_feat)
        
        return fused_feat



class PoolingUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.maxpooling = nn.MaxPool2d(2, 2, dilation=1)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1)
    
    def forward(self, x):
        # print("up_sample")
        x_ = self.maxpooling(x)
        # x_ = self.upsample(x_)
        x_ = F.interpolate(x_, mode='bilinear', size=x.shape[-2:], align_corners=True)
        # import pdb; pdb.set_trace()
        x = self.conv1x1(torch.cat((x, x_), 1))
        return x
    
'''
#RGB AND IR ALIGN Spatial Transformer Network
# class STN(nn.Module):
#     """
#     Spatial Transformer Network module by hmz
#     """
#     def __init__(self, use_dropout=False):
#         super(STN, self).__init__()
#         self._ksize = 3
#         self.dropout = use_dropout

#         # localization net 
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(2),
#             nn.ReLU()
#             )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(2),
#             nn.ReLU()
#             )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(2),
#             nn.ReLU()
#             )
#         # self.conv4 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)

#         self.fc1 = nn.Linear(2*4*4, 16)
#         self.fc2 = nn.Linear(16, 6)
        
#         # Initialize the weights/bias with identity transformation
#         self.fc2.weight.data.zero_()
#         self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

#     def forward(self, qv, qt, v):
    
#         one_feat = qv
#         another_feat = qt

#         feat = torch.cat((one_feat, another_feat), 1)
        
        
#         # x = F.relu(self.conv1(feat.detach()))
#         # # x = F.max_pool2d(x, 2)
#         # x = F.relu(self.conv2(x))
#         # x = F.max_pool2d(x, 2)
#         # x = F.relu(self.conv3(x))
#         # x = F.max_pool2d(x, 2)
#         # x = F.relu(self.conv4(x))
#         x = self.conv1(feat.detach())
#         # print(x.size())
#         x = self.conv2(x)
#         # print(x.size())
#         x = self.conv3(x)
#         # print(x.size())
#         x = F.adaptive_avg_pool2d(x, 4)
#         x = x.view(-1, 2*4*4)

#         if self.dropout:
#             x = F.dropout(self.fc1(x), p=0.5)
#             x = F.dropout(self.fc2(x), p=0.5)
#         else:
#             x = self.fc1(x)
#             x = self.fc2(x) # params [Nx6]
        
#         x = x.view(-1, 2,3) # change it to the 2x3 matrix 
#         # print(x.size())
#         affine_grid_points = F.affine_grid(x, one_feat.size())
#         assert(affine_grid_points.size(0) == one_feat.size(0)), "The batch sizes of the input images must be same as the generated grid."
#         qv = F.grid_sample(one_feat, affine_grid_points)
#         v = F.grid_sample(v, affine_grid_points)
#         # print(qv.size())
#         # print(v.size())

#         return qv, v




# class STNforNodetach(nn.Module):
#     """
#     Spatial Transformer Network module by hmz
#     """
#     def __init__(self, use_dropout=False):
#         super(STNforNodetach, self).__init__()
#         self._ksize = 3
#         self.dropout = use_dropout

#         # localization net 
#         self.conv1 = nn.Conv2d(512, 256, kernel_size=self._ksize, stride=1, padding=1, bias=False)
#         self.conv2 = nn.Conv2d(256, 64, kernel_size=self._ksize, stride=1, padding=1, bias=False)
#         self.conv3 = nn.Conv2d(64, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
#         self.conv4 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
#         # self.bn1 = nn.BatchNorm2d(32)
#         # self.bn2 = nn.BatchNorm2d(32)
#         # self.bn3 = nn.BatchNorm2d(32)
#         # self.bn4 = nn.BatchNorm2d(32)

#         self.fc1 = nn.Linear(32*5*5, 256)
#         self.fc2 = nn.Linear(256, 6)
        
#         # Initialize the weights/bias with identity transformation
#         self.fc2.weight.data.zero_()
#         self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

#         # feature extraction net
#         # in: batch_size x 512 x h x w, out: batch_size x 256 x h x w
#         self.conv_a = nn.Sequential(
#             nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             )
        

#         # liner layer
#         self.layer = nn.Sequential(
#             nn.Linear(256*5*5, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU()
#         )
        
#         self.mlp = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU()
#         )
        

#     def forward(self, x):
#         rois= [None]*5
#         pre_cat = [None]*5
#         loss = 0
#         xx = x
#         prev = None
#         pret = None
#         # rois = xx[0]
#         for i in range(5):
#             if i<2:
#                 v = xx[0][i]
#                 t = xx[1][i]
#                 # v = self.conv_a(v)
#                 # t = self.conv_a(t)
#                 # v = F.adaptive_max_pool2d(v, 5)
#                 # t = F.adaptive_max_pool2d(v, 5)
#                 # v = self.layer(v.view(-1, 256*5*5))
#                 # vmlp = self.mlp(v)
#                 # v = v + vmlp
#                 # t = self.layer(t.view(-1, 256*5*5))
#                 # # print(v)
#                 # # print(t)
#                 # lossthis = F.cosine_similarity(v, t)
#                 # loss = loss + lossthis

#                 one_feat = xx[0][i]
#                 another_feat = xx[1][i]

#                 feat = torch.cat((one_feat, another_feat), 1)

#                 # v = F.adaptive_avg_pool2d(one_feat, 1)
#                 # v = self.layer(v.view(-1, 256))
#                 # t = F.adaptive_avg_pool2d(t, 1)
#                 # t = self.layer(t.view(-1, 256))
#                 # t:1*2 v:1*2 compute loss between them
#                 # loss = nn.BCELoss()
#                 # pre_cat[i] = loss(v, t)*0.1
#                 # pre_cat[i] = torch.cat((v, t), 0)
#                 # prev.append(v)
#                 # pret.append(t)
#                 # prev = torch.cat((prev, v), 0) if prev is not None else v
#                 # pret = torch.cat((pret, t), 0) if pret is not None else t

#                 x = F.relu(self.conv1(feat.detach()))
#                 # x = F.relu(self.conv1(feat))
#                 x = F.max_pool2d(x, 2)
#                 x = F.relu(self.conv2(x))
#                 x = F.max_pool2d(x, 2)
#                 x = F.relu(self.conv3(x))
#                 # x = F.max_pool2d(x, 2)
#                 x = F.relu(self.conv4(x))
#                 x = F.adaptive_avg_pool2d(x, 5)
                
#                 x = x.view(-1, 32*5*5)
#                 if self.dropout:
#                     x = F.dropout(self.fc1(x), p=0.5)
#                     x = F.dropout(self.fc2(x), p=0.5)
#                 else:
#                     x = self.fc1(x)
#                     x = self.fc2(x) # params [Nx6]
                
#                 x = x.view(-1, 2,3) # change it to the 2x3 matrix 
#                 # print(x.size())
#                 affine_grid_points = F.affine_grid(x, one_feat.size())
#                 assert(affine_grid_points.size(0) == one_feat.size(0)), "The batch sizes of the input images must be same as the generated grid."
#                 rois[i] = F.grid_sample(one_feat, affine_grid_points, align_corners=True)
#             else:
#                 rois[i] = xx[0][i]
#         # print("rois found to be of size:{}".format(rois.size()))
#         # print(pre_cat[1])
#         # pre = torch.cat((prev, pret), 0)  
#         # print(pre)
#         return rois, xx[1], loss
    

# class STNfor(nn.Module):
#     """
#     Spatial Transformer Network module by hmz
#     """
#     def __init__(self, use_dropout=False):
#         super(STNfor, self).__init__()
#         self._ksize = 3
#         self.dropout = use_dropout

#         # localization net 
#         self.conv1 = nn.Conv2d(512, 256, kernel_size=self._ksize, stride=1, padding=1, bias=False)
#         self.conv2 = nn.Conv2d(256, 64, kernel_size=self._ksize, stride=1, padding=1, bias=False)
#         self.conv3 = nn.Conv2d(64, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
#         self.conv4 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
#         # self.bn1 = nn.BatchNorm2d(32)
#         # self.bn2 = nn.BatchNorm2d(32)
#         # self.bn3 = nn.BatchNorm2d(32)
#         # self.bn4 = nn.BatchNorm2d(32)

#         self.fc1 = nn.Linear(32*5*5, 256)
#         self.fc2 = nn.Linear(256, 6)
        
#         # Initialize the weights/bias with identity transformation
#         self.fc2.weight.data.zero_()
#         self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

#         # self.conv_a = nn.Sequential(
#         #     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
#         #     nn.BatchNorm2d(512),
#         #     nn.ReLU(),
#         #     nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(),
#         #     )
        

#         # # liner layer
#         # self.layer = nn.Sequential(
#         #     nn.Linear(32*8*8, 512),
#         #     nn.ReLU(),
#         #     nn.Linear(512, 128),
#         #     nn.ReLU()
#         # )
        
#         # self.mlp = nn.Sequential(
#         #     nn.Linear(256, 256),
#         #     nn.ReLU()
#         # )
        

#     def forward(self, x):
#         rois= [None]*5
#         # pre_cat = [None]*5
#         # loss = 0
#         xx = x
#         prev = None
#         pret = None
#         # rois = xx[0]
#         for i in range(5):
#             if i<5:

#                 # v = xx[0][i]
#                 # t = xx[1][i]
#                 # v = self.conv_a(v)
#                 # t = self.conv_a(t)
#                 # v = F.adaptive_max_pool2d(v, 8)
#                 # t = F.adaptive_max_pool2d(v, 8)
#                 # v = self.layer(v.view(-1, 32*8*8))
#                 # vmlp = self.mlp(v)
#                 # v = v + vmlp
#                 # t = self.layer(t.view(-1, 32*8*8))
#                 # loss = loss + F.cosine_similarity(v, t)

                
#                 one_feat = xx[0][i]
#                 another_feat = xx[1][i]

#                 feat = torch.cat((one_feat, another_feat), 1)

#                 # v = F.adaptive_avg_pool2d(v, 1)
#                 # v = self.layer(v.view(-1, 256))
#                 # t = F.adaptive_avg_pool2d(t, 1)
#                 # t = self.layer(t.view(-1, 256))
#                 # t:1*2 v:1*2 compute loss between them
#                 # loss = nn.BCELoss()
#                 # pre_cat[i] = loss(v, t)*0.1
#                 # pre_cat[i] = torch.cat((v, t), 0)
#                 # prev.append(v)
#                 # pret.append(t)
#                 # prev = torch.cat((prev, v), 0) if prev is not None else v
#                 # pret = torch.cat((pret, t), 0) if pret is not None else t

#                 x = F.relu(self.conv1(feat.detach()))
#                 # x = F.relu(self.conv1(feat))
#                 x = F.max_pool2d(x, 2)
#                 x = F.relu(self.conv2(x))
#                 x = F.max_pool2d(x, 2)
#                 x = F.relu(self.conv3(x))
#                 # x = F.max_pool2d(x, 2)
#                 x = F.relu(self.conv4(x))
#                 x = F.adaptive_avg_pool2d(x, 5)
                
#                 x = x.view(-1, 32*5*5)
#                 if self.dropout:
#                     x = F.dropout(self.fc1(x), p=0.5)
#                     x = F.dropout(self.fc2(x), p=0.5)
#                 else:
#                     x = self.fc1(x)
#                     x = self.fc2(x) # params [Nx6]
                
#                 x = x.view(-1, 2,3) # change it to the 2x3 matrix 
#                 # print(x.size())
#                 affine_grid_points = F.affine_grid(x, one_feat.size())
#                 assert(affine_grid_points.size(0) == one_feat.size(0)), "The batch sizes of the input images must be same as the generated grid."
#                 rois[i] = F.grid_sample(one_feat, affine_grid_points)
#             else:
#                 rois[i] = xx[0][i]
#         # print("rois found to be of size:{}".format(rois.size()))
#         # print(pre_cat[1])
#         # pre = torch.cat((prev, pret), 0)  
#         # print(pre)
#         return rois, xx[1]
'''  

class BasicIRNet(nn.Module):
    """
    Implementation based on methods from the AIM 2022 Challenge on
    Efficient and Accurate Quantized Image Super-Resolution on Mobile NPUs
    https://arxiv.org/pdf/2211.05910.pdf
    """

    def __init__(self,
                 in_plane,
                 upscale) -> None:
        super(BasicIRNet, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_plane, in_plane, 3, padding=1)
        )

        self.body = nn.ModuleList()
        import math
        self.num_upsample = int(math.log2(upscale))
        for i in range(self.num_upsample):
            self.body.append(conv3x3(int(in_plane/2**i), int(in_plane / 2**(i+1))))

        self.end = nn.Conv2d(int(in_plane / 2**(self.num_upsample)), 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        # self.last = nn.Conv2d(2, 1, kernel_size=3)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        x = self.head(x)
        for i in range(self.num_upsample):
            x = resize(self.body[i](x), scale_factor=(2,2), mode='bilinear')
        # for i in range(self.num_upsample):
        #     x = self.body[i](x)
        out = self.end(x)
        out = self.sigmoid(out)
        return out



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution + batch norm + relu"""
    return torch.nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU()
    )


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        import warnings
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

