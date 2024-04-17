import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss
from mmcv.runner.base_module import BaseModule
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.datasets.pipelines import to_tensor

from projects.mmdet3d_plugin.datasets.map_utils.rasterize import preprocess_map
from projects.mmdet3d_plugin.instagram.modules.graph import vectorize_graph

def normalize_2d_bbox(bboxes, pc_range):

    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[...,0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[...,1:2] = cxcywh_bboxes[...,1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h,patch_w,patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes

def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])

    return bboxes
def denormalize_2d_pts(pts, pc_range):
    """Denormalize point coordinates

    Args:
        pts (list[torch.Tensor]): [num_vec] list of shape [num_pts, 2]
        pc_range (_type_): _description_

    Returns:
        list[torch.Tensor]: denormalized pts (with same shape)
    """
    new_pts = [pt.clone() for pt in pts]
    for i, pt in enumerate(pts):
        new_pts[i][...,0:1] = (pt[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0]) # pt*30 - 15
        new_pts[i][...,1:2] = (pt[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1]) # pt*60 - 30
    return new_pts
@HEADS.register_module()
class InstaGraMHead(BaseModule):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 num_classes,
                 as_two_stage=False,
                 transformer=None,
                 sample_dist=1.5,
                 bev_h=30,
                 bev_w=30,
                 voxel_size=[0.15, 0.15, 4],
                 positional_encoding=None,
                 transform_method='minmax',
                 loss_vtx=dict(
                     type='BinaryCrossEntropy',
                     use_sigmoid=True,
                     class_weight=None,
                     loss_weight=2.0,),
                 loss_dtm=dict(
                     type='MSELoss',
                     loss_weight=2.0,),
                 loss_graph=None,
                 **kwargs
                 ):
        super(InstaGraMHead, self).__init__()

        self.num_classes = num_classes
        self.sample_dist = sample_dist
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.voxel_size = voxel_size
        self.transform_method = transform_method
        self.fp16_enabled = False

        self.as_two_stage = as_two_stage
        self.bev_encoder_type = transformer.encoder.type
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        
        self.loss_vtx = build_loss(loss_vtx)
        self.loss_dtm = build_loss(loss_dtm)
        self.loss_graph = build_loss(loss_graph)
        
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        self.pc_range = self.transformer.encoder.pc_range # [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        
        # InstaGraM
        self.dist_thr = transformer.decoder.dist_thr
        
        self._init_layers()

    def _init_layers(self):
        """Initialize BEV embedding of head."""

        if not self.as_two_stage:
            if self.bev_encoder_type == 'BEVFormerEncoder':
                self.bev_embedding = nn.Embedding(
                    self.bev_h * self.bev_w, self.embed_dims)
            else:
                self.bev_embedding = None

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
    
    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None,  only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape # [B, 6, 256, 15, 25]
        dtype = mlvl_feats[0].dtype
        # import pdb;pdb.set_trace()
        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding.weight.to(dtype) # [200*100, 256]

            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                device=bev_queries.device).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev
        )

        bev_embed, vertex, distance, matches, vertices, graph_cls, masks = outputs
        outs = {
            'bev_embed': bev_embed,
            'bev_pts_scores': vertex,
            'bev_embed_preds': distance,
            'pts_preds': vertices,
            'cls_preds': graph_cls,
            'adj_preds': matches,
            'pts_masks': masks,
        }

        return outs
    def transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box. Single batch input only

        Args:
            pts (list[Tensor]): the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
                Has length num_vec of tensor shape [num_pts, 2]
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            (list[Tensor]): The bbox [cx, cy, w, h] transformed from points.
                Has length num_vec of tensor [4]
        """
        # [num_vec] list of [num_pts] tensor
        pts_y = [pt[:, 0] if y_first else pt[:, 1] for pt in pts]
        pts_x = [pt[:, 1] if y_first else pt[:, 0] for pt in pts]
        if self.transform_method == 'minmax':
            # import pdb;pdb.set_trace()
            # tensor [num_vec, 1]
            xmin = torch.stack([pt_x.min(dim=-1, keepdim=True)[0] for pt_x in pts_x])
            xmax = torch.stack([pt_x.max(dim=-1, keepdim=True)[0] for pt_x in pts_x])
            ymin = torch.stack([pt_y.min(dim=-1, keepdim=True)[0] for pt_y in pts_y])
            ymax = torch.stack([pt_y.max(dim=-1, keepdim=True)[0] for pt_y in pts_y])
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=-1) # [num_vec, 4]
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # import pdb;pdb.set_trace()
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        # import pdb;pdb.set_trace()
        assign_result, order_index = self.assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # pts_sampling_result = self.sampler.sample(assign_result, pts_pred,
        #                                       gt_pts)

        
        # import pdb;pdb.set_trace()
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # pts targets
        # import pdb;pdb.set_trace()
        # pts_targets = torch.zeros_like(pts_pred)
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            # import pdb;pdb.set_trace()
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_single(self,
                    bev_pts_scores,
                    bev_embed_preds,
                    pts_preds,
                    cls_preds,
                    adj_preds,
                    pts_masks,
                    gt_vectors_list,
                    gt_vtm,
                    gt_dtm):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            bev_pts_scores (Tensor): Vertex classification score in BEV, 
                has shape [bs, cs*cs+1, h//8, w//8].
            bev_embed_preds (Tensor): Distance transform map in BEV                    
                has shape [bs, num_classes, h, w].
            pts_preds (Tensor): Vertex coordinates in BEV, normalized
                format (x, y, score), has shape (bs, N, 3).
            cls_preds (Tensor): Vertex classification, 
                has shape (bs, num_classes, N).
            adj_preds (Tensor): Adjacency score matrix of vertices, 
                has shape (bs, N+1, N+1). N+1 includes dustbin
                score.
            pts_masks (Tensor): Mask for valid vertices as float
                (0.0 or 1.0), has shape (bs, N, 1).
            gt_vectors_list (list[list[dict]]): Ground truth vectors
                (bs) length list of (num_gt) list of dict('pts', 'pts_num', 'type').
            gt_vtm (Tensor): List of vertex score map
                has shape (bs, cs*cs+1, h//8, w//8).
            gt_dtm (Tensor): Mask for valid vertices as float
                has shape (bs, num_classes, h, w).
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        loss_vtx = self.loss_vtx(bev_pts_scores, gt_vtm)
        loss_dtm = self.loss_dtm(bev_embed_preds.sigmoid(), gt_dtm)
        loss_cls, loss_match, gt_adj_mat, gt_labels = self.loss_graph(adj_preds, pts_preds, cls_preds, pts_masks, gt_vectors_list)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_vtx = torch.nan_to_num(loss_vtx)
            loss_dtm = torch.nan_to_num(loss_dtm)
            loss_cls = torch.nan_to_num(loss_cls)
            loss_match = torch.nan_to_num(loss_match)
        return loss_vtx, loss_dtm, loss_cls, loss_match #, gt_adj_mat, gt_labels

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                bev_pts_scores (Tensor): Vertex classification score in BEV, 
                    has shape [bs, cs*cs+1, h//8, w//8].
                bev_embed_preds (Tensor): Distance transform map in BEV                    
                    has shape [bs, num_classes, h, w].
                pts_preds (Tensor): Vertex coordinates in BEV, normalized
                    format (x, y, score), has shape (bs, N, 3).
                cls_preds (Tensor): Vertex classification, 
                    has shape (bs, num_classes, N).
                adj_preds (Tensor): Adjacency score matrix of vertices, 
                    has shape (bs, N+1, N+1). N+1 includes dustbin
                    score.
                pts_masks (Tensor): Mask for valid vertices as float
                    (0.0 or 1.0), has shape (bs, N, 1).
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        
        # import pdb;pdb.set_trace()
        bev_pts_scores = preds_dicts['bev_pts_scores'] # (bs, 3, h//8, w//8)
        bev_embed_preds = preds_dicts['bev_embed_preds'] # (bs, 3, h, w)
        pts_preds  = preds_dicts['pts_preds'] # (bs, 3, N, 3)
        cls_preds  = preds_dicts['cls_preds'] # (bs, 3, N)
        adj_preds = preds_dicts['adj_preds'] # (bs, 3, N+1, N+1)
        pts_masks = preds_dicts['pts_masks'] # (bs, 3, N, 1)

        device = gt_labels_list[0].device
        
        gt_vectors_list = []
        gt_vtm_list = []
        gt_dtm_list = []
        for gt_vecs, gt_labels in zip(gt_vecs_list, gt_labels_list):
            prep_dict = preprocess_map(dict(gt_labels_3d=gt_labels, gt_bboxes_3d=gt_vecs),
                                       self.pc_range,
                                       voxel_size=self.voxel_size,
                                       num_classes=self.num_classes,
                                       dt_threshold=self.dist_thr)
            gt_vectors_list.append(prep_dict['gt_vectors'])
            gt_vtm_list.append(to_tensor(prep_dict['vertex_mask']).to(device))
            gt_dtm_list.append(to_tensor(prep_dict['distance_transform']).to(device))
        gt_vtm = torch.stack(gt_vtm_list).to(device).long().argmax(1) # b h w
        gt_dtm = torch.stack(gt_dtm_list).to(device)

        loss_vtx, loss_dtm, loss_cls, loss_match = self.loss_single(
            bev_pts_scores, bev_embed_preds, pts_preds, cls_preds,
            adj_preds, pts_masks, gt_vectors_list, gt_vtm, gt_dtm)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        # if enc_cls_scores is not None:
        #     binary_labels_list = [
        #         torch.zeros_like(gt_labels_list[i])
        #         for i in range(len(all_gt_labels_list))
        #     ]
        #     # TODO bug here
        #     enc_loss_cls, enc_losses_bbox, enc_losses_iou, enc_losses_pts, enc_losses_dir = \
        #         self.loss_single(enc_cls_scores, enc_bbox_preds, enc_pts_preds,
        #                          gt_bboxes_list, binary_labels_list, gt_pts_list,gt_bboxes_ignore)
        #     loss_dict['enc_loss_cls'] = enc_loss_cls
        #     loss_dict['enc_loss_bbox'] = enc_losses_bbox
        #     loss_dict['enc_losses_iou'] = enc_losses_iou
        #     loss_dict['enc_losses_pts'] = enc_losses_pts
        #     loss_dict['enc_losses_dir'] = enc_losses_dir

        # loss from the last decoder layer
        loss_dict['loss_vtx'] = loss_vtx
        loss_dict['loss_dtm'] = loss_dtm
        loss_dict['loss_cls'] = loss_cls
        loss_dict['loss_match'] = loss_match
        # loss_dict['gt_adj_mat'] = gt_adj_mat
        # loss from other decoder layers
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts:
                bev_pts_scores (Tensor): Vertex classification score in BEV, 
                    has shape [bs, cell*cell+1, h//8, w//8].
                bev_embed_preds (Tensor): Distance transform map in BEV                    
                    has shape [bs, num_classes, h, w].
                pts_preds (Tensor): Vertex coordinates in BEV, normalized
                    format (x, y, score), has shape (bs, num_classes, N, 3).
                adj_preds (Tensor): Adjacency score matrix of vertices, 
                    has shape (bs, num_classes, N+1, N+1). N+1 includes dustbin
                    score.
                pts_masks (Tensor): Mask for valid vertices as float
                    (0.0 or 1.0), has shape (bs, num_classes, N, 1).
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # bboxes: xmin, ymin, xmax, ymax
        ret_list = []
        num_samples = preds_dicts['pts_preds'].shape[0]
        for i in range(num_samples):
            # pts: [num_vec] list of tensor [num_pts, 2]
            # scores: tensor [num_vec]
            # labels: tensor [num_vec]
            pts, scores, labels = vectorize_graph(preds_dicts['pts_preds'][i], 
                                                  preds_dicts['adj_preds'][i],
                                                  preds_dicts['cls_preds'][i],
                                                  preds_dicts['pts_masks'][i])
            if len(pts) == 0:
                bboxes = scores.new_tensor([])
            else:
                bboxes = self.transform_box(pts) # tensor [num_vec, 4] cx cy w h            
                pts = denormalize_2d_pts(pts, self.pc_range)
                bboxes = denormalize_2d_bbox(bboxes, self.pc_range)
            
            ret_list.append([bboxes, scores, labels, pts])

        return ret_list

