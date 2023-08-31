from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmdet.models.builder import LOSSES
from mmcv.runner.base_module import BaseModule

def gen_dx_bx(xbound, ybound):
    dx = [row[2] for row in [xbound, ybound]] # [0.15, 0.15]
    bx = [row[0] + row[2] / 2.0 for row in [xbound, ybound]] # [-29.925, -14.925]
    nx = [(row[1] - row[0]) / row[2] for row in [xbound, ybound]] # [400, 200]
    return dx, bx, nx


# temp
class CEWithSoftmaxLoss(torch.nn.Module):
    def __init__(self):
        super(CEWithSoftmaxLoss, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, ypred, ytgt): # b, 65, 25, 50
        # ypred: b, 65, 25, 50
        # ytgt: b, 65, 25, 50 values [0-64)
        loss = self.loss_fn(ypred, ytgt)
        return loss

class NLLLoss(torch.nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()
        self.loss_fn = torch.nn.NLLLoss()

    def forward(self, ypred, ytgt):
        # ypred: b, 65, 25, 50, onehot
        # ytgt: b, 65, 25, 50
        ytgt = torch.argmax(ytgt, dim=1) # b, 25, 50 values [0-64)
        loss = self.loss_fn(ypred, ytgt)
        return loss


@LOSSES.register_module()
class GraphLoss(BaseModule):
    def __init__(self, 
                 pc_range, 
                 voxel_size, 
                 cdist_thr: float=1.5, 
                 reduction='mean', 
                 loss_weight=dict(pts=0.1, match=0.005),
                 cost_class:float=1.0, 
                 cost_dist:float=5.0) -> None:
        super(GraphLoss, self).__init__()
        
        # patch_size: [30.0, 60.0] list
        xbound = [pc_range[0],pc_range[3], voxel_size[0]]
        ybound = [pc_range[1],pc_range[4], voxel_size[1]]
        self.dx, self.bx, self.nx = gen_dx_bx(xbound, ybound)
        self.bound = (np.array(self.dx)/2 - np.array(self.bx)) # [30.0, 15.0]
        self.cdist_threshold = np.linalg.norm(cdist_thr / (2*self.bound)) # norlamize distance threshold in meter / 45.0
        self.reduction = reduction
        self.pts_weight = loss_weight['pts']
        self.match_weight = loss_weight['match']

        # self.ce_fn = torch.nn.CrossEntropyLoss()
        self.nll_fn = torch.nn.NLLLoss()

    def forward(self, matches: torch.Tensor, positions: torch.Tensor, masks: torch.Tensor, vectors_gt: list):
        # matches: b c N+1 N+1
        # positions: b c N 3, x y score
        # masks: b c N 1
        # vectors_gt: [b] list of [instance] list of dict
        # matches = matches.exp()

        # iterate in batch
        closs_list = []
        mloss_list = []
        # semloss_list = []
        matches_gt = []
        # semantics_gt = []
        positions = positions[..., :-1] # b c N 2, x y
        for match, position, mask, vector_gt in zip(matches, positions, masks, vectors_gt):
            # match: c N+1 N+1
            # position: c N 2, pixel coords
            # mask: c N 1, M ones
            # vector_gt: [instance] list of dict
            
            ccloss_list = []
            cmloss_list = []
            cmatches_gt = []
            
            for ci, (cmatch, cposition, cmask) in enumerate(zip(match, position, mask)):
                # cmatch: N+1 N+1
                # cposition: N 2
                # cmask: N 1
                cmask = cmask.squeeze(-1) # N
                # cposition_valid = cposition / (torch.tensor(self.nx, device=cposition.device)-1) # normalize 0~1, N 2
                cposition_valid = cposition[cmask == 1] # M 2; x, y
            
                pts_list = [] # len: P
                pts_ins_list = [] # len: P
                pts_ins_order = [] # len: P
                # pts_type_list = []
                for ins, vector in enumerate(vector_gt): # dict
                    pts, pts_num, line_type = vector['pts'], vector['pts_num'], vector['type']
                    pts = pts[:pts_num] # [p, 2] array in meters
                    # normalize coordinates 0~1
                    if line_type == ci:
                        [(pts_list.append((pt + self.bound) / (2*self.bound)), pts_ins_order.append(i)) for i, pt in enumerate(pts)]
                        [pts_ins_list.append(ins) for _ in pts] # instance ID for all vectors
            
                position_gt = torch.tensor(np.array(pts_list), device=position.device).float() # [P, 2] shaped tensor
                match_gt = torch.zeros_like(cmatch) # N+1 N+1

                if len(position_gt) > 0 and len(cposition_valid) > 0:
                    # compute chamfer distance # [N, P] shaped tensor
                    cdist = torch.cdist(cposition_valid, position_gt) # [M, P]
                    nearest_dist, nearest = cdist.min(-1) # [M, ] distances and indices of nearest position_gt -> nearest_ins = [pts_ins_list[n] for n in nearest]
                    # nearest_dist, nearest = cdist.min(0) # [P, ] distances and indices of nearest cposition_valid -> nearest_ins = [pts_ins_list[n] for n in nearest]

                    if len(nearest) > 1: # at least two vertices
                        nearest_ins = []
                        # for i, d in enumerate(nearest_dist):
                        #     nearest_ins.append(pts_ins_list[i] if d < self.cdist_threshold else -1)
                        # for i in range(min(nearest_ins), max(nearest_ins)+1): # for all instance IDs
                        #     indices = [nearest[ni] for ni, x in enumerate(nearest_ins) if x == i] # ni: gt vector index, x: nearest instance ID
                        #     ins_order = []
                        #     [[ins_order.append(oii.item()) for oii in torch.where(nearest==oi)[0]] for oi in indices]
                        for n, d in zip(nearest, nearest_dist):
                            nearest_ins.append(pts_ins_list[n] if d < self.cdist_threshold else -1)
                        for i in range(max(nearest_ins)+1): # for all instance IDs
                            indices = [ni for ni, x in enumerate(nearest_ins) if x == i] # ni: vector index, x: nearest instance ID
                            ins_order = [pts_ins_order[nearest[oi]] for oi in indices]
                            indices_sorted = [idx for ord, idx in sorted(zip(ins_order, indices))]
                            match_gt[indices_sorted[:-1], indices_sorted[1:]] = 1.0
                        dist_map = torch.cdist(cposition_valid, cposition_valid)

                        # for idx_pred, idx_gt in enumerate(nearest):
                        #     semantic_gt[pts_type_list[idx_gt], idx_pred] = 1.0
                        
                        match_gt_sum_backward = match_gt[:, :-1].sum(0) # [N]
                        # leave only one match along row dimension with closest vector
                        multi_cols, = torch.where(match_gt_sum_backward > 1) # [num_cols]
                        for multi_col in multi_cols:
                            rows, = torch.where(match_gt[:, multi_col] > 0)
                            match_gt[rows, multi_col] = 0.0
                            _, min_row_idx = dist_map[rows, multi_col].min(0)
                            match_gt[rows[min_row_idx], multi_col] = 1.0
                        
                        match_gt_sum_forward = match_gt[:-1].sum(1)
                        multi_rows, = torch.where(match_gt_sum_forward > 1)
                        for multi_row in multi_rows:
                            cols, = torch.where(match_gt[multi_row, :] > 0)
                            match_gt[multi_row, cols] = 0.0
                            _, min_col_idx = dist_map[multi_row, cols].min(0)
                            match_gt[multi_row, cols[min_col_idx]] = 1.0

                        cmask_bins = torch.cat([cmask, cmask.new_tensor(1).expand(1)], 0) # M+1
                        match_gt_sum_forward = match_gt[:-1].sum(1) # [N]
                        match_gt[:-1][match_gt_sum_forward == 0, -1] = 1.0
                        assert torch.min(match_gt[:-1].sum(1)) == 1, f"minimum value of row-wise sum expected 1, but got: {torch.min(match_gt[:-1].sum(1))}"
                        assert torch.max(match_gt[:-1].sum(1)) == 1, f"maximum value of row-wise sum expected 1, but got: {torch.max(match_gt[:-1].sum(1))}"

                        match_gt_sum_backward = match_gt[:, :-1].sum(0)
                        match_gt[:, :-1][-1, match_gt_sum_backward == 0] = 1.0
                        assert torch.min(match_gt[:, :-1].sum(0)) == 1, f"minimum value of col-wise sum expected 1, but got: {torch.min(match_gt[:, :-1].sum(0))}"
                        assert torch.max(match_gt[:, :-1].sum(0)) == 1, f"maximum value of col-wise sum expected 1, but got: {torch.max(match_gt[:, :-1].sum(0))}"

                        cmatch_valid = cmatch[cmask_bins == 1][:, cmask_bins == 1] # M+1 M+1
                        match_gt_valid = match_gt[cmask_bins == 1][:, cmask_bins == 1] # M M+1
                        assert torch.min(match_gt_valid[:-1].sum(1)) == 1, f"minimum value of row-wise sum expected 1, but got: {torch.min(match_gt_valid[:-1].sum(1))}"
                        assert torch.max(match_gt_valid[:-1].sum(1)) == 1, f"maximum value of row-wise sum expected 1, but got: {torch.max(match_gt_valid[:-1].sum(1))}"
                        assert torch.min(match_gt_valid[:, :-1].sum(0)) == 1, f"minimum value of col-wise sum expected 1, but got: {torch.min(match_gt_valid[:, :-1].sum(0))}"
                        assert torch.max(match_gt_valid[:, :-1].sum(0)) == 1, f"maximum value of col-wise sum expected 1, but got: {torch.max(match_gt_valid[:, :-1].sum(0))}"

                        # add minibatch dimension and class first
                        cmatch_valid = cmatch_valid.unsqueeze(0).transpose(1, 2).contiguous() # 1 M+1 M+1 class dim first
                        match_gt_valid = match_gt_valid.unsqueeze(0) # 1 M+1 M+1

                        # backward col -> row
                        match_gt_valid_backward = match_gt_valid.argmax(1) # col -> row [1, M+1]
                        cmatch_loss_backward = self.nll_fn(cmatch_valid[..., :-1], match_gt_valid_backward[..., :-1])

                        # forward row -> col
                        # match_valid = match_valid.transpose(1, 2) # [1, M+1, M+1]
                        match_gt_valid_forward = match_gt_valid.argmax(2) # row -> col [1, M+1]
                        cmatch_loss_forward = self.nll_fn(cmatch_valid[..., :-1], match_gt_valid_forward[..., :-1])

                        match_loss = (cmatch_loss_forward + cmatch_loss_backward)
                        # match_loss = match_loss_forward

                        # semantic_valid = semantic[:, mask == 1].unsqueeze(0) # [1, 3, M]
                        # semantic_gt_valid = semantic_gt[:, mask == 1].unsqueeze(0) # [1, 3, M]
                        # assert torch.min(semantic_gt_valid.sum(1)) == 1, f"minimum value of semantic gt sum expected 1, but got: {torch.min(semantic_gt_valid.sum(1))}"
                        # assert torch.max(semantic_gt_valid.sum(1)) == 1, f"maximum value of semantic gt sum expected 1, but got: {torch.max(semantic_gt_valid.sum(1))}"
                        # semantic_gt_valid = semantic_gt_valid.argmax(1) # [1, M]

                        # semantic_loss = self.nll_fn(semantic_valid, semantic_gt_valid)

                        # _, nearest_gt = cdist.min(-1)
                        # coord_loss = F.l1_loss(cposition_valid, position_gt[nearest_gt])                
                        coord_loss = F.l1_loss(cposition_valid, position_gt[nearest])                
                    else:
                        coord_loss = position_gt.new_tensor(0.0)
                        match_loss = position_gt.new_tensor(0.0)
                        # semantic_loss = position_gt.new_tensor(0.0)
                else:
                    coord_loss = position_gt.new_tensor(0.0)
                    match_loss = position_gt.new_tensor(0.0)
                    # semantic_loss = position_gt.new_tensor(0.0)
                
                ccloss_list.append(coord_loss) # c list of float
                cmloss_list.append(match_loss) # c list of float
                cmatches_gt.append(match_gt) # c list of [N+1 N+1] tensor
            
            coord_loss = torch.stack(ccloss_list).mean()
            match_loss = torch.stack(cmloss_list).mean()
            match_gt = torch.stack(cmatches_gt) # c N+1 N+1
            
            closs_list.append(coord_loss)
            mloss_list.append(match_loss)
            matches_gt.append(match_gt) # b list of [c N+1 N+1] tensor
        
        closs_batch = torch.stack(closs_list) # [b,]
        mloss_batch = torch.stack(mloss_list) # [b,]
        matches_gt = torch.stack(matches_gt) # b c N+1 N+1

        if self.reduction == 'none':
            pass
        elif self.reduction == 'mean':
            closs_batch = torch.mean(closs_batch)
            mloss_batch = torch.mean(mloss_batch)
            # semloss_batch = torch.mean(semloss_batch)
        elif self.reduction == 'sum':
            closs_batch = torch.sum(closs_batch)
            mloss_batch = torch.sum(mloss_batch)
            # semloss_batch = torch.sum(semloss_batch)
        else:
            raise NotImplementedError
        
        return self.pts_weight*closs_batch, self.match_weight*mloss_batch, matches_gt



def calc_loss():
    pass
