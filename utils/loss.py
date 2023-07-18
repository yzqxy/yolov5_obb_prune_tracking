# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import math
from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel
import numpy as np
import torch.nn.functional as F

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        #边框和角度loss
        self.kld_loss_n = KLDloss(1,fun='log1p')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)


        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.stride = det.stride # tensor([8., 16., 32., ...])
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        # self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.ssi = list(self.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        """
        Args:
            p (list[P3_out,...]): torch.Size(b, self.na, h_i, w_i, self.no), self.na means the number of anchors scales
            targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels])

        Return：
            total_loss * bs (tensor): [1] 
            torch.cat((lbox, lobj, lcls, ltheta)).detach(): [4]
        """

        device = targets.device
        lcls = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        box_loss = torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
   

        # Losses # 依次遍历三个feature map的预测输出pi
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[...,0], dtype=pi.dtype,device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                prediction_pos = pi[b, a, gj, gi]  # prediction subset corresponding to targets, (n_targets, self.no)

                xy      = prediction_pos[:, :2].sigmoid() * 2. - 0.5
                wh      = (prediction_pos[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                angle   = (prediction_pos[:, 4:5].sigmoid() - 0.5) * math.pi
                pbox = torch.cat((xy, wh, angle), 1)


                #方法一 KLDloss
                # kldloss = self.kld_loss_n(pbox,tbox[i])
                # box_loss +=kldloss.mean()
                #  # Objectness    
                # tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * (1 - kldloss).detach().clamp(0).type(tobj.dtype)  # iou ratio
                # 方法二 probloss
                probloss = probiou_loss(pbox,tbox[i])
                box_loss +=probloss.mean()
                # Objectness    
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * (1 - probloss).detach().clamp(0).type(tobj.dtype)  # iou ratio
                # #方法三 KFiou loss
                # kfiouloss = kfiou_loss(pbox,tbox[i],pbox,tbox[i])
                # box_loss +=kfiouloss.mean()
                # # Objectness    
                # tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * (1 - kfiouloss).detach().clamp(0).type(tobj.dtype)  # iou ratio
  
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(prediction_pos[:, 6:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(prediction_pos[:, 6:], t)  # BCE


            obji = self.BCEobj(pi[..., 5], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        box_loss *= self.hyp['box']
        bs = tobj.shape[0]  # batch size


        return ( box_loss +lobj+ lcls ) * bs, torch.cat(( box_loss, lobj,lcls)).detach()


    def build_targets(self, p, targets):
        #input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        # print('nt',nt)
     
        tcls, tbox, indices, anch = [], [], [], []
        # gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        feature_wh = torch.ones(2, device=targets.device)  # feature_wh
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0], # tensor: (5, 2)
                            [1, 0], 
                            [0, 1], 
                            [-1, 0], 
                            [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ],
                            device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i] 
            # gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain=[1, 1, w, h, w, h, 1, 1]
            feature_wh[0:2] = torch.tensor(p[i].shape)[[3, 2]]  # xyxy gain=[w_f, h_f]

            # Match targets to anchors
            # t = targets * gain # xywh featuremap pixel
            t = targets.clone() # (na, n_gt_all_batch, c+1)
            t[:, :, 2:6] /= self.stride[i] # xyls featuremap pixel
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # edge_ls ratio, torch.size(na, n_gt_all_batch, 2)
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare, torch.size(na, n_gt_all_batch)
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter; Tensor.size(n_filter1, c+1)

                # Offsets
                gxy = t[:, 2:4]  # grid xy; (n_filter1, 2)
                # gxi = gain[[2, 3]] - gxy  # inverse
                gxi = feature_wh[[0, 1]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m)) # (5, n_filter1)
                t = t.repeat((5, 1, 1))[j] # (n_filter1, c+1) -> (5, n_filter1, c+1) -> (n_filter2, c+1)
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] # (5, n_filter1, 2) -> (n_filter2, 2)
            else:
                t = targets[0] # (n_gt_all_batch, c+1)
                offsets = 0

            # Define, t (tensor): (n_filter2, [img_index, clsid, cx, cy, l, s, theta, gaussian_θ_labels, anchor_index])
            b, c = t[:, :2].long().T  # image, class; (n_filter2)
            gxy = t[:, 2:4]  # grid xy
            gwh_theta = t[:, 4:7]  # grid wh
            # theta = t[:, 6]

       
            # print('gaussian_theta_labels',gaussian_theta_labels)
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, -1].long()  # anchor indices 取整
            # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, feature_wh[1] - 1), gi.clamp_(0, feature_wh[0] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh_theta), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class


        # return tcls, tbox, indices, anch
        return tcls, tbox, indices, anch


class KLDloss(nn.Module):

    def __init__(self, taf=1.0, fun="sqrt"):
        super(KLDloss, self).__init__()
        self.fun = fun
        self.taf = taf
        self.pi = 3.141592
    def forward(self, pred, target): # pred [[x,y,w,h,angle], ...]
        #assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 5)
        target = target.view(-1, 5)
  
        delta_x = pred[:, 0] - target[:, 0]
        delta_y = pred[:, 1] - target[:, 1]
        
        pre_angle_radian = pred[:, 4]
        targrt_angle_radian = target[:, 4]


        # pre_angle_radian =  self.pi *(((pred[:, 4] * 180 / self.pi ) + 90)/180)
        # targrt_angle_radian = self.pi *(((target[:, 4] * 180 / self.pi ) + 90)/180)

        delta_angle_radian = pre_angle_radian - targrt_angle_radian

        kld =  0.5 * (
                        4 * torch.pow( ( delta_x.mul(torch.cos(targrt_angle_radian)) + delta_y.mul(torch.sin(targrt_angle_radian)) ), 2) / torch.pow(target[:, 2], 2)
                      + 4 * torch.pow( ( delta_y.mul(torch.cos(targrt_angle_radian)) - delta_x.mul(torch.sin(targrt_angle_radian)) ), 2) / torch.pow(target[:, 3], 2)
                     )\
             + 0.5 * (
                        torch.pow(pred[:, 3], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                      + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                      + torch.pow(pred[:, 3], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                      + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                     )\
             + 0.5 * (
                        torch.log(torch.pow(target[:, 3], 2) / torch.pow(pred[:, 3], 2))
                      + torch.log(torch.pow(target[:, 2], 2) / torch.pow(pred[:, 2], 2))
                     )\
             - 1.0

        

        if self.fun == "sqrt":
            kld = kld.clamp(1e-7).sqrt()
        elif self.fun == "log1p":
            kld = torch.log1p(kld.clamp(1e-7))
        else:
            pass

        kld_loss = 1 - 1 / (self.taf + kld)

        return kld_loss
    


def gbb_form(boxes):
    xy, wh, angle = torch.split(boxes, [2, 2, 1], dim=-1)
    return torch.concat([xy, wh.pow(2) / 12., angle], dim=-1)


def rotated_form(a_, b_, angles):
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    a = a_ * torch.pow(cos_a, 2) + b_ * torch.pow(sin_a, 2)
    b = a_ * torch.pow(sin_a, 2) + b_ * torch.pow(cos_a, 2)
    c = (a_ - b_) * cos_a * sin_a
    return a, b, c


def probiou_loss(pred, target, eps=1e-3, mode='l1'):
    """
        pred    -> a matrix [N,5](x,y,w,h,angle - in radians) containing ours predicted box ;in case of HBB angle == 0
        target  -> a matrix [N,5](x,y,w,h,angle - in radians) containing ours target    box ;in case of HBB angle == 0
        eps     -> threshold to avoid infinite values
        mode    -> ('l1' in [0,1] or 'l2' in [0,inf]) metrics according our paper

    """

    gbboxes1 = gbb_form(pred)
    gbboxes2 = gbb_form(target)

    x1, y1, a1_, b1_, c1_ = gbboxes1[:,
                                     0], gbboxes1[:,
                                                  1], gbboxes1[:,
                                                               2], gbboxes1[:,
                                                                            3], gbboxes1[:,
                                                                                         4]
    x2, y2, a2_, b2_, c2_ = gbboxes2[:,
                                     0], gbboxes2[:,
                                                  1], gbboxes2[:,
                                                               2], gbboxes2[:,
                                                                            3], gbboxes2[:,
                                                                                         4]

    a1, b1, c1 = rotated_form(a1_, b1_, c1_)
    a2, b2, c2 = rotated_form(a2_, b2_, c2_)

    t1 = 0.25 * ((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) * (torch.pow(x1 - x2, 2))) + \
         0.5 * ((c1+c2)*(x2-x1)*(y1-y2))
    t2 = (a1 + a2) * (b1 + b2) - torch.pow(c1 + c2, 2)
    t3_ = (a1 * b1 - c1 * c1) * (a2 * b2 - c2 * c2)
    t3 = 0.5 * torch.log(t2 / (4 * torch.sqrt(F.relu(t3_)) + eps))

    B_d = (t1 / t2) + t3
    # B_d = t1 + t2 + t3

    B_d = torch.clip(B_d, min=eps, max=100.0)
    l1 = torch.sqrt(1.0 - torch.exp(-B_d) + eps)
    l_i = torch.pow(l1, 2.0)
    l2 = -torch.log(1.0 - l_i + eps)

    if mode == 'l1':
        probiou = l1
    if mode == 'l2':
        probiou = l2

    return probiou



def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma

def kfiou_loss(pred,
               target,
               pred_decode=None,
               targets_decode=None,
               fun=None,
               beta=1.0 / 9.0,
               eps=1e-6):
    """Kalman filter IoU loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        pred_decode (torch.Tensor): Predicted decode bboxes.
        targets_decode (torch.Tensor): Corresponding gt decode bboxes.
        fun (str): The function applied to distance. Defaults to None.
        beta (float): Defaults to 1.0/9.0.
        eps (float): Defaults to 1e-6.

    Returns:
        loss (torch.Tensor)
    """
    xy_p = pred[:, :2]
    xy_t = target[:, :2]

    pred_decode = gbb_form(pred_decode)
    targets_decode = gbb_form(targets_decode)

    _, Sigma_p = xy_wh_r_2_xy_sigma(pred_decode)
    _, Sigma_t = xy_wh_r_2_xy_sigma(targets_decode)

    Sigma_p=Sigma_p.float()
    Sigma_t=Sigma_t.float()

    # Smooth-L1 norm
    diff = torch.abs(xy_p - xy_t)
    xy_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                          diff - 0.5 * beta).sum(dim=-1)
    Vb_p = 4 * Sigma_p.det().sqrt()
    Vb_t = 4 * Sigma_t.det().sqrt()
    K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
    Sigma = Sigma_p - K.bmm(Sigma_p)
    Vb = 4 * Sigma.det().sqrt()
    Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
    KFIoU = Vb / (Vb_p + Vb_t - Vb + eps)

    if fun == 'ln':
        kf_loss = -torch.log(KFIoU + eps)
    elif fun == 'exp':
        kf_loss = torch.exp(1 - KFIoU) - 1
    else:
        kf_loss = 1 - KFIoU

    loss = (xy_loss + kf_loss).clamp(0)

    return loss