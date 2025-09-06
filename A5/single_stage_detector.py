import time
import math
import torch 
import torch.nn as nn
from torch import optim
import torchvision
from a5_helper import *
import matplotlib.pyplot as plt


def hello_single_stage_detector():
    print("Hello from single_stage_detector.py!")


def GenerateAnchor(anc, grid):
  """
  Anchor generator.

  Inputs:
  - anc: Tensor of shape (A, 2) giving the shapes of anchor boxes to consider at
    each point in the grid. anc[a] = (w, h) gives the width and height of the
    a'th anchor shape.
  - grid: Tensor of shape (B, H', W', 2) giving the (x, y) coordinates of the
    center of each feature from the backbone feature map. This is the tensor
    returned from GenerateGrid.
  
  Outputs:
  - anchors: Tensor of shape (B, A, H', W', 4) giving the positions of all
    anchor boxes for the entire image. anchors[b, a, h, w] is an anchor box
    centered at grid[b, h, w], whose shape is given by anc[a]; we parameterize
    boxes as anchors[b, a, h, w] = (x_tl, y_tl, x_br, y_br), where (x_tl, y_tl)
    and (x_br, y_br) give the xy coordinates of the top-left and bottom-right
    corners of the box.
  """
  anchors = None
  ##############################################################################
  # TODO: Given a set of anchor shapes and a grid cell on the activation map,  #
  # generate all the anchor coordinates for each image. Support batch input.   #
  ##############################################################################
  # Replace "pass" statement with your code
  centers = grid.unsqueeze(1)
 
  anc = anc.reshape(1,-1,1,1,2)
  br = centers + anc / 2
  tl = centers - anc / 2
  anchors = torch.cat([tl,br],-1)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return anchors


def GenerateProposal(anchors, offsets, method='YOLO'):
  """
  Proposal generator.

  Inputs:
  - anchors: Anchor boxes, of shape (B, A, H', W', 4). Anchors are represented
    by the coordinates of their top-left and bottom-right corners.
  - offsets: Transformations of shape (B, A, H', W', 4) that will be used to
    convert anchor boxes into region proposals. The transformation
    offsets[b, a, h, w] = (tx, ty, tw, th) will be applied to the anchor
    anchors[b, a, h, w]. For YOLO, assume that tx and ty are in the range
    (-0.5, 0.5).
  - method: Which transformation formula to use, either 'YOLO' or 'FasterRCNN'
  
  Outputs:
  - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
    coordinates of their top-left and bottom-right corners. Applying the
    transform offsets[b, a, h, w] to the anchor [b, a, h, w] should give the
    proposal proposals[b, a, h, w].
  
  """
  assert(method in ['YOLO', 'FasterRCNN'])
  proposals = None
  ##############################################################################
  # TODO: Given anchor coordinates and the proposed offset for each anchor,    #
  # compute the proposal coordinates using the transformation formulas above.  #
  ##############################################################################
  # Replace "pass" statement with your code
  anchors_xywh = anchors.clone()
  anchors_xywh[...,[0,1]] = (anchors[...,[0,1]] + anchors[...,[2,3]]) / 2
  anchors_xywh[...,[2,3]] = anchors[...,[2,3]] - anchors[...,[0,1]]
  if method == 'YOLO':
    anchors_xywh[...,[0,1]] += offsets[...,[0,1]]
    anchors_xywh[...,[2,3]] = anchors_xywh[...,[2,3]] * offsets[...,[2,3]].exp() 
  else:
    anchors_xywh[...,[0,1]] += offsets[...,[0,1]] * anchors_xywh[...,[2,3]]
    anchors_xywh[...,[2,3]] = anchors_xywh[...,[2,3]] * offsets[...,[2,3]].exp()
  proposals = torch.zeros_like(anchors_xywh)
  proposals[...,[0,1]] = anchors_xywh[...,[0,1]] - anchors_xywh[...,[2,3]] / 2
  proposals[...,[2,3]] = anchors_xywh[...,[0,1]] + anchors_xywh[...,[2,3]] / 2
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return proposals


def IoU(proposals, bboxes):
  """
  Compute intersection over union between sets of bounding boxes.

  Inputs:
  - proposals: Proposals of shape (B, A, H', W', 4)
  - bboxes: Ground-truth boxes from the DataLoader of shape (B, N, 5).
    Each ground-truth box is represented as tuple (x_lr, y_lr, x_rb, y_rb, class).
    If image i has fewer than N boxes, then bboxes[i] will be padded with extra
    rows of -1.
  
  Outputs:
  - iou_mat: IoU matrix of shape (B, A*H'*W', N) where iou_mat[b, i, n] gives
    the IoU between one element of proposals[b] and bboxes[b, n].

  For this implementation you DO NOT need to filter invalid proposals or boxes;
  in particular you don't need any special handling for bboxxes that are padded
  with -1.
  """
  iou_mat = None
  ##############################################################################
  # TODO: Compute the Intersection over Union (IoU) on proposals and GT boxes. #
  # No need to filter invalid proposals/bboxes (i.e., allow region area <= 0). #
  # However, you need to make sure to compute the IoU correctly (it should be  #
  # 0 in those cases.                                                          # 
  # You need to ensure your implementation is efficient (no for loops).        #
  # HINT:                                                                      #
  # IoU = Area of Intersection / Area of Union, where                          #
  # Area of Union = Area of Proposal + Area of BBox - Area of Intersection     #
  # and the Area of Intersection can be computed using the top-left corner and #
  # bottom-right corner of proposal and bbox. Think about their relationships. #
  ##############################################################################
  # Replace "pass" statement with your code
  proposals = proposals.reshape(proposals.shape[0],-1,4)
  iou_mat = torch.zeros(proposals.shape[0],proposals.shape[1],bboxes.shape[1],device=bboxes.device,dtype=bboxes.dtype)
  proposals = proposals.unsqueeze(2)
  bboxes = bboxes.unsqueeze(1)
  S_p = (proposals[...,2] - proposals[...,0]) * (proposals[...,3] - proposals[...,1])
  
  S_b = (bboxes[...,2] - bboxes[...,0]) * (bboxes[...,3] - bboxes[...,1])
  inter_x1 = torch.max(proposals[...,0],bboxes[...,0])
  inter_y1 = torch.max(proposals[...,1],bboxes[...,1])
  inter_x2 = torch.min(proposals[...,2],bboxes[...,2])
  inter_y2 = torch.min(proposals[...,3],bboxes[...,3])
  S_i = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
  S_u = S_p + S_b - S_i
  iou_mat = S_i / (S_u + 1e-8)
  iou_mat = iou_mat.clamp(min=0)
  invalid_mask = (S_u <= 0) | (S_p <= 0) | (S_b <= 0)
  iou_mat[invalid_mask] = 0
  invalid_boxes_mask = (bboxes[..., 0] < 0)  # 假设 x1<0 表示无效框
  iou_mat.masked_fill_(invalid_boxes_mask, 0)


    
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return iou_mat


class PredictionNetwork(nn.Module):
  def __init__(self, in_dim, hidden_dim=128, num_anchors=9, num_classes=20, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0 and num_anchors != 0)
    self.num_classes = num_classes
    self.num_anchors = num_anchors

    ##############################################################################
    # TODO: Set up a network that will predict outputs for all anchors. This     #
    # network should have a 1x1 convolution with hidden_dim filters, followed    #
    # by a Dropout layer with p=drop_ratio, a Leaky ReLU nonlinearity, and       #
    # finally another 1x1 convolution layer to predict all outputs. You can      #
    # use an nn.Sequential for this network, and store it in a member variable.  #
    # HINT: The output should be of shape (B, 5*A+C, 7, 7), where                #
    # A=self.num_anchors and C=self.num_classes.                                 #
    ##############################################################################
    # Make sure to name your prediction network pred_layer.
    self.pred_layer = None
    # Replace "pass" statement with your code
    self.pred_layer = nn.Sequential(nn.Conv2d(in_dim,hidden_dim,1),
                                    nn.Dropout(p=drop_ratio),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(hidden_dim,5 * self.num_anchors + self.num_classes,1))
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  def _extract_anchor_data(self, anchor_data, anchor_idx):
    """
    Inputs:
    - anchor_data: Tensor of shape (B, A, D, H, W) giving a vector of length
      D for each of A anchors at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving anchor indices to extract

    Returns:
    - extracted_anchors: Tensor of shape (M, D) giving anchor data for each
      of the anchors specified by anchor_idx.
    """
    B, A, D, H, W = anchor_data.shape
    anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
    extracted_anchors = anchor_data[anchor_idx]
    return extracted_anchors
  
  def _extract_class_scores(self, all_scores, anchor_idx):
    """
    Inputs:
    - all_scores: Tensor of shape (B, C, H, W) giving classification scores for
      C classes at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors at
      which to extract classification scores

    Returns:
    - extracted_scores: Tensor of shape (M, C) giving the classification scores
      for each of the anchors specified by anchor_idx.
    """
    B, C, H, W = all_scores.shape
    A = self.num_anchors
    all_scores = all_scores.contiguous().permute(0, 2, 3, 1).contiguous()
    all_scores = all_scores.view(B, 1, H, W, C).expand(B, A, H, W, C)
    all_scores = all_scores.reshape(B * A * H * W, C)
    extracted_scores = all_scores[anchor_idx]
    return extracted_scores

  def forward(self, features, pos_anchor_idx=None, neg_anchor_idx=None):
    """
    Run the forward pass of the network to predict outputs given features
    from the backbone network.

    Inputs:
    - features: Tensor of shape (B, in_dim, 7, 7) giving image features computed
      by the backbone network.
    - pos_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as positive. These are only given during training; at test-time
      this should be None.
    - neg_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as negative. These are only given at training; at test-time this
      should be None.
    
    The outputs from this method are different during training and inference.
    
    During training, pos_anchor_idx and neg_anchor_idx are given and identify
    which anchors should be positive and negative, and this forward pass needs
    to extract only the predictions for the positive and negative anchors.

    During inference, only features are provided and this method needs to return
    predictions for all anchors.

    Outputs (During training):
    - conf_scores: Tensor of shape (2*M, 1) giving the predicted classification
      scores for positive anchors and negative anchors (in that order).
    - offsets: Tensor of shape (M, 4) giving predicted transformation for
      positive anchors.
    - class_scores: Tensor of shape (M, C) giving classification scores for
      positive anchors.

    Outputs (During inference):
    - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
      scores for all anchors.
    - offsets: Tensor of shape (B, A, 4, H, W) giving predicted transformations
      all all anchors.
    - class_scores: Tensor of shape (B, C, H, W) giving classification scores for
      each spatial position.
    """
    conf_scores, offsets, class_scores = None, None, None
    ############################################################################
    # TODO: Use backbone features to predict conf_scores, offsets, and         #
    # class_scores. Make sure conf_scores is between 0 and 1 by squashing the  #
    # network output with a sigmoid. Also make sure the first two elements t^x #
    # and t^y of offsets are between -0.5 and 0.5 by squashing with a sigmoid  #
    # and subtracting 0.5.                                                     #
    #                                                                          #
    # During training you need to extract the outputs for only the positive    #
    # and negative anchors as specified above.                                 #
    #                                                                          #
    # HINT: You can use the provided helper methods self._extract_anchor_data  #
    # and self._extract_class_scores to extract information for positive and   #
    # negative anchors specified by pos_anchor_idx and neg_anchor_idx.         #
    ############################################################################
    # Replace "pass" statement with your code
    output = self.pred_layer(features)
    B, _, H, W = features.shape
    A=self.num_anchors
    C=self.num_classes
    #将原始数据重组为(B, A, D=5, H=7, W=7)
    anchor_data_raw = output[:,:5 * A,:,:].reshape(-1,A,5,H,W)
    conf_scores_raw = anchor_data_raw[:,:,0:1,:,:]
    offsets_raw = anchor_data_raw[:,:,1:,:,:]
    class_scores_raw = output[:,5 * A:,:,:]

    conf_scores_squashed = conf_scores_raw.sigmoid()
    offsets_squashed = offsets_raw.clone()
    offsets_squashed[:,:,:2,:,:] = torch.sigmoid(offsets_squashed[:,:,:2,:,:]) - 0.5   
    if pos_anchor_idx is None:#inference
      conf_scores = conf_scores_squashed.squeeze(2)
      offsets = offsets_squashed
      class_scores = class_scores_raw
    else:                     #training

      class_scores = self._extract_class_scores(class_scores_raw,pos_anchor_idx)
      
      
      
      #提取正锚框的数据
      
      offsets = self._extract_anchor_data(offsets_squashed,pos_anchor_idx)
      conf_scores_pos = self._extract_anchor_data(conf_scores_squashed,pos_anchor_idx)
      #提取负锚框的数据
      conf_scores_neg = self._extract_anchor_data(conf_scores_squashed,neg_anchor_idx)
      #置信度合并
      conf_scores = torch.cat([conf_scores_pos,conf_scores_neg],dim=0)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return conf_scores, offsets, class_scores


class SingleStageDetector(nn.Module):
  def __init__(self):
    super().__init__()

    self.anchor_list = torch.tensor([[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]]) # READ ONLY
    self.feat_extractor = FeatureExtractor()
    self.num_classes = 20
    self.pred_network = PredictionNetwork(1280, num_anchors=self.anchor_list.shape[0], \
                                          num_classes=self.num_classes)
  def forward(self, images, bboxes):
    """
    Training-time forward pass for the single-stage detector.

    Inputs:
    - images: Input images, of shape (B, 3, 224, 224)
    - bboxes: GT bounding boxes of shape (B, N, 5) (padded)

    Outputs:
    - total_loss: Torch scalar giving the total loss for the batch.
    """
    # weights to multiple to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 1 # for offsets
    w_cls = 1 # for class_prob

    total_loss = None
    ##############################################################################
    # TODO: Implement the forward pass of SingleStageDetector.                   #
    # A few key steps are outlined as follows:                                   #
    # i) Image feature extraction,                                               #
    # ii) Grid and anchor generation,                                            #
    # iii) Compute IoU between anchors and GT boxes and then determine activated/#
    #      negative anchors, and GT_conf_scores, GT_offsets, GT_class,           #
    # iv) Compute conf_scores, offsets, class_prob through the prediction network#
    # v) Compute the total_loss which is formulated as:                          #
    #    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss,  #
    #    where conf_loss is determined by ConfScoreRegression, w_reg by          #
    #    BboxRegression, and w_cls by ObjectClassification.                      #
    # HINT: Set `neg_thresh=0.2` in ReferenceOnActivatedAnchors in this notebook #
    #       (A5-1) for a better performance than with the default value.         #
    ##############################################################################
    # Replace "pass" statement with your code
    
    feature = self.feat_extractor(images)
    B,_,H,W = feature.shape
    grid = GenerateGrid(B)
    self.anchor_list = self.anchor_list.cuda()
    anchor = GenerateAnchor(self.anchor_list,grid)
    
    A,H_,W_ = anchor.shape[1:4]
    iou_mat = IoU(anchor,bboxes)
    
    activated_anc_ind,negative_anc_ind,GT_conf_scores,GT_offsets,GT_class,activated_anc_coord,negative_anc_coord = ReferenceOnActivatedAnchors(anchor,bboxes,grid,iou_mat,neg_thresh=0.2,method='YOLO')
    conf_scores, offsets, class_prob = self.pred_network(feature,activated_anc_ind,negative_anc_ind)
    conf_loss = ConfScoreRegression(conf_scores,GT_conf_scores)
    reg_loss = BboxRegression(offsets,GT_offsets)
    cls_loss = ObjectClassification(class_prob,GT_class,B,A*H_*W_,activated_anc_ind)
    
    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss 
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return total_loss
  
  def inference(self, images, thresh=0.5, nms_thresh=0.7):
    """"
    Inference-time forward pass for the single stage detector.

    Inputs:
    - images: Input images
    - thresh: Threshold value on confidence scores
    - nms_thresh: Threshold value on NMS

    Outputs:
    - final_propsals: Keeped proposals after confidence score thresholding and NMS,
                      a list of B (*x4) tensors
    - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
    - final_class: Corresponding class predictions, a list of B  (*x1) tensors
    """
    final_proposals, final_conf_scores, final_class = [], [], []
    ##############################################################################
    # TODO: Predicting the final proposal coordinates `final_proposals`,         #
    # confidence scores `final_conf_scores`, and the class index `final_class`.  #
    # The overall steps are similar to the forward pass but now you do not need  #
    # to decide the activated nor negative anchors.                              #
    # HINT: Thresholding the conf_scores based on the threshold value `thresh`.  #
    # Then, apply NMS (torchvision.ops.nms) to the filtered proposals given the  #
    # threshold `nms_thresh`.                                                    #
    # The class index is determined by the class with the maximal probability.   #
    # Note that `final_propsals`, `final_conf_scores`, and `final_class` are all #
    # lists of B 2-D tensors (you may need to unsqueeze dim=1 for the last two). #
    ##############################################################################
    # Replace "pass" statement with your code

    #   PredNetwork Outputs (During inference):
    # - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
    #   scores for all anchors.
    # - offsets: Tensor of shape (B, A, 4, H, W) giving predicted transformations
    #   all all anchors.
    # - class_scores: Tensor of shape (B, C, H, W) giving classification scores for
    #   each spatial position.
    feature = self.feat_extractor(images)
    B,_,H,W = feature.shape
    grid = GenerateGrid(B)
    self.anchor_list = self.anchor_list.to(device=images.device)
    anchor = GenerateAnchor(self.anchor_list,grid)#(B, A, H, W, 4)
    
    conf_scores, offsets, class_prob = self.pred_network(feature)
    C = class_prob.shape[1]
    conf_scores = conf_scores.permute(0,2,3,1).reshape(B,-1) #(B,H*W*A')
    offsets = offsets.permute(0,3,4,1,2).reshape(B,-1,4) #(B,H*W*A',4)
    class_prob = class_prob.permute(0,2,3,1).reshape(B,-1,C)#(B,H*W,C)
    anchor = anchor.permute(0,2,3,1,4).reshape(B,-1,4) #(B,H*W*A',4)
    for i in range(B):
      conf_scores_keep_ind = (conf_scores[i] > thresh).nonzero()
      conf_scores_keep = conf_scores[i][conf_scores > thresh]
      ##########
      offsets_keep = offsets[conf_scores_keep_ind]
      anchor_keep = anchor[conf_scores_keep_ind].reshape(-1,4) #(B*H*W*A',4)
      proposals = offsets_keep + anchor_keep #(B*H*W*A',4)
      final_ind = torchvision.ops.nms(proposals,conf_scores_keep,nms_thresh)
      final_proposals = proposals[final_ind] 
      final_conf_scores = conf_scores_keep[final_ind]
      final_class = torch.argmax(class_prob,dim=1,keepdim=True)#(B*H*W,1)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return final_proposals, final_conf_scores, final_class


def nms(boxes, scores, iou_threshold=0.5, topk=None):
  """
  Non-maximum suppression removes overlapping bounding boxes.

  Inputs:
  - boxes: top-left and bottom-right coordinate values of the bounding boxes
    to perform NMS on, of shape Nx4
  - scores: scores for each one of the boxes, of shape N
  - iou_threshold: discards all overlapping boxes with IoU > iou_threshold; float
  - topk: If this is not None, then return only the topk highest-scoring boxes.
    Otherwise if this is None, then return all boxes that pass NMS.

  Outputs:
  - keep: torch.long tensor with the indices of the elements that have been
    kept by NMS, sorted in decreasing order of scores; of shape [num_kept_boxes]
  """

  if (not boxes.numel()) or (not scores.numel()):
    return torch.zeros(0, dtype=torch.long)

  keep = None
  #############################################################################
  # TODO: Implement non-maximum suppression which iterates the following:     #
  #       1. Select the highest-scoring box among the remaining ones,         #
  #          which has not been chosen in this step before                    #
  #       2. Eliminate boxes with IoU > threshold                             #
  #       3. If any boxes remain, GOTO 1                                      #
  #       Your implementation should not depend on a specific device type;    #
  #       you can use the device of the input if necessary.                   #
  # HINT: You can refer to the torchvision library code:                      #
  #   github.com/pytorch/vision/blob/master/torchvision/csrc/cpu/nms_cpu.cpp  #
  #############################################################################
  # Replace "pass" statement with your code
  
  scores_valid = torch.ones_like(scores)
  num_remain = int(scores_valid.sum().item())
  while num_remain > 0:
    valid_ind = (scores_valid > 0).nonzero()
    max_valid_idx = torch.argmax(scores[scores_valid > 0],keepdim=True)
    max_idx = valid_ind[max_valid_idx]
    scores_valid[max_idx] = 0
    num_remain -= 1
    valid_ind = (scores_valid > 0).nonzero()
    if keep is None:
      keep = max_idx.long()
    else:
      keep = torch.cat([keep,max_idx.long()],0)
    proposal = boxes[max_idx].reshape(1,1,1,1,4)
    rest = boxes[scores_valid > 0].reshape(1,num_remain,4)
    iou_mat = IoU(proposal,rest).squeeze()
    invalid_ind = (iou_mat > iou_threshold).nonzero()
    scores_to_elim_ind = valid_ind[invalid_ind]
    scores_valid[scores_to_elim_ind] = 0
    num_remain = int(scores_valid.sum().item())
  if not (topk is None):
    keep = keep[:topk]
  keep = keep.squeeze()
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return keep

def ConfScoreRegression(conf_scores, GT_conf_scores):
  """
  Use sum-squared error as in YOLO

  Inputs:
  - conf_scores: Predicted confidence scores
  - GT_conf_scores: GT confidence scores
  
  Outputs:
  - conf_score_loss
  """
  # the target conf_scores for negative samples are zeros
  GT_conf_scores = torch.cat((torch.ones_like(GT_conf_scores), \
                              torch.zeros_like(GT_conf_scores)), dim=0).view(-1, 1)
  conf_score_loss = torch.sum((conf_scores - GT_conf_scores)**2) * 1. / GT_conf_scores.shape[0]
  return conf_score_loss


def BboxRegression(offsets, GT_offsets):
  """"
  Use sum-squared error as in YOLO
  For both xy and wh

  Inputs:
  - offsets: Predicted box offsets
  - GT_offsets: GT box offsets
  
  Outputs:
  - bbox_reg_loss
  """
  bbox_reg_loss = torch.sum((offsets - GT_offsets)**2) * 1. / GT_offsets.shape[0]
  return bbox_reg_loss

