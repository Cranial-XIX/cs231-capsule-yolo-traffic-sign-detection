import numpy as np
import torch
import torch.nn.functional as F
import utils

def cnn_loss(scores, y, params):
    return (-F.log_softmax(scores, dim=1).gather(
        1, y.unsqueeze(1))).sum() / y.size(0)


def capsule_loss(scores, y, params, x=None, recon=None):
    left = F.relu(0.9 - scores) ** 2
    right = F.relu(scores - 0.1) ** 2
    labels = torch.eye(params.n_classes).to(
        params.device).index_select(dim=0, index=y)
    margin = labels * left + 0.5 * (1. - labels) * right
    margin_loss = margin.sum() / y.size(0)

    recon_loss = 0
    if params.recon:
        recon_loss = F.mse_loss(x, recon)

    return margin_loss + recon_loss


def compute_iou_xy(boxes_pred, boxes_true):
    '''
    Compute intersection over union of two set of boxes
    Args:
      boxes_pred: shape (num_objects, B, 4)
      boxes_true: shape (num_objects, 1, 4)
    Return:
      iou: shape (num_objects, B)
    '''

    num_objects = boxes_pred.size(0)
    B = boxes_pred.size(1)

    lt = torch.max(
        boxes_pred[:,:,:2],                             # [num_objects, B, 2]
        boxes_true[:,:,:2].expand(num_objects, B, 2)    # [num_objects, 1, 2] -> (num_objects, B, 2]
    )

    rb = torch.min(
        boxes_pred[:,:,2:],                             # [num_objects, B, 2]
        boxes_true[:,:,2:].expand(num_objects, B, 2)    # [num_objects, 1, 2] -> (num_objects, B, 2]
    )

    wh = rb - lt # width and height => [num_objects, B, 2]
    wh[wh<0] = 0 # if no intersection, set to zero
    inter = wh[:,:,0] * wh[:,:,1] # [num_objects, B]

    # [num_objects, B, 1] * [num_objects, B, 1] -> [num_objects, B]
    area1 = (boxes_pred[:,:,2]-boxes_pred[:,:,0]) * (boxes_pred[:,:,3]-boxes_pred[:,:,1]) 
    area2 = ((boxes_true[:,:,2]-boxes_true[:,:,0]) * (boxes_true[:,:,3]-boxes_true[:,:,1])).expand(num_objects, B)

    iou = inter / (area1 + area2 - inter) # [num_objects, B]
    return iou

def dark_loss(y_pred, y_true, params):
    # y_pred (:, n_grid, n_grid, 5 * B + C)
    # y_true (:, n_grid, n_grid, 5 + C)
    y_true = y_true.float()

    l_coord, l_noobj = params.l_coord, params.l_noobj
    B, C = params.n_boxes, params.n_classes
    batch_size, n_grid, _, _ = y_true.size()

    # seperate boxes and classes
    y_pred_boxes = y_pred[:,:,:,:5*B]
    y_true_boxes = y_true[:,:,:,:5]

    # add one dimension to seperate B bounding boxes of y_pred
    y_pred_boxes = y_pred_boxes.unsqueeze(-1).view(
        batch_size, n_grid, n_grid, B, 5) 
    y_true_boxes = y_true_boxes.unsqueeze(-1).view(
        batch_size, n_grid, n_grid, 1, 5)

    # mask for grid cells with object and wihout object 
    obj_mask = (y_true_boxes[:,:,:,0,0] == 1) 
    noobj_mask = (y_true_boxes[:,:,:,0,0] == 0)

    # initialize loss
    obj_loss_xy = obj_loss_wh = obj_loss_pc = obj_loss_class = noobj_loss_pc = 0

    # Compute loss for boxes in grid cells containing no object
    if len(y_pred_boxes[noobj_mask]) != 0:
        noobj_y_pred_boxes_pc = y_pred_boxes[noobj_mask][:,:,0]
        noobj_loss_pc = torch.sum((noobj_y_pred_boxes_pc)**2)

    # Compute loss for boxes in grid cells containing object
    if len(y_pred_boxes[obj_mask]) != 0:
        # boxes coords (xc, yc, w, h) in grid cells with object
        obj_true_cwh = y_true_boxes[obj_mask][:,:,1:5]  #(n_objects, 1, 4)
        obj_pred_cwh = y_pred_boxes[obj_mask][:,:,1:5]  #(n_objects, B, 4)
        obj_pred_pc = y_pred_boxes[obj_mask][:,:,0]  #(n_objects, B)
        n_objects = obj_true_cwh.size(0)

        # Compute iou between true boxes and B predicted boxes 
        obj_pred_xyxy = utils.cwh_to_xy_torch(obj_pred_cwh, params.darknet_input, n_grid)
        obj_true_xyxy = utils.cwh_to_xy_torch(obj_true_cwh, params.darknet_input, n_grid)
        iou = compute_iou_xy(obj_pred_xyxy, obj_true_xyxy)

        # Find the target boxes responsible for prediction (boxes with max iou)
        max_iou, max_iou_indices = torch.max(iou, dim=1)

        is_target = torch.zeros_like(iou)
        is_target[range(n_objects), max_iou_indices] = 1
        target_mask = (is_target == 1)
        not_target_mask = (is_target == 0)

        # The loss for boxes not responsible for prediction
        not_target_pred_pc = obj_pred_pc[not_target_mask]
        noobj_loss_pc += torch.sum((not_target_pred_pc)**2)

        # The loss for boxes responsible for prediction
        target_pred_pc = obj_pred_pc[target_mask]
        obj_loss_pc = torch.sum((target_pred_pc - max_iou)**2)

        target_pred_xy = obj_pred_cwh[target_mask][:,0:2]  #(n_objects, 2)
        target_true_xy = obj_true_cwh[:,0,0:2]
        obj_loss_xy = torch.sum((target_pred_xy - target_true_xy)**2)

        target_pred_wh = obj_pred_cwh[target_mask][:,2:4]
        target_true_wh = obj_true_cwh[:,0,2:4]
        obj_loss_wh = torch.sum((torch.sqrt(target_pred_wh) - torch.sqrt(target_true_wh))**2)

        if C != 0:
            y_pred_classes = y_pred[:,:,:,5*B:]
            y_true_classes = y_true[:,:,:,5:]
            obj_true_classes = y_true_classes[obj_mask]
            obj_pred_classes = y_pred_classes[obj_mask]
            obj_loss_class = torch.sum((obj_true_classes - obj_pred_classes)**2)

    loss = (l_coord * obj_loss_xy + \
        l_coord * obj_loss_wh + \
        obj_loss_pc + \
        l_noobj * noobj_loss_pc + \
        obj_loss_class) / batch_size

    return loss


def darkcapsule_loss(caps, y, params):
    y = y.float()
    caps = caps * np.sqrt(2)
    y_r, y_phi = utils.polar_transform(y[:,:,:,:5])     # (:,7,7) (:,7,7,5)
    y_cls = y[:,:,:,5:]                                 # (:,7,7,43)
    cap_phi, cap_cls = caps[:,:,:,:5], caps[:,:,:,5:]   # (:,7,7,5), (:,7,7,43)

    cap_r = (caps ** 2).sum(dim=-1) ** 0.5              # (:,7,7)
    left = F.relu(0.9 - cap_r) ** 2                     # (:,7,7)
    right = F.relu(cap_r - 0.1) ** 2                    # (:,7,7)

    obj_loss = y_r * left + 0.5 * (1 - y_r) * right

    coord_loss = -cap_phi * y_phi
    class_loss = (cap_cls - y_cls) ** 2
    return (obj_loss.sum() + coord_loss.sum() + class_loss.sum()) / y.size(0)


def darkcapsule2_loss(caps, y, params):
    y = y.float()
    r_box, phi_box = utils.polar_transform(y[:,:,:,:5]) # (:,7,7) (:,7,7,5)
    y_cls = y[:,:,:,5:] # (:,7,7,43)
    r = (caps ** 2).sum(dim=-1) ** 0.5 # (:,7,7,43)
    left = F.relu(r_box.unsqueeze(3)*0.9 - r) ** 2
    right = F.relu(r - 0.1) ** 2
    margin_loss = params.l_coord * y_cls * left + \
        params.l_noobj * (1. - y_cls) * right
    direction_loss = (caps / r.unsqueeze(4)) * phi_box.unsqueeze(3)
    return (margin_loss.sum() + direction_loss.sum()) / y.size(0)
