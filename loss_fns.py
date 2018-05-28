import torch
import torch.nn.functional as F

def cnn_loss(scores, y, params):
    return (-F.log_softmax(scores, dim=1).gather(1, y.unsqueeze(1))).sum()


def capsule_loss(scores, y, params):
    left = F.relu(0.9 - scores) ** 2
    right = F.relu(scores - 0.1) ** 2
    labels = torch.eye(params.n_classes).to(
        params.device).index_select(dim=0, index=y)
    margin_loss = labels * left + 0.5 * (1. - labels) * right
    return margin_loss.sum() / y.size(0)

def compute_iou(boxes_pred, boxes_true):
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
        boxes_pred[:,:,:2],                      # [num_objects, B, 2]
        boxes_true[:,:,:2].expand(num_objects, B, 2)    # [num_objects, 1, 2] -> (num_objects, B, 2]
    )

    rb = torch.min(
        boxes_pred[:,:,2:],                      # [num_objects, B, 2]
        boxes_true[:,:,2:].expand(num_objects, B, 2)    # [num_objects, 1, 2] -> (num_objects, B, 2]
    )

    wh = rb - lt # width and height => [num_objects, B, 2]
    wh[wh<0] = 0 # if no intersection, set to zero
    inter = wh[:,:,0] * wh[:,:,1] # [num_objects, B]

    # [num_objects, B, 1] * [num_objects, B, 1] -> [num_objects, B]
    area1 = (boxes_pred[:,:,2]-boxes_pred[:,:,0]) * (boxes_pred[:,:,2]-boxes_pred[:,:,0]) 
    
    # [num_objects, 1, 1] * [num_objects, 1, 1] -> [num_objects, 1] -> [num_objects, B]
    area2 = ((boxes_true[:,:,2]-boxes_true[:,:,0]) * (boxes_true[:,:,2]-boxes_true[:,:,0])).expand(num_objects, B)

    iou = inter / (area1 + area2 - inter) # [num_objects, B]
    
    return iou

def dark_loss(y_pred, y_true, params):
    # y_pred (batch_size, num_grid, num_grid, 5 * B + C)
    # y_true (batch_size, num_grid, num_grid, 5 + C)
    y_true = y_true.float()
    l_coord, l_noobj, B, C = params.l_coord, params.l_noobj, params.num_boxes, params.num_classes
    batch_size, num_grid, _, _ = y_true.shape

    # seperate boxes and classes
    y_pred_boxes = y_pred[:, :, :, 0:5*B]
    y_pred_classes = y_pred[:, :, :, 5*B:]
    y_true_boxes = y_true[:, :, :, 0:5]

    # add one dimension to seperate B bounding boxes of y_pred
    y_pred_boxes = y_pred_boxes.unsqueeze(-1).view(batch_size, num_grid, num_grid, B, 5) 
    y_true_boxes = y_true_boxes.unsqueeze(-1).view(batch_size, num_grid, num_grid, 1, 5)

    # mask for grid cells with object and wihout object 
    obj_mask = (y_true_boxes[:, :, :, 0, 0] == 1) 
    noobj_mask = (y_true_boxes[:, :, :, 0, 0] == 0)

    # initialize loss
    obj_loss_xy, obj_loss_wh, obj_loss_pc, obj_loss_class, noobj_loss_pc = 0, 0, 0, 0, 0

    # Compute loss for boxes in grid cells containing no object
    if len(y_pred_boxes[noobj_mask]) != 0:
        noobj_y_pred_boxes_pc = y_pred_boxes[noobj_mask][:, :, 0]
        noobj_loss_pc = torch.sum((noobj_y_pred_boxes_pc)**2)

    # Compute loss for boxes in grid cells containing object
    if len(y_pred_boxes[obj_mask]) != 0:
        # boxes coords (xc, yc, w, h) in grid cells with object
        obj_true_xywh = y_true_boxes[obj_mask][:, :, 1:5]  #(num_objects, 1, 4)
        obj_pred_xywh = y_pred_boxes[obj_mask][:, :, 1:5]  #(num_objects, B, 4)
        obj_pred_pc = y_pred_boxes[obj_mask][:, :, 0]  #(num_objects, B)
        num_objects = obj_true_xywh.shape[0]

        # Compute iou between true boxes and B predicted boxes  
        iou = compute_iou(obj_pred_xywh, obj_true_xywh)  #(num_objects, B)

        # Find the target boxes responsible for prediction (boxes with max iou)
        max_iou, max_iou_indices = torch.max(iou, dim=1)

        is_target = torch.zeros(iou.shape)
        is_target[range(iou.shape[0]), max_iou_indices] = 1
        target_mask = (is_target == 1)
        not_target_mask = (is_target == 0)

        # The loss for boxes not responsible for prediction
        not_target_pred_pc = obj_pred_pc[not_target_mask]
        noobj_loss_pc += torch.sum((not_target_pred_pc)**2)

        # The loss for boxes responsible for prediction
        target_pred_pc = obj_pred_pc[target_mask]
        obj_loss_pc = torch.sum((target_pred_pc - 1)**2)

        target_pred_xy = obj_pred_xywh[target_mask][:, 0:2]  #(num_objects, 2)
        target_true_xy = obj_true_xywh[:, 0, 0:2]
        obj_loss_xy = torch.sum((target_pred_xy - target_true_xy)**2)

        target_pred_wh = obj_pred_xywh[target_mask][:, 2:4]
        target_true_wh = obj_true_xywh[:, 0, 2:4]
        obj_loss_wh = torch.sum((torch.sqrt(target_pred_wh) - torch.sqrt(target_true_wh))**2)

        if C != 0:
            y_true_classes = y_true[:, :, :, 5:]
            obj_true_classes = y_true_classes[obj_mask]
            obj_pred_classes = y_pred_classes[obj_mask]
            obj_loss_class = torch.sum((obj_true_classes - obj_pred_classes)**2)

    loss = 1./ batch_size * (l_coord * obj_loss_xy + l_coord * obj_loss_wh + obj_loss_pc + l_noobj * noobj_loss_pc + obj_loss_class)
    return loss

def darkcapsule_loss():
    pass