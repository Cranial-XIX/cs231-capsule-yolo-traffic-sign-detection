import config
import json
import numpy as np
import os
import pickle
import shutil
import torch

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# =============================================================================
# Training related utils
# =============================================================================
class Params():
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! " \
            "Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, params, optimizer=None):
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location = params.device)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    return checkpoint

# =============================================================================
# Loss related utils
# =============================================================================
def sine_and_cosine(x):
    return torch.sin(x), torch.cos(x)


def polar_transform(x):
    assert x.shape[-1] == 5, "polar transform failed, dimension mismatched"
    sh = x.shape
    x = x.view(-1, 5)
    r, x, y, h, w = torch.chunk(x, 5, 1)
    f1, f2, f3, f4 = x*np.pi, y*np.pi, h*np.pi, w*np.pi*2
    (s1, c1), (s2, c2), (s3, c3), (s4, c4) = list(
        map(sine_and_cosine, [f1, f2, f3, f4]))

    x1 = s1
    x2 = s1 * c2
    x3 = s1 * s2 * c3
    x4 = s1 * s2 * s3 * c4
    x5 = s1 * s2 * s3 * s4

    x_hat = torch.cat([x1, x2, x3, x4, x5], 1)
    return r.view(*sh[:-1]), x_hat.view(*sh[:-1], 5)


# =============================================================================
# Data related utils
# =============================================================================
def load_data(data_dir, is_small=False):
    if is_small:
        train_data_path = data_dir + config.tr_sm_d
        eval_data_path = data_dir + config.ev_sm_d
    else:
        train_data_path = data_dir + config.tr_d
        eval_data_path = data_dir + config.ev_d

    x_tr, y_tr = pickle.load(open(train_data_path, 'rb'))
    x_ev, y_ev = pickle.load(open(eval_data_path, 'rb'))
    return x_tr, y_tr, x_ev, y_ev


def make_small_data(data_dir, n=128):
    x_tr, y_tr, x_ev, y_ev = load_data(data_dir)
    train_data_path = data_dir + '/' + config.tr_sm_d
    eval_data_path = data_dir + '/' + config.ev_sm_d
    pickle.dump((x_tr[:n], y_tr[:n]), open(train_data_path, 'wb'))
    pickle.dump((x_ev[:n], y_ev[:n]), open(eval_data_path, 'wb'))


def center_rgb(x):
    return (x - 128) / 128


def augmentation(x, model_name, max_shift=4, max_lightness_increase=0.05):
    _, h, w, _ = x.shape
    if model_name in ('capsule', 'cnn'):
        h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
        source_height_slice = slice(max(0, h_shift), h_shift + h)
        source_width_slice = slice(max(0, w_shift), w_shift + w)
        target_height_slice = slice(max(0, -h_shift), -h_shift + h)
        target_width_slice = slice(max(0, -w_shift), -w_shift + w)

        shifted_image = np.zeros_like(x)
        shifted_image[:, source_height_slice, source_width_slice, :] = \
            x[:, target_height_slice, target_width_slice, :]

    hsv = rgb_to_hsv((x.reshape(-1, 3) + 1) / 2)
    hsv[:, 2] += np.random.rand() * max_lightness_increase
    rgb = hsv_to_rgb(hsv).reshape(-1, h, w, 3)
    
    return rgb


def shuffle(x, y):
    i = np.random.permutation(len(y))
    return x[i], y[i]

def get_image_name(i):
    if i < 10:
        name = '0000' + str(i) + '.ppm'
    elif i < 100:
        name = '000' + str(i) + '.ppm'
    elif i < 1000:
        name = '00' + str(i) + '.ppm'
    elif i < 10000:
        name = '0' + str(i) + '.ppm'
    else:
        name = str(i) + '.ppm'
    assert(len(name) == 9)
    return name


# =============================================================================
# Bounding box related utils
# =============================================================================
def xy_to_cwh(box_xy):
    # box_xy [x1, y1, x2, y2]
    # Given top left point and right bottom point coordinates
    # Compute center coordinates, height and weight
    x1, y1, x2, y2 = box_xy
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    cwh = [xc, yc, w, h]
    return cwh

def cwh_to_xy(box_cwh):
    # box_cwh (xc, yc, w, h)
    # Given top left point and right bottom point coordinates
    # Compute center coordinates, height and weight
    xc, yc, w, h = box_cwh
    x1 = xc - w / 2
    x2 = xc + w / 2
    y1 = yc - h / 2
    y2 = yc + h / 2
    xy = [x1, y1, x2, y2]
    return xy

def resize_box_xy(orig_hw, resized_hw, box_xy):
    # Resize box
    # orig_h, orig_w: orginal image size
    # resized_h, resized_w: resized image size
    # x1, y1, x2, y2: orginal box coords
    orig_h, orig_w = orig_hw
    resized_h, resized_w = resized_hw
    x1, y1, x2, y2 = box_xy
    w_ratio = 1. * resized_w / orig_w
    h_ratio = 1. * resized_h / orig_h
    resized_x1 = x1 * w_ratio
    resized_x2 = x2 * w_ratio
    resized_y1 = y1 * h_ratio
    resized_y2 = y2 * h_ratio
    resized_xy = [resized_x1, resized_y1, resized_x2, resized_y2]
    return resized_xy

def normalize_box_cwh(image_hw, n_grid, box_cwh):
    # Normalize box height and weight to be 0-1
    image_h, image_w = image_hw
    xc, yc, box_w, box_h = box_cwh
    normalized_w = 1. * box_w / image_w
    normalized_h = 1. * box_h / image_h

    grid_w = 1. * image_w / n_grid
    grid_h = 1. * image_h / n_grid
    col = int(xc / grid_w)
    row = int(yc / grid_h)
    normalized_xc = 1. * (xc - col * grid_w) / grid_w
    normalized_yc = 1. * (yc - row * grid_h) / grid_h
    normalized_cwh = [normalized_xc, normalized_yc, normalized_w, normalized_h]
    positon = [row, col]
    return normalized_cwh, positon


def denorm_boxes_cwh_vec(image_hw, n_grid, norm_cwh, grid_indices):
    """ denormalize bounding box (vectorized version)
    
    Args:
        - image_hw: height and width of images for each box, tuple (h, w)
          if images of same size. Shape (num_boxes, 2) if different size.
        - n_grid: number of grid
        - norm_cwh: normalized xc, yc, w, h of boxes, of shape (num_boxes, 4)
        - grid_indices: #row, #col of boxes in grids, of shape (num_boxes, 2)
    
    Return:
        - cwh: denormalized xc, yc, w, h of boxes, of shape (num_boxes, 4).
    """ 
    image_hw = np.array(image_hw).reshape(-1, 2)
    image_wh = image_hw[:, [1, 0]]
    grids_wh = 1. * image_wh / n_grid
    scale = np.concatenate((grids_wh, image_wh), axis=1)
    cwh = norm_cwh * scale
    cwh[:, 0:2] += grid_indices[:, [1, 0]] * grids_wh
    return cwh

def cwh_to_xy_vec(cwh):
    """ Convert center, width, height of a box to upper left and lower 
        right coordinates (vectorized version). 
    
    Args:
        cwh: xc, yc, w, h of boxes. of shape (num_boxes, 4)
    
    Return:
        xy: x1, y1, x2, y2 of boxes. of shape (num_boxes, 4)
    """
    xy = np.zeros_like(cwh)
    xy[:, 0] = cwh[:, 0] - cwh[:, 2] / 2
    xy[:, 1] = cwh[:, 1] - cwh[:, 3] / 2
    xy[:, 2] = cwh[:, 0] + cwh[:, 2] / 2
    xy[:, 3] = cwh[:, 1] + cwh[:, 3] / 2
    return xy

def cwh_to_xy_torch(cwh):
    """ Convert center, width, height of a box to upper left and lower 
        right coordinates (torch version). 
    
    Args:
        cwh: xc, yc, w, h of boxes. of shape  (n_objects, B, 4)
    
    Return:
        xy: x1, y1, x2, y2 of boxes. of shape (n_objects, B, 4)
    """
    xy = torch.zeros_like(cwh)
    xy[:, :, 0] = cwh[:, :, 0] - cwh[:, :, 2] / 2
    xy[:, :, 1] = cwh[:, :, 1] - cwh[:, :, 3] / 2
    xy[:, :, 2] = cwh[:, :, 0] + cwh[:, :, 2] / 2
    xy[:, :, 3] = cwh[:, :, 1] + cwh[:, :, 3] / 2
    return xy

def y_to_boxes_vec(y, params, image_hw = None, conf_th = 0.5):
    """Convert output of network or ground truth to boxes (vectorized version).
    
    Args:
        - y: output of network or ground truth.
          shape (batch_size, n_grid, n_grid, 5 * B + C), for gt B = 1.
        - conf_th: confidence threshold for containing object or not
        - image_hw: height and width of images. For metric, None. For predict,
        of shape (batch_size, 2).
        - n_classes: number of classes
    
    Return:
        - image_indices: the index of image for each box 
          of shape (num_boxes,)
        - xy: the cooridnates (x1, y1, x2, y2) of boxes,
          of shape (num_boxes, 4)
        - classes: the class index of boxes,
          of shape (num_boxes,) or None
    """
    batch_size, n_grid, _, D = y.shape
    C = params.n_classes
    B = int((D - C) / 5)

    y_boxes = y[:, :, :, 0:5*B]
    y_boxes = y_boxes.reshape(batch_size, n_grid, n_grid, B, 5)
    indices = np.argwhere(y_boxes[:, :, :, :, 0] > conf_th) #(num_boxes, 4)
    mask = y_boxes[:, :, :, :, 0] > conf_th
    cwh = y_boxes[mask, 1:5]
    image_indices = indices[:, 0]
    grid_indices = indices[:, 1:3]

    if image_hw is None:
        image_hw = (params.darknet_input, params.darknet_input)
    else:
        image_hw = image_hw[image_indices]

    cwh = denorm_boxes_cwh_vec(image_hw, n_grid, cwh, grid_indices)
    xy = cwh_to_xy_vec(cwh)

    if C != 0:
        y_classes = y[:, :, :, 5*B:]
        classes_onehot = y_classes[indices[:, 0], indices[:, 1], indices[:, 2]]
        classes = np.argmax(classes_onehot, axis = 1)
    else:
        classes = None
    return image_indices, xy, classes

def cwh_to_xy_torch(cwh, img_size, n_grid):
    """ Convert normalized center, width, height of a box to upper left and lower 
        right coordinates (torch version). 
    
    Args:
        cwh: xc, yc, w, h of boxes, of shape (n_objects, B, 4)
        indices: #row, #col of boxes, of shape (n_objects, 2)
    
    Return:
        xy: x1, y1, x2, y2 of boxes. of shape (n_objects, B, 4)
    """
    grid_size = 1. * img_size / n_grid
    xy = torch.zeros_like(cwh)
    xy[:, :, 0] = cwh[:, :, 0] * grid_size - cwh[:, :, 2] * img_size / 2
    xy[:, :, 1] = cwh[:, :, 1] * grid_size - cwh[:, :, 3] * img_size / 2
    xy[:, :, 2] = cwh[:, :, 0] * grid_size + cwh[:, :, 2] * img_size / 2
    xy[:, :, 3] = cwh[:, :, 1] * grid_size + cwh[:, :, 3] * img_size / 2
    xy = xy.detach()
    return xy

