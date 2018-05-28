import json
import numpy as np
import os
import pickle
import shutil
import torch


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
# Training related utils
# =============================================================================
def sine_and_cosine(x):
    return np.sin(x), np.cos(x)


def polar_transform(x):
    assert x.shape[-1] == 5, "polar transform failed, dimension mismatched"
    sh = x.shape
    x = x.reshape(-1, 5)
    pc, x, y, h, w = np.hsplit(x, 5)
    r, f1, f2, f3, f4 = pc, x*np.pi, y*np.pi, h*np.pi, w*np.pi*2
    (s1, c1), (s2, c2), (s3, c3), (s4, c4) = list(
        map(sine_and_cosine, [f1, f2, f3, f4]))

    x1 = s1
    x2 = s1 * c2
    x3 = s1 * s2 * c3
    x4 = s1 * s2 * s3 * c4
    x5 = s1 * s2 * s3 * s4

    x_hat = r * np.concatenate([x1, x2, x3, x4, x5], 1)
    return x_hat.reshape(*sh[:-1], 5)
# =============================================================================
# Data related utils
# =============================================================================
def load_data(train_data_path, eval_data_path):
    x_tr, y_tr = pickle.load(open(train_data_path, 'rb'))
    x_ev, y_ev = pickle.load(open(eval_data_path, 'rb'))
    return x_tr, y_tr, x_ev, y_ev


def center_rgb(x):
    return (x - 128) / 128


def shuffle(x, y):
    i = np.random.permutation(len(y))
    return x[i], y[i]

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

def denormalize_box_cwh(image_hw, n_grid, norm_box_cwh, grid):
    image_h, image_w = image_hw 
    normalized_xc, normalized_yc, normalized_w, normalized_h = norm_box_cwh
    row, col = grid

    box_w = normalized_w * image_w
    box_h = normalized_h * image_h
    grid_w = 1. * image_w / n_grid
    grid_h = 1. * image_h / n_grid
    xc = normalized_xc * grid_w + col * grid_w
    yc = normalized_yc * grid_h + row * grid_h
    cwh = [xc, yc, box_w, box_h]
    return cwh