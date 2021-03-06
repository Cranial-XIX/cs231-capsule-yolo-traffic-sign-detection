import numpy as np
import cv2
import utils
import plot
import os
import torchvision.transforms as transforms
import torch
import config

def dark_pred(images, model, model_dir, params, restore_file, is_end = True, conf_th = 0.5, y = None):
    """ Darknet prediction 
    
    Args:
        - images: list of images, of shape (n_images, ). Image size can be different.
        - model: darknet model.
        - model_dir: directory where weights are saved.
        - restore_file: "last" or "best"
        - is_end: whether it is end to end or used in dark->cnn/capsule
    
    Return:
        If is metric: 
            - y_hat: of shape(n_images, n_grid, n_grid, 5B + C)
        If is end to end:
            - output_images: images with boxes and classes, of shape (n_images, )
        If is used in dark-cnn/capsule:
            - output_crops: crops of boxes, of shape (n_boxes, capsule_input, capsule_input, 3)
            - image_indices: the index of image for each box 
              of shape (num_boxes,)
            - boxes_xy: the cooridnates (x1, y1, x2, y2) of boxes,
              of shape (num_boxes, 4)
    """
    restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
    print("Restoring parameters from {}".format(restore_path))
    utils.load_checkpoint(restore_path, model, params)
        
    image_hw = np.array([image.shape[0:2] for image in images])
    input_hw = (params.darknet_input, params.darknet_input)
    x = torch.Tensor([cv2.resize(image, input_hw) for image in images])

    model.eval()
    with torch.no_grad():
        x = x.permute(0, 3, 1, 2).to(device=params.device, dtype=torch.float32)
        y_hat = model(x)

    y_hat = y_hat.data.cpu().numpy()
    image_indices, boxes_xy, classes = utils.y_to_boxes_vec(y_hat, params, image_hw = image_hw, conf_th = conf_th)
    output_images, crops_bch = plot.draw_boxes_vec(images, image_indices, boxes_xy, classes)

    if y is not None:
        true_indices, true_xy, true_classes = utils.y_to_boxes_vec(y, params, image_hw = image_hw, conf_th = conf_th)
        output_images, _ = plot.draw_boxes_vec(output_images, true_indices, true_xy, true_classes, color=(0,0,255))

    if is_end:
        return y_hat, output_images

    capsule_input = (params.capsule_input, params.capsule_input)
    output_crops = np.array([cv2.resize(crop, capsule_input) for crops in crops_bch for crop in crops])
    return y_hat, output_crops, image_indices, boxes_xy

def class_pred(x, model, model_dir, params, restore_file):
    restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
    print("Restoring parameters from {}".format(restore_path))
    utils.load_checkpoint(restore_path, model, params)
    
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(x).float().permute(0, 3, 1, 2).to(
                device=params.device)
        y_hat = model(x)

    y_hat = y_hat.data.cpu().numpy()
    classes = np.argmax(y_hat, axis = 1)
    return y_hat, classes

def dark_class_pred(images, dark_model, dark_model_dir, dark_params, class_model, class_model_dir,
    class_params, restore_file):
    dark_y_hat, dark_crops, image_indices, boxes_xy = dark_pred(images, dark_model, dark_model_dir, dark_params, restore_file, is_end = False)
    dark_crops = utils.center_rgb(dark_crops)
    class_y_hat, classes = class_pred(dark_crops, class_model, class_model_dir, class_params, restore_file)
    output_images, _ = plot.draw_boxes_vec(images, image_indices, boxes_xy, classes)
    y_hat = utils.combine_y_hat(images, dark_y_hat, class_y_hat, image_indices, boxes_xy, dark_params)
    return y_hat, output_images

