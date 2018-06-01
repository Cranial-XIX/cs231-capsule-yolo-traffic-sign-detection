import numpy as np
import cv2
import utils
import plot
import os
import torchvision.transforms as transforms
import torch
import config

def dark_pred(images, model, model_dir, params, restore_file, is_end = True ,is_metric = False):
    """ Darknet prediction 
    
    Args:
        - images: list of images, of shape (n_images, ). Image size can be different.
        - model: darknet model.
        - model_dir: directory where weights are saved.
        - restore_file: "last" or "best"
        - is_end: whether it is end to end or used in dark->cnn/capsule
        - is_metric: whether it is used in metric
    
    Return:
        If is metric: 
            - y_pred: of shape(n_images, n_grid, n_grid, 5B + C)
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
    transformer = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize(input_hw), 
                                      transforms.ToTensor()])
    x = torch.stack([transformer(image) for image in images])
    
    print("!!!!!!!!!!!change model to train for overfit!!!!!!!!!!")
    model.train()
    x = x.to(device=params.device, dtype=torch.float32)
    y_pred = model(x)

    if is_metric:
        return y_pred

    y_pred = y_pred.data.numpy()
    image_indices, boxes_xy, classes = utils.y_to_boxes_vec(y_pred, image_hw, params.n_classes, conf_th = 0.5)
    output_images, crops_bch = plot.draw_boxes_vec(images, image_indices, boxes_xy, classes)

    if is_end:
        return output_images

    capsule_input = (params.capsule_input, params.capsule_input)
    output_crops = np.array([cv2.resize(crop, capsule_input) for crops in crops_bch for crop in crops])
    return output_crops, image_indices, boxes_xy

def class_pred(x, model, model_dir, params, restore_file):
    restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
    print("Restoring parameters from {}".format(restore_path))
    utils.load_checkpoint(restore_path, model, params)

    print("!!!!!!!!!!!change model to train for overfit!!!!!!!!!!")
    model.train()
    x = torch.from_numpy(x).float().permute(0, 3, 1, 2).to(
            device=params.device)
    y_pred = model(x)

    y_pred = y_pred.data.numpy()
    classes = np.argmax(y_pred, axis = 1)
    return classes

def dark_class_pred(images, dark_model, class_model, dark_model_dir, class_model_dir,
    dark_params, class_params, restore_file):
    dark_crops, image_indices, boxes_xy = dark_pred(images, dark_model, dark_model_dir, dark_params, restore_file, is_end = False)
    classes = class_pred(dark_crops, class_model, class_model_dir, class_params, restore_file)
    output_images, _ = plot.draw_boxes_vec(images, image_indices, boxes_xy, classes)
    return output_images

