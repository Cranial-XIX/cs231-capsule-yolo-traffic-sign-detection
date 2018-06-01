import numpy as np
import cv2
import utils
import plot
import os
import torchvision.transforms as transforms
import torch
from loss_fns import cnn_loss, capsule_loss, dark_loss, darkcapsule_loss

def dark_pred(images, model, model_dir, restore, params, y_true = None, is_metric = False):
	""" Darknet prediction 
    
    Args:
    	images: list of images, of shape (n_images, ). Image size can be different.
    	model: darknet model.
    	model_dir: directory where weights are saved.
    	restore: "last" or "best"
    
    Return:
        If is metric: 
        	y_pred: of shape(n_images, n_grid, n_grid, 5B + C)
		If not metric:
			output_images: images with boxes, of shape (n_images, )
			output_crops: crops of boxes, of shape (n_boxes, capsule_input, capsule_input, 3)
			image_indices: (n_boxes, )
	"""
	restore_path = os.path.join(model_dir, restore + '.pth.tar')
	print("Restoring parameters from {}".format(restore_path))
	utils.load_checkpoint(restore_path, model, params)
		
	image_hw = np.array([image.shape[0:2] for image in images])
	input_hw = (params.darknet_input, params.darknet_input)
	transformer = transforms.Compose([transforms.ToPILImage(),
				    			      transforms.Resize(input_hw), 
			                          transforms.ToTensor()])
	x = torch.stack([transformer(image) for image in images])

	model.eval()
	x = x.to(device=params.device, dtype=torch.float32)
	y_pred = model(x)

	if is_metric:
		return y_pred

	if y_true is not None:
		y_true = torch.from_numpy(y_true).to(device=params.device)
		loss = dark_loss(y_pred, y_true, params)
		print("loss:", loss)

	y_pred = y_pred.data.numpy()
	image_indices, xy, classes = utils.y_to_boxes_vec(y_pred, image_hw, params.n_classes, conf_th = 0.5)
	output_images, crops_bch = plot.draw_boxes_vec(images, image_indices, xy, classes)

	capsule_input = (params.capsule_input, params.capsule_input)
	output_crops = np.array([cv2.resize(crop, capsule_input) for crops in crops_bch for crop in crops])
	return output_images, output_crops, image_indices