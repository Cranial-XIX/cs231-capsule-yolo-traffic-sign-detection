import numpy as np
import cv2
import utils
import plot
import os
import torchvision.transforms as transforms
import torch
from loss_fns import cnn_loss, capsule_loss, dark_loss, darkcapsule_loss

def dark_pred(images, model, model_dir, restore, params, y_true = None):
	restore_path = os.path.join(model_dir, restore + '.pth.tar')
	print("Restoring parameters from {}".format(restore_path))
	utils.load_checkpoint(restore_path, model, params)
		
	image_hw = np.array([image.shape[0:2] for image in images])
	input_hw = (params.darknet_input, params.darknet_input)
	transformer = transforms.Compose([transforms.ToPILImage(),
				    			      transforms.Resize(input_hw), 
			                          transforms.ToTensor()])
	x = torch.stack([transformer(image) for image in images])

	print("!!!!!!!!!!model train mode for overfit (need change)!!!!!!!!!!!!!!!!")
	
	model.train()
	x = x.to(device=params.device, dtype=torch.float32)
	y_pred = model(x)

	if y_true is not None:
		y_true = torch.from_numpy(y_true).to(device=params.device)
		loss = dark_loss(y_pred, y_true, params)
		print("loss:", loss)
	y_pred = y_pred.data.numpy()
	
	image_indices, xy, classes = utils.y_to_boxes_vec(y_pred, image_hw, params.n_classes, conf_th = 0.5)
	output_images, crops_bch = plot.draw_boxes_vec(images, image_indices, xy, classes)

	capsule_input = (params.capsule_input, params.capsule_input)

	output_crops = np.array([cv2.resize(crop, capsule_input) for crops in crops_bch for crop in crops])
	print(output_crops.shape)
	
	for i, img in enumerate(output_images):
		cv2.imshow('image ' + str(i), img)
	for i, crop in enumerate(output_crops):
		cv2.imshow('crop ' + str(i), crop)

	cv2.waitKey(0)
	return output_images