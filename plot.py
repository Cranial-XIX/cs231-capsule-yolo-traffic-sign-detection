import config
import numpy as np
import cv2

def draw_boxes(image, xy, classes = None):
    """ Plot boxes on one image.
        
    Args:
        - image: of shape (image_h, image_w, 3)
        - xy: the cooridnates (x1, y1, x2, y2) of boxes,
          of shape (num_boxes, 4)
        - classes: the class index of boxes,
          of shape (num_boxes, 1) or None

    Return:
        - image with boxes and class names
    """ 
    names_file = config.GTSDB + '/class_names.txt'
    class_names = np.loadtxt(names_file, dtype = str, delimiter = '\n')
    
    new_img = image.copy()
    crops = [image[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2 in xy]

    for i in range(xy.shape[0]):
        x1, y1, x2, y2 = xy[i].astype(int)
        cv2.rectangle(new_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        if classes is not None:
            c = int(classes[i])
            xc = (x1 + x2) // 2
            yc = (y1 + y2) // 2
            cv2.putText(new_img, class_names[c], (xc, yc), 
                0, 0.5, (0, 255, 0))

    return new_img, crops

def draw_boxes_vec(images, image_indices, xy, classes = None):
    """ Plot boxes on images (for a batch of images).
    
    Args:
        - images: a list of images
        - image_indices: the index of image for each box (num_boxes, )
        - xy: the cooridnates (x1, y1, x2, y2) of boxes,
          of shape (num_boxes, 4)
        - classes: the class index of boxes,
          of shape (num_boxes, 1) or None

    Return:
        - images with boxes and class names
    """
    new_images = []
    crops_bch = []
    for i in range(len(images)):
        mask = (image_indices == i)
            
        if classes is not None:
            new_img, crops_img = draw_boxes(images[i], xy[mask], classes[mask])
        else:
            new_img, crops_img = draw_boxes(images[i], xy[mask])

        new_images.append(new_img)
        crops_bch.append(crops_img)
        
    return new_images, crops_bch