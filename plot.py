import config
import numpy as np
import cv2

def draw_boxes(image, xy, classes = None, filename = None):
    """ Plot boxes on one image.
        
    Args:
        - image: of shape (image_h, image_w, 3)
        - xy: the cooridnates (x1, y1, x2, y2) of boxes,
          of shape (num_boxes, 4)
        - classes: the class index of boxes,
          of shape (num_boxes, 1)
        - save_path: directory where results are saved

    Return:
        - image with boxes and class names
    """ 
    new_img = image.copy()
    names_file = config.GTSDB + '/class_names.txt'
    class_names = np.loadtxt(names_file, dtype = str, delimiter = '\n')
    crops = [image[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2 in xy]

    for i in range(xy.shape[0]):
        x1, y1, x2, y2 = xy[i].astype(int)
        cv2.rectangle(new_img, (x1, y1), (x2, y2), (255, 0, 0), 1)

        if classes is not None:
            c = int(classes[i])
            xc = (x1 + x2) // 2
            yc = (y1 + y2) // 2
            cv2.putText(new_img, class_names[c], (xc, yc), 
                0, 0.5, (255, 0, 0))
    
    if filename is not None:
        cv2.imwrite(filename, new_img)

    return new_img, crops

def draw_boxes_vec(images, image_indices, xy, classes, save_dir = None, batch_name = ""):
    """ Plot boxes on images (for a batch of images).
    
    Args:
        - images: of shape (batch_size, image_h, image_w, 3)
        - image_indices: the index of image for each box (num_boxes, )
        - xy: the cooridnates (x1, y1, x2, y2) of boxes,
          of shape (num_boxes, 4)
        - classes: the class index of boxes,
          of shape (num_boxes, 1)
        - save_path: directory where results are saved
    """
    new_images = []
    crops_bch = []
    for i in range(images.shape[0]):
        mask = (image_indices == i)
        if save_dir is not None:
            filename = save_dir + '/' + batch_name + "_" + str(i) 
        else:
            filename = None
            
        if classes is not None:
            new_img, crops_img = draw_boxes(images[i], xy[mask], classes[mask], filename)
        else:
            new_img, crops_img = draw_boxes(images[i], xy[mask], None, filename)

        new_images.append(new_img)
        crops_bch.append(crops_img)
        
    return new_images, crops_bch