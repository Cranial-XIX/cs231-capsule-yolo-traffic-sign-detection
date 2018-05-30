import config
import numpy as np
import cv2

def draw_boxes(image, xy, classes, filename = None):
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
    names_file = config.GTSDB + '/class_names.txt'
    class_names = np.loadtxt(names_file, dtype = str, delimiter = '\n')
    for i in range(xy.shape[0]):
        x1, y1, x2, y2 = xy[i].astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        if classes is not None:
            c = int(classes[i])
            xc = (x1 + x2) // 2
            yc = (y1 + y2) // 2
            cv2.putText(image, class_names[c], (xc, yc), 
                cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 0))
    
    if filename is not None:
        cv2.imwrite(filename, image)

    return image

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
    for i in range(images.shape[0]):
        mask = (image_indices == i)
        if save_dir is not None:
            filename = save_dir + '/' + batch_name + "_" + str(i) 
        else:
            filename = None
        new_img = draw_boxes(images[i], xy[mask], classes[mask], filename)
        new_images.append(new_img)
    new_images = np.array(new_images)
    return new_images


