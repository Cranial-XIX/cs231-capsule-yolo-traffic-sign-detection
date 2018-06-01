import config
import csv
import cv2
import numpy as np
import pickle
import utils
import os

from tqdm import trange

def gtsrb(root=config.GTSRB):
    x_tr, y_tr, x_ev, y_ev, x_te, y_te = [], [], [], [], [], []
    classes = np.arange(0, 43)
    for c in trange(43):
        class_name = format(classes[c], '05d')
        prefix = root + '/Images/' + class_name + '/'
        f = open(prefix + 'GT-' + class_name + '.csv')
        reader = csv.reader(f, delimiter=';')
        next(reader, None)
        x, y = [], []
        for row in reader:
            im = cv2.imread(prefix + row[0])
            im = im[np.int(row[4]):np.int(row[6]), 
                    np.int(row[3]):np.int(row[5]), :]
            x.append(im)
            y.append(c)
        split = len(y) // 10
        x, y = utils.shuffle(np.array(x), np.array(y))
        x, y = x.tolist(), y.tolist()
        x_ev += x[:split]
        y_ev += y[:split]
        x_te += x[split:2*split]
        y_te += y[split:2*split]
        x_tr += x[2*split:]
        y_tr += y[2*split:]
        f.close()

    size = (32, 32)
    x_tr = [cv2.resize(x, size) for x in x_tr]
    x_ev = [cv2.resize(x, size) for x in x_ev]
    x_te = [cv2.resize(x, size) for x in x_te]

    x_tr, y_tr = np.array(x_tr).astype(np.float32), np.array(y_tr)
    x_ev, y_ev = np.array(x_ev).astype(np.float32), np.array(y_ev)
    x_te, y_te = np.array(x_te).astype(np.float32), np.array(y_te)

    x_tr, x_ev, x_te = list(map(utils.center_rgb, [x_tr, x_ev, x_te]))

    x_tr, y_tr = utils.shuffle(x_tr, y_tr)
    x_ev, y_ev = utils.shuffle(x_ev, y_ev)
    x_te, y_te = utils.shuffle(x_ev, y_te)

    pickle.dump((x_tr, y_tr), open(root+'/train.p', 'wb'))
    pickle.dump((x_ev, y_ev), open(root+'/eval.p', 'wb'))
    pickle.dump((x_te, y_te), open(root+'/test.p', 'wb'))


def gtsdb(params, root=config.GTSDB):
    data_dir = root + '/raw_GTSDB'
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.ppm')]
    data_size = len(image_files)
    raw_data = np.loadtxt(data_dir + '/gt.txt', delimiter = ';', dtype= str)
    
    image_names = raw_data[:, 0]
    box_coords = raw_data[:, 1:5].astype(float)
    classes = raw_data[:, 5].astype(int)

    X, Y = [], []
    conflict_count = 0

    for i in trange(data_size):
        name = image_files[i]
        image = cv2.imread(os.path.join(data_dir, name))
        resized_image = cv2.resize(image, (params.darknet_input, params.darknet_input))
        X.append(resized_image)

        # Load bounding boxes
        y = np.zeros((params.n_grid, params.n_grid, 5 + params.n_classes))
        orig_hw = image.shape[0:2]
        resized_hw = resized_image.shape[0:2]
        indices = np.argwhere(image_names == name).reshape(-1,)

        for index in indices:
            box_xy = box_coords[index]
            resized_box_xy = utils.resize_box_xy(orig_hw, resized_hw, box_xy)
            box_cwh = utils.xy_to_cwh(resized_box_xy)
            (xc, yc, w, h), (row, col) = utils.normalize_box_cwh(resized_hw, params.n_grid, box_cwh)
            
            # skip if the grid cell has object
            if y[row, col, 0] == 1:
                conflict_count += 1
                continue

            y[row, col, 0:5] = [1, xc, yc, w, h]
            if params.n_classes != 0:
                c = classes[index]
                y[row, col, 5 + c] = 1

        Y.append(y)

    X, Y = np.array(X), np.array(Y)
    X, Y = utils.shuffle(X, Y)

    split = data_size // 10
    X_ev = X[:split]
    Y_ev = Y[:split]
    X_te = X[split:2*split]
    Y_te = Y[split:2*split]
    X_tr = X[2*split:]
    Y_tr = Y[2*split:]
    X_small = X[0:2]
    Y_small = Y[0:2]

    pickle.dump((X_tr, Y_tr), open(root+'/train.p', 'wb'))
    pickle.dump((X_ev, Y_ev), open(root+'/eval.p', 'wb'))
    pickle.dump((X_te, Y_te), open(root+'/test.p', 'wb'))
    pickle.dump((X_small, Y_small), open(root+'/small.p', 'wb'))
    
    # Get names for each class
    class_names = np.loadtxt(data_dir+'/Readme.txt', skiprows=39, delimiter = '\n', dtype = str)
    for i, name in enumerate(class_names):
        class_names[i] = name.split('=')[1]
    np.savetxt(root+'/class_names.txt', class_names, delimiter='\n', fmt='%s')

    print('Build dataset done.')
    print('Train shape:', X_tr.shape, Y_tr.shape)
    print('Val shape:', X_ev.shape, Y_ev.shape)
    print('Test shape:', X_te.shape, Y_te.shape)
    print('Number of boxes:', box_coords.shape[0])
    print('Conflict count:', conflict_count)

def gtsdb_aug(params, root = 'data', output_size=1000):

    # directory path
    data_dir = root + '/GTSDB/raw_GTSDB/'
    class_dir = root + '/GTSRB/Images/'
    

    # files from raw_GTSDB
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.ppm')]
    data_size = len(image_files)
    raw_data = np.loadtxt(data_dir + '/gt.txt', delimiter = ';', dtype= str)
    image_names = raw_data[:, 0]
    box_coords = raw_data[:, 1:5].astype(float)
    classes = raw_data[:, 5].astype(int)

    # some pre-defined parameters
    add_signs = params.add_signs
    resized_hw = [params.darknet_input, params.darknet_input]

    # two data loaders
    X, Y = [], []

    for itr_image in trange(output_size):

        # randomly select one image
        orig_image_name = random.choice(os.listdir(data_dir))
        while "ppm" not in orig_image_name: 
            orig_image_name = random.choice(os.listdir(data_dir))
        image = cv2.imread(os.path.join(data_dir, orig_image_name))

        # extract the existing signs' bounding boxes
        signs_list = {}
        indices = np.argwhere(image_names == orig_image_name).reshape(-1,)
        num_orign_signs = len(indices)
        box_xy = box_coords[indices]

        # occlude the existing and paste "add_signs" new signs
        idx = 0
        max_signs = num_orign_signs + add_signs
        num_signs = np.random.randint(num_orign_signs, max_signs+1)

        # randomly select num_signs
        for itr_sign in range(num_signs):

            # a class from 43 classes
            class_name = random.choice(os.listdir(class_dir))
            while "0" not in class_name: 
                class_name = random.choice(os.listdir(class_dir))

            # a sign from that class
            sign_name = random.choice(os.listdir(class_dir + class_name + '/'))
            while "ppm" not in sign_name: 
                sign_name = random.choice(os.listdir(class_dir + class_name + '/'))
            
            # load sign bounding boxes
            data_signs = np.loadtxt(class_dir +  class_name + '/GT-' + class_name \
                + '.csv', delimiter = ';', dtype= str)[1:]
            
            selected = np.argwhere(data_signs == sign_name)[0][0]

            box_coords_data = data_signs[selected, 1:8].astype(int)
            #   key    height width startX startY endX endY  class
            # "name"    0      1      2       3     4     5    6

            signs_list[str(data_signs[selected,0])] = box_coords_data

        # array y records new bounding boxes recording 
        y = np.zeros((params.n_grid, params.n_grid, 5 + params.n_classes))
        
        # perform data augmentation
        for key in signs_list:

            # one sign's info
            single_sign = signs_list[key]
            class_name = utils.get_image_name(single_sign[6])[:-4]
            sign = cv2.imread(class_dir +  '/' + class_name + '/' + key)

            # FROM this sign
            fromX1, fromY1, fromX2, fromY2 = single_sign[2:6]
            fromH, fromW = fromY2 - fromY1, fromX2 - fromX1

            # 1. occlude existing signs
            if idx < num_orign_signs:

                # TO the selected image
                toX1, toY1, toX2, toY2 = box_xy[idx].astype(int)
                toH, toW= toY2 - toY1, toX2 - toX1 
                ratioH, ratioW  = toH / fromH, toW/fromW
                rescaleH, rescaleW = int(ratioH*fromH), int(ratioW*fromW) 

                # resize the selected sign to fit into the space of existing signs              
                resized_single_sign = \
                    cv2.resize(sign[fromY1:fromY2, fromX1:fromX2], (toW, toH))

                # paste
                image[toY1:toY2, toX1:toX2] = resized_single_sign
                
                # record the new bounding boxes
                # Note: box_xy is the same as the existing signs
                new_box_xy = box_xy[idx].astype(int)
                resized_box_xy = \
                    utils.resize_box_xy(image.shape[0:2], resized_hw, new_box_xy)
                box_cwh = utils.xy_to_cwh(resized_box_xy)
                (xc, yc, w, h), (row, col) = \
                    utils.normalize_box_cwh(resized_hw, params.n_grid, box_cwh)
                y[row, col, 0:5] = [1, xc, yc, w, h]
                y[row, col, 5 + single_sign[6]] = 1
                
                idx += 1
            
            # 2. add new signs
            else:

                # TO a random position of the selected image
                X_start = np.random.randint(0, image.shape[1] -  single_sign[0])
                Y_start = np.random.randint(0, image.shape[0] -  single_sign[1])
                toX1, toY1, toX2, toY2 = X_start, Y_start, X_start+fromW, Y_start+fromH
                toH, toH = toY2 - toY1, toX2 - toX1

                # paste
                image[toY1:toY2, toX1:toX2] = sign[fromY1:fromY2, fromX1:fromX2]
               
                # record the new bounding boxes
                # Note: box_xy is the newly defined bounding boxes
                new_box_xy = [toX1, toY1, toX2, toY2]
                resized_box_xy = \
                    utils.resize_box_xy(image.shape[0:2], resized_hw, new_box_xy)
                box_cwh = utils.xy_to_cwh(resized_box_xy)
                (xc, yc, w, h), (row, col) = \
                    utils.normalize_box_cwh(resized_hw, params.n_grid, box_cwh)
                y[row, col, 0:5] = [1, xc, yc, w, h]
                y[row, col, 5 + single_sign[6]] = 1

        resized_image = cv2.resize(image, (params.darknet_input, params.darknet_input))
        Y.append(y)
        X.append(resized_image)

    X, Y = np.array(X), np.array(Y)
    
    print('data augmentation is done')
    print('X shape:')
    print(X.shape)
    print('Y shape:')
    print(Y.shape)

    # copied from gtsdb()
    X, Y = utils.shuffle(X, Y)
    split = X.shape[0] // 10
    X_ev = X[:split]
    Y_ev = Y[:split]
    X_te = X[split:2*split]
    Y_te = Y[split:2*split]
    X_tr = X[2*split:]
    Y_tr = Y[2*split:]

    pickle.dump((X_tr, Y_tr), open(root+'/GTSDB/train.p', 'wb'))
    pickle.dump((X_ev, Y_ev), open(root+'/GTSDB/eval.p', 'wb'))
    pickle.dump((X_te, Y_te), open(root+'/GTSDB/test.p', 'wb'))


if __name__ == "__main__":
    np.random.seed(0)
    gtsrb()
    params = utils.Params('./experiments/darknet_r/params.json')
    gtsdb(params)
