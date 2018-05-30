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
        resized_image = cv2.resize(image, (params.image_resize, params.image_resize))
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

if __name__ == "__main__":
    np.random.seed(0)
    # gtsrb()
    params = utils.Params('./experiments/darknet_r/params.json')
    gtsdb(params)
