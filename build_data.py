import config
import csv
import cv2
import numpy as np
import pickle
import utils

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


def gtsdb(root=config.GTSDB):
    pass


if __name__ == "__main__":
    np.random.seed(0)
    gtsrb()