# model list
model_names = ['cnn', 'capsule', 'darknet_d', 'darknet_r', 'darkcapsule']

# data folder
GTSRB = 'data/GTSRB'
GTSDB = 'data/GTSDB'

# data name
tr_d = '/train.p'
ev_d = '/eval.p'
te_d = '/test.p'

tr_sm_d = '/train_small.p'
ev_sm_d = '/eval_small.p'
te_sm_d = '/test_small.p'

# data directory
data_dir = {
    'cnn'             : GTSRB,
    'capsule'         : GTSRB,
    'darknet_d'       : GTSDB,
    'darknet_r'       : GTSDB,
    'darkcapsule'     : GTSDB,
}

# model directory
model_dir = {
    'cnn'             : 'experiments/cnn',
    'capsule'         : 'experiments/capsule',
    'darknet_d'       : 'experiments/darknet_d',
    'darknet_r'       : 'experiments/darknet_r',
    'darkcapsule'     : 'experiments/darkcapsule',
}

# input shape
input_shape = {
    'cnn'             : (3, 32, 32),
    'capsule'         : (3, 32, 32),
    'darknet_d'       : (3, 224, 224),
    'darknet_r'       : (3, 224, 224),
    'darkcapsule'     : (3, 224, 224),
}

colors = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'
]