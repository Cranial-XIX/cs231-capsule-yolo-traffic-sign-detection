# model list
model_names = ['cnn', 'capsule', 'darknet_d', 'darknet_r', 'darkcapsule']

# data folder
GTSRB = 'data/GTSRB'
GTSDB = 'data/GTSDB'

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