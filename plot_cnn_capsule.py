import numpy as np
import os
import pickle
import sys
import torch
import utils
from matplotlib import pyplot as plt

from metrics import recog_acc, recog_auc, recog_pr, detect_AP, detect_and_recog_mAP, detect_acc, darkcapsule_acc, detect_and_recog_acc
from models import ConvNet, CapsuleNet, DarkNet, DarkCapsuleNet
from predict_fns import dark_pred, class_pred, dark_class_pred


def load_params(model_dir):
    json_path = os.path.join(model_dir, 'params.json')
    params = utils.Params(json_path)
    params.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return params

def plot_metrics(name, model_nn):
    param_dir = './experiments/' + name
    data_dir = './data/GTSRB' 
    params = load_params(param_dir)
    x, y = pickle.load(open(data_dir + '/test.p', 'rb'))
    prs = []
    accs = []
    aucs = []

    for fr in np.arange(0.1, 1.1, 0.1):
        model = model_nn(params).to(device=params.device)
        model_dir = './experiments/{}s/{}{:.1f}'.format(name, name, fr)
        y_hat, output = class_pred(x, model, model_dir, params, 'best')
        pr = recog_pr(y, y_hat, params)
        acc = recog_acc(y, y_hat, params)
        auc = recog_auc(y, y_hat, params)
        prs.append(pr)
        accs.append(acc)
        aucs.append(auc)
    prs = np.array(prs)
    accs = np.array(accs)
    aucs = np.array(aucs)
    return prs, accs, aucs

cnn_prs, cnn_accs, cnn_aucs = plot_metrics('cnn', ConvNet)
cap_prs, cap_accs, cap_aucs = plot_metrics('capsule', CapsuleNet)
fr = np.arange(0.1, 1.1, 0.1)

plt.plot(fr, cnn_prs, label = 'cnn_prs')
plt.plot(fr, cnn_accs, label = 'cnn_accs')
plt.plot(fr, cnn_aucs, label = 'cnn_aucs')
plt.plot(fr, cap_prs, label = 'cap_prs')
plt.plot(fr, cap_accs, label = 'cap_accs')
plt.plot(fr, cap_aucs, label = 'cap_aucs')

np.save('./experiments/cnn/cnn_prs', cnn_prs)
np.save('./experiments/cnn/cnn_accs', cnn_accs)
np.save('./experiments/cnn/cnn_aucs', cnn_aucs)
np.save('./experiments/cnn/cap_prs', cap_prs)
np.save('./experiments/cnn/cap_accs', cap_accs)
np.save('./experiments/cnn/cap_aucs', cap_aucs)
plt.legend()
plt.show()






    