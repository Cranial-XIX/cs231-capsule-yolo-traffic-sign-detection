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
    total = len(y)
    n_batch = (total + params.batch_size - 1) // params.batch_size
    x_split, y_split = np.array_split(x, n_batch), np.array_split(y, n_batch)

    prs = []
    accs = []
    aucs = []

    for fr in np.arange(0.1, 1.1, 0.1):
        model = model_nn(params).to(device=params.device)

        model_dir = './experiments/{}s/{}{:.1f}'.format(name, name, fr)
        restore_path = os.path.join(model_dir, 'best' + '.pth.tar')
        print("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, params)

        model.eval()

        y_hat = []
        with torch.no_grad():
            for i, (x_bch, y_bch) in enumerate(zip(x_split, y_split)):
                x_bch = torch.from_numpy(x_bch).float().permute(0, 3, 1, 2).to(
                device=params.device)

                y_hat_bc = model(x_bch)
                y_hat_bc = y_hat_bc.data.cpu().numpy()

                y_hat.append(y_hat_bc)

        y_hat = np.concatenate(y_hat, axis=0)

        pr = recog_pr(y, y_hat, params)
        acc = recog_acc(y, y_hat, params)
        auc = recog_auc(y, y_hat, params)
        prs.append(pr)
        accs.append(acc)
        aucs.append(auc)

        np.save('./experiments/{}/{}_prs'.format(name, name), prs)
        np.save('./experiments/{}/{}_accs'.format(name, name), accs)
        np.save('./experiments/{}/{}_aucs'.format(name, name), aucs)

    prs = np.array(prs)
    accs = np.array(accs)
    aucs = np.array(aucs)
    return prs, accs, aucs

cnn_prs, cnn_accs, cnn_aucs = plot_metrics('cnn', ConvNet)
cap_prs, cap_accs, cap_aucs = plot_metrics('capsule', CapsuleNet)

np.save('./experiments/cnn/cap_prs', cap_prs)
np.save('./experiments/cnn/cap_accs', cap_accs)
np.save('./experiments/cnn/cap_aucs', cap_aucs)
np.save('./experiments/cnn/cnn_prs', cnn_prs)
np.save('./experiments/cnn/cnn_accs', cnn_accs)
np.save('./experiments/cnn/cnn_aucs', cnn_aucs)

fr = np.arange(0.1, 1.1, 0.1)
plt.plot(fr, cnn_prs, label = 'cnn_prs')
plt.plot(fr, cnn_accs, label = 'cnn_accs')
plt.plot(fr, cnn_aucs, label = 'cnn_aucs')
plt.plot(fr, cap_prs, label = 'cap_prs')
plt.plot(fr, cap_accs, label = 'cap_accs')
plt.plot(fr, cap_aucs, label = 'cap_aucs')

plt.legend()
plt.show()






    