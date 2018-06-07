import numpy as np
import os
import pickle
import sys
import torch
import torch.nn.functional as F
import utils
import cv2
from matplotlib import pyplot as plt

from metrics import recog_acc, recog_auc, recog_pr, detect_AP, detect_and_recog_mAP, detect_acc, darkcapsule_acc, detect_and_recog_acc
from models import ConvNet, CapsuleNet, DarkNet, DarkCapsuleNet
from predict_fns import dark_pred, class_pred, dark_class_pred


def load_params(model_dir):
    json_path = os.path.join(model_dir, 'params.json')
    params = utils.Params(json_path)
    params.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return params

def show(a, b):
    cv2.imshow('ori', (a+1)/2)
    cv2.imshow('recon', (b+1)/2)
    cv2.waitKey(0)

param_dir = './experiments/capsule'

params = load_params(param_dir)
x, y = pickle.load(open('./data/GTSRB/train_small.p', 'rb'))

i = 0
xxx = x[i]
yy = y[i]


xx = torch.from_numpy(xxx).float().unsqueeze(0).permute(0, 3, 1, 2).to(device=params.device)
yy = torch.from_numpy(np.array(yy).reshape(1,)).to(device=params.device)


model = CapsuleNet(params).to(device=params.device)
model_dir = './experiments/capsule1.0'
restore_path = os.path.join(model_dir, 'best.pth.tar')
print("Restoring parameters from {}".format(restore_path))
utils.load_checkpoint(restore_path, model, params)
model.eval()


x = F.relu(model.conv1(xx))
x = model.primary_capsules(x)
x = model.traffic_sign_capsules(x).squeeze()

t = torch.gather(x.unsqueeze(0), 1, yy.repeat(16, 1).t().unsqueeze(1)).squeeze()
decoded = model.decoder(t)
decoded = decoded.permute(0, 2, 3, 1).squeeze()

show(xxx, decoded.data.numpy())

    