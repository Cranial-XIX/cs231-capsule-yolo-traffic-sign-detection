import argparse
import config
import cv2
import numpy as np
import os
import pickle
import sys
import time
import torch
import utils

from metrics import recog_auc, recog_pr, detect_AP, detect_and_recog_mAP
from models import ConvNet, CapsuleNet, DarkNet, DarkCapsuleNet
from loss_fns import cnn_loss, capsule_loss, dark_loss, darkcapsule_loss
from predict_fns import dark_pred, class_pred, dark_class_pred
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torchsummary import summary
from tqdm import trange, tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='cnn', help=' | '.join(config.model_names))
parser.add_argument('--mode', default='train', help='train | predict | overfit')
parser.add_argument('--summary', default=True, help='if summarize model', action='store_true')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--restore', default='last', help="last | best")
parser.add_argument('--combine', default=None, help="darknet_r | darknet_d")
parser.add_argument('--recon', default=False, help='if use reconstruction loss', action='store_true')
parser.add_argument('--recon_coef', default=5e-4, help='reconstruction coefficient')


def train(x, y, model, optimizer, loss_fn, metric, params):
    model.train()

    x, y = utils.shuffle(x, y)
    total = len(y)
    n_batch = (total + params.batch_size - 1) // params.batch_size
    x_split, y_split = np.array_split(x, n_batch), np.array_split(y, n_batch)
    n = y.shape[0]
    t = trange(n_batch)
    avg_loss = 0
    y_hat = []

    for i, (x_bch, y_bch) in enumerate(zip(x_split, y_split)):
        x_bch = utils.augmentation(x_bch)
        x_bch = torch.from_numpy(x_bch).float().permute(0, 3, 1, 2).to(
            device=params.device)
        y_bch = torch.from_numpy(y_bch).to(device=params.device)

        if params.model == 'capsule' and params.recon:
            y_hat_bch, recon = model(x_bch, y_bch, True)
            loss = loss_fn(y_hat_bch, y_bch, params, x_bch, recon)
        else:
            y_hat_bch = model(x_bch)
            loss = loss_fn(y_hat_bch, y_bch, params)

        y_hat.append(y_hat_bch.data.numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t.set_postfix(loss='{:05.3f}'.format(loss.item()))
        t.update()

        avg_loss += loss.item() / n

    y_hat = np.concatenate(y_hat, axis=0)

    # shrink size for faster calculation of metric
    if n > config.max_metric_samples:
        i = np.random.choice(n, config.max_metric_samples)
        y, y_hat = y[i], y_hat[i]

    metric_score = metric(y, y_hat, params)

    return avg_loss, metric_score


def evaluate(x, y, model, loss_fn, metric, params):
    model.eval()

    total = len(y)
    n_batch = (total + params.batch_size - 1) // params.batch_size
    x_split, y_split = np.array_split(x, n_batch), np.array_split(y, n_batch)
    n = y.shape[0]
    avg_loss = 0
    y_hat = []

    with torch.no_grad():
        for i, (x_bch, y_bch) in enumerate(zip(x_split, y_split)):
            x_bch = torch.from_numpy(x_bch).float().permute(0, 3, 1, 2).to(
                device=params.device)
            y_bch = torch.from_numpy(y_bch).to(device=params.device)

            if params.model == 'capsule' and params.recon:
                y_hat_bch, recon = model(x_bch, y_bch, True)
                loss = loss_fn(y_hat_bch, y_bch, params, x_bch, recon)
            else:
                y_hat_bch = model(x_bch)
                loss = loss_fn(y_hat_bch, y_bch, params)

            y_hat.append(y_hat_bch.data.numpy())
            avg_loss += loss / n

    # shrink size for faster calculation of metric
    if n > config.max_metric_samples:
        i = np.random.choice(n, config.max_metric_samples)
        y, y_hat = y[i], y_hat[i]

    y_hat = np.concatenate(y_hat, axis=0)
    metric_score = metric(y, y_hat, params)

    return avg_loss, metric_score


def train_and_evaluate(model, optimizer, loss_fn, metric, params,
    data_dir, model_dir, is_small=False, restore_file=None):
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        print("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, params, optimizer)

    losses_tr = []
    losses_ev = []
    metrics_tr = []
    metrics_ev = []
    best_loss_ev = float('inf')

    x_tr, y_tr, x_ev, y_ev = utils.load_data(data_dir, is_small)

    for epoch in range(params.n_epochs):
        loss_tr, metric_tr = train(
            x_tr, y_tr, model, optimizer, loss_fn, metric, params)
        loss_ev, metric_ev = evaluate(
            x_ev, y_ev, model, loss_fn, metric, params)

        params.writer.add_scalar('train_loss', loss_tr, epoch)
        params.writer.add_scalar('eval_loss', loss_ev, epoch)
        params.writer.add_scalar('train_metric', metric_tr, epoch)
        params.writer.add_scalar('eval_metric', metric_ev, epoch)

        is_best = loss_ev < best_loss_ev

        utils.save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_dict' : optimizer.state_dict()
            }, 
            is_best=is_best,
            checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            best_loss_ev = loss_ev

        tqdm.write(
            "epoch {} | train loss: {:05.3f} | eval loss: {:05.3f} |" \
            " best eval loss: {:05.3f} | " \
            "train metric: {:05.3f} | eval metric: {:05.3f}".format(
                epoch+1, loss_tr, loss_ev,
                best_loss_ev, metric_tr, metric_ev))

        losses_tr.append(loss_tr)
        losses_ev.append(loss_ev)
        metrics_tr.append(metric_tr)
        metrics_ev.append(metric_ev)

        np.save(os.path.join(model_dir, 'losses_tr'), losses_tr)
        np.save(os.path.join(model_dir, 'losses_ev'), losses_ev)
        np.save(os.path.join(model_dir, 'metrics_tr'), metrics_tr)
        np.save(os.path.join(model_dir, 'metrics_ev'), metrics_ev)

    params.writer.close()


def get_data_and_model_dir(model_name):
    if not model_name in config.model_names:
        print("Did not recognize model, choose from: ", *config.model_names)
        sys.exit()
    return config.data_dir[model_name], config.model_dir[model_name]


def load_params(model_dir, args):
    json_path = os.path.join(model_dir, 'params.json')
    params = utils.Params(json_path)

    params.device = "cuda" if torch.cuda.is_available() else "cpu"
    params.seed = args.seed
    params.dropout = args.dropout

    params.writer = SummaryWriter()
    return params


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir, model_dir = get_data_and_model_dir(args.model)
    params = load_params(model_dir, args)
    params.model = args.model
    params.recon = args.recon
    params.recon_coef = args.recon_coef

    # set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if params.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    model_loss_predict = {
        'cnn'             : (ConvNet, cnn_loss, class_pred, recog_auc),
        'capsule'         : (CapsuleNet, capsule_loss, class_pred, recog_auc),
        'darknet_d'       : (DarkNet, dark_loss, dark_pred, detect_AP),
        'darknet_r'       : (DarkNet, dark_loss, dark_pred, detect_and_recog_mAP),
        'darkcapsule'     : (DarkCapsuleNet, darkcapsule_loss, None, detect_and_recog_mAP),
    }

    model, loss_fn, predict_fn, metric = model_loss_predict[args.model]
    model = model(params).to(device=params.device)

    if args.summary:
        summary(model, config.input_shape[args.model])

    optimizer = Adam(model.parameters(), lr=args.lr)

    if args.mode == 'train':
        train_and_evaluate(model, optimizer, loss_fn, metric, params,
            data_dir, model_dir)
        
    if args.mode == 'overfit':
        utils.make_small_data(data_dir, 1)
        train_and_evaluate(
            model, optimizer, loss_fn, metric, params,
            data_dir, model_dir, is_small=True)

    if args.mode == 'predict':
        x_tr, y_tr, x_ev, y_ev = utils.load_data(data_dir)
        x = x_tr[0:2]
        y = y_tr[0:2]

        if args.combine is None:
            y_hat, output = predict_fn(x, model, model_dir, params, args.restore)
            print(y.shape, y_hat.shape)
            pickle.dump((y, y_hat), open('./debug/{}.p'.format(args.model), 'wb'))
        else:
            if args.model not in ('darknet_d', 'darknet_r') or \
            args.combine not in ('cnn', 'capsule'):
                print("Invalid combine")
                sys.exit()

            class_model_dir = get_data_and_model_dir(args.combine)[1]
            class_params = load_params(class_model_dir, args)
            class_model = model_loss_predict[args.combine][0]
            class_model = class_model(class_params) \
                          .to(device=class_params.device)

            dark_y_hat, class_y_hat, output = dark_class_pred(x, model, model_dir, params, 
                class_model, class_model_dir, class_params, args.restore)

            pickle.dump((y, dark_y_hat, class_y_hat), open('./debug/{}-{}.p'.format(args.model, args.combine), 'wb'))

        if args.model in ('darknet_d', 'darknet_r'):
            for i, image in enumerate(output):
                cv2.imshow(str(i), image)
            cv2.waitKey(0)