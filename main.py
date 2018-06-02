import argparse
import config
import numpy as np
import os
import sys
import time
import torch
import tqdm
import utils
import cv2
import pickle

from models import ConvNet, CapsuleNet, DarkNet, DarkCapsuleNet
from loss_fns import cnn_loss, capsule_loss, dark_loss, darkcapsule_loss
from predict_fns import dark_pred, class_pred, dark_class_pred
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torchsummary import summary
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='cnn', help=' | '.join(config.model_names))
parser.add_argument('--mode', default='train', help='train | predict | overfit')
parser.add_argument('--summary', default=True, help='if summarize model', action='store_true')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--restore', default='last', help="last | best")
parser.add_argument('--combine', default=None, help="darknet_r | darknet_d")

def train(x, y, model, optimizer, loss_fn, params):
    model.train()

    x, y = utils.shuffle(x, y)
    total = len(y)
    n_batch = (total + params.batch_size - 1) // params.batch_size
    x_split, y_split = np.array_split(x, n_batch), np.array_split(y, n_batch)
    n = y.shape[0]
    t = trange(n_batch)
    avg_loss = 0

    for i, (x_bch, y_bch) in enumerate(zip(x_split, y_split)):
        x_bch = torch.from_numpy(x_bch).float().permute(0, 3, 1, 2).to(
            device=params.device)
        y_bch = torch.from_numpy(y_bch).to(device=params.device)

        y_hat_bch = model(x_bch)
        loss = loss_fn(y_hat_bch, y_bch, params)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t.set_postfix(loss='{:05.3f}'.format(loss.item()))
        t.update()

        avg_loss += loss.item() / n

    return avg_loss


def evaluate(x, y, model, loss_fn, params):
    model.eval()

    total = len(y)
    n_batch = (total + params.batch_size - 1) // params.batch_size
    x_split, y_split = np.array_split(x, n_batch), np.array_split(y, n_batch)
    n = y.shape[0]
    avg_loss = 0

    with torch.no_grad():
        for i, (x_bch, y_bch) in enumerate(zip(x_split, y_split)):
            x_bch = torch.from_numpy(x_bch).float().permute(0, 3, 1, 2).to(
                device=params.device)
            y_bch = torch.from_numpy(y_bch).to(device=params.device)

            y_hat_bch = model(x_bch)
            loss = loss_fn(y_hat_bch, y_bch, params)
            avg_loss += loss / n

    return avg_loss

def train_and_evaluate(model, optimizer, loss_fn, params,
    data_dir, model_dir, is_small=False, restore_file=None):
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        print("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, params, optimizer)

    losses_tr = []
    losses_ev = []
    best_loss_ev = float('inf')

    x_tr, y_tr, x_ev, y_ev = utils.load_data(data_dir, is_small)

    for epoch in range(params.n_epochs):
        loss_tr = train(x_tr, y_tr, model, optimizer, loss_fn, params)
        loss_ev = evaluate(x_ev, y_ev, model, loss_fn, params)

        params.writer.add_scalar('train_loss', loss_tr, epoch)
        params.writer.add_scalar('eval_loss', loss_ev, epoch)

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

        tqdm.write("epoch {} | train loss: {:05.3f} | eval loss: {:05.3f} |" \
            " best eval loss: {:05.3f}".format(
                epoch+1, loss_tr, loss_ev, best_loss_ev))

        losses_tr.append(loss_tr)
        losses_ev.append(loss_ev)
        np.save(os.path.join(model_dir, 'losses_tr'), losses_tr)
        np.save(os.path.join(model_dir, 'losses_ev'), losses_ev)
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

    # set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if params.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    model_loss_predict = {
        'cnn'             : (ConvNet, cnn_loss, class_pred),
        'capsule'         : (CapsuleNet, capsule_loss, class_pred),
        'darknet_d'       : (DarkNet, dark_loss, dark_pred),
        'darknet_r'       : (DarkNet, dark_loss, dark_pred),
        'darkcapsule'     : (DarkCapsuleNet, darkcapsule_loss, None),
    }

    model, loss_fn, predict_fn = model_loss_predict[args.model]
    model = model(params).to(device=params.device)

    if args.summary:
        summary(model, config.input_shape[args.model])

    optimizer = Adam(model.parameters(), lr=args.lr)

    if args.mode == 'train':
        train_and_evaluate(model, optimizer, loss_fn, params,
            data_dir, model_dir)
        
    if args.mode == 'overfit':
        utils.make_small_data(data_dir, 5)
        train_and_evaluate(model, optimizer, loss_fn, params,
            data_dir, model_dir, is_small=True)

    if args.mode == 'predict':
        x_tr, y_tr, x_ev, y_ev = utils.load_data(data_dir, True)
        x = x_ev[0:10]
        y = x_ev[0:10]

        if args.combine is None:
            y_hat, output = predict_fn(x_tr, model, model_dir, params, args.restore)
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
        



        # if args.model in ('darknet_d', 'darknet_r'):
        #     for i, image in enumerate(output):
        #         cv2.imshow(str(i), image)
        #     cv2.waitKey(0)
