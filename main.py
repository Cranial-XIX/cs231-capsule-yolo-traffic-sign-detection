import argparse
import config
import numpy as np
import pickle
import os
import sys
import torch
import utils

from models import ConvNet, CapsuleNet, DarkNetD, DarkNetR, DarkCapsuleNet
from loss_fns import cnn_loss, capsule_loss, dark_d_loss, dark_r_loss, darkcapsule_loss
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torchsummary import summary
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='cnn', help=' | '.join(config.model_names))
parser.add_argument('--mode', default='train', help='train | predict')
parser.add_argument('--summary', default=True, help='if summarize model', action='store_true')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')


def load_data(train_data_path, eval_data_path):
    x_tr, y_tr = pickle.load(open(train_data_path, 'rb'))
    x_ev, y_ev = pickle.load(open(eval_data_path, 'rb'))
    return x_tr, y_tr, x_ev, y_ev


def train(x, y, model, optimizer, loss_fn, params):
    model.train()

    x, y = utils.shuffle(x, y)
    total = len(y)
    n_batch = total // params.batch_size
    x_split, y_split = np.array_split(x, n_batch), np.array_split(y, n_batch)
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

        avg_loss += loss.item() / n_batch

    return avg_loss


def evaluate(x, y, model, loss_fn, params):
    model.eval()

    total = len(y)
    n_batch = total // params.batch_size
    x_split, y_split = np.array_split(x, n_batch), np.array_split(y, n_batch)
    avg_loss = 0

    with torch.no_grad():
        for i, (x_bch, y_bch) in enumerate(zip(x_split, y_split)):
            x_bch = torch.from_numpy(x_bch).float().permute(0, 3, 1, 2).to(
                device=params.device)
            y_bch = torch.from_numpy(y_bch).to(device=params.device)

            y_hat_bch = model(x_bch)
            loss = loss_fn(y_hat_bch, y_bch, params)
            avg_loss += loss.item() / num_steps

    return avg_loss

def train_and_evaluate(model, optimizer, loss_fn, params,
    train_data_path, eval_data_path, model_dir, restore_file=None):
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        print("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, params, optimizer)

    losses_tr = []
    losses_ev = []
    best_loss_ev = float('inf')
    x_tr, y_tr, x_ev, y_ev = load_data(train_data_path, eval_data_path)

    for epoch in range(params.n_epochs):
        loss_tr = train(x_tr, y_tr, model, optimizer, loss_fn, params).item()
        loss_ev = evaluate(x_ev, y_ev, model, loss_fn, params).item()

        params.writer.add_scalar('train_loss', loss_tr, epoch)
        params.writer.add_scalar('eval_loss', loss_ev, epoch)

        is_best = loss_ev < best_ev_loss

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

        print("train loss: {:05.3f} | eval loss: {:05.3f} |" \
            " best eval loss: {:05.3f}".format(
                loss_tr, loss_ev, best_loss_ev))

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


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir, model_dir = get_data_and_model_dir(args.model)
    json_path = os.path.join(model_dir, 'params.json')
    params = utils.Params(json_path)

    params.device = "cuda" if torch.cuda.is_available() else "cpu"
    params.seed = args.seed

    params.writer = SummaryWriter()
    # set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if params.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    model_and_loss = {
        'cnn'             : (ConvNet(params), cnn_loss),
        'capsule'         : (CapsuleNet(params), capsule_loss),
        'darknet_d'       : (DarkNetD(params), dark_d_loss),
        'darknet_r'       : (DarkNetR(params), dark_r_loss),
        'darkcapsule'     : (DarkCapsuleNet(params), darkcapsule_loss),
    }
    model, loss_fn = model_and_loss[args.model]
    if args.summary:
        summary(model, config.input_shape[args.model])

    optimizer = Adam(model.parameters(), lr=args.lr)
    if args.mode == 'train':
        train_and_evaluate(model, optimizer, loss_fn, params,
            data_dir+'/train.p', data_dir+'/eval.p', model_dir)