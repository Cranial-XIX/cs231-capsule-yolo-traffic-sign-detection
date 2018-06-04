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

from metrics import recog_acc, recog_auc, recog_pr, detect_AP, detect_and_recog_mAP, detect_acc
from models import ConvNet, CapsuleNet, DarkNet, DarkCapsuleNet
from loss_fns import cnn_loss, capsule_loss, dark_loss, darkcapsule_loss
from predict_fns import dark_pred, class_pred, dark_class_pred
from tensorboardX import SummaryWriter
from torch.optim import Adam, lr_scheduler
from torchsummary import summary
from tqdm import trange, tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='cnn', help=' | '.join(config.model_names))
parser.add_argument('--mode', default='train', help='train | predict | overfit')
parser.add_argument('--summary', default=True, help='if summarize model', action='store_true')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--restore', default=None, help="last | best")
parser.add_argument('--combine', default=None, help="darknet_r | darknet_d")
parser.add_argument('--recon', default=False, help='if use reconstruction loss', action='store_true')
parser.add_argument('--recon_coef', default=5e-4, help='reconstruction coefficient')
parser.add_argument('--eval_every', default=1, type=int, help='evaluate metric every # epochs')
parser.add_argument('--fine_tune', default=-1, type=int, help='number of fixed layer in fine tuning')
parser.add_argument('--no_metric', help='do not compute metric', action='store_true')
parser.add_argument('--save', default=False, help='save result', action='store_true')
parser.add_argument('--model_dir', default=None, help='model dir')
parser.add_argument('--show', default=False, help='save result', action='store_true')



def train(x, y, model, optimizer, loss_fn, metric, params, if_eval=True):
    model.train()

    x, y = utils.shuffle(x, y)
    total = len(y)
    n_batch = (total + params.batch_size - 1) // params.batch_size
    x_split, y_split = np.array_split(x, n_batch), np.array_split(y, n_batch)
    n = y.shape[0]
    t = trange(n_batch)
    avg_loss = 0
    y_hat = []
    avg_iou = 0

    for i, (x_bch, y_bch) in enumerate(zip(x_split, y_split)):
        # x_bch = utils.augmentation(x_bch, params.model)
        x_bch = torch.from_numpy(x_bch).float().permute(0, 3, 1, 2).to(
            device=params.device)
        y_bch = torch.from_numpy(y_bch).to(device=params.device)

        if params.model == 'capsule' and params.recon:
            y_hat_bch, recon = model(x_bch, y_bch, True)
            loss = loss_fn(y_hat_bch, y_bch, params, x_bch, recon)
        else:
            y_hat_bch = model(x_bch)
            loss = loss_fn(y_hat_bch, y_bch, params)

        y_hat.append(y_hat_bch.data.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t.set_postfix(loss='{:05.3f}'.format(loss.item()))
        t.update()

        avg_loss += loss.item() / n_batch

        if args.model == 'darknet_d':
            avg_iou += params.avg_iou.item() / n_batch

    y_hat = np.concatenate(y_hat, axis=0)

    # shrink size for faster calculation of metric
    metric_score = -1

    if if_eval and not args.no_metric:
        if n > config.max_metric_samples:
            i = np.random.choice(n, config.max_metric_samples).astype(int)
            y, y_hat = y[i], y_hat[i]
        metric_score = metric(y, y_hat, params)
    
    if args.model == 'darknet_d':
        tqdm.write("train avg iou: {:05.3f}".format(avg_iou))
    return avg_loss, metric_score


def evaluate(x, y, model, loss_fn, metric, params, if_eval=True):
    model.eval()

    total = len(y)
    n_batch = (total + params.batch_size - 1) // params.batch_size
    x_split, y_split = np.array_split(x, n_batch), np.array_split(y, n_batch)
    n = y.shape[0]
    avg_loss = 0
    y_hat = []
    avg_iou = 0

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

            y_hat.append(y_hat_bch.data.cpu().numpy())
            avg_loss += loss / n_batch

            if args.model == 'darknet_d':
                avg_iou += params.avg_iou.item() / n_batch
                
    y_hat = np.concatenate(y_hat, axis=0)
    
    # shrink size for faster calculation of metric
    metric_score = -1

    if if_eval and not args.no_metric:
        if n > config.max_metric_samples:
            i = np.random.choice(n, config.max_metric_samples).astype(int)
            y, y_hat = y[i], y_hat[i]

        metric_score = metric(y, y_hat, params)
    
    if args.model == 'darknet_d':
        tqdm.write("test avg iou: {:05.3f}".format(avg_iou))

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
    best_metric_ev = float('inf')
    best_loss_ev = float('inf')

    x_tr, y_tr, x_ev, y_ev = utils.load_data(data_dir, is_small)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=params.lr_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=params.lr_decay)
    
    for epoch in range(params.n_epochs):
        if_eval = ((epoch+1) % params.eval_every == 0)
        loss_tr, metric_tr = train(
            x_tr, y_tr, model, optimizer, loss_fn, metric, params, if_eval)
        loss_ev, metric_ev = evaluate(
            x_ev, y_ev, model, loss_fn, metric, params, if_eval)

        scheduler.step()

        params.writer.add_scalar('train_loss', loss_tr, epoch)
        params.writer.add_scalar('eval_loss', loss_ev, epoch)

        is_best = metric_ev < best_metric_ev

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
            best_metric_ev = metric_ev

        if loss_ev < best_loss_ev:
            best_loss_ev = loss_ev 

        if if_eval:
            params.writer.add_scalar('train_metric', metric_tr, epoch)
            params.writer.add_scalar('eval_metric', metric_ev, epoch)
            tqdm.write(
                "epoch {} | train loss: {:05.3f} | eval loss: {:05.3f} |" \
                " best eval loss: {:05.3f} | " \
                "train metric: {:05.3f} | eval metric: {:05.3f} | best eval metric {:05.3f}".format(
                    epoch+1, loss_tr, loss_ev,
                    best_loss_ev, metric_tr, metric_ev, best_metric_ev))
            metrics_tr.append(metric_tr)
            metrics_ev.append(metric_ev)
            np.save(os.path.join(model_dir, 'metrics_tr'), metrics_tr)
            np.save(os.path.join(model_dir, 'metrics_ev'), metrics_ev)

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

    params.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    params.seed = args.seed
    params.dropout = args.dropout

    params.writer = SummaryWriter()
    return params


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir, model_dir = get_data_and_model_dir(args.model)
    if args.model_dir is not None:
        model_dir= args.model_dir
        
    params = load_params(model_dir, args)
    params.model = args.model
    params.recon = args.recon
    params.recon_coef = args.recon_coef
    params.eval_every = args.eval_every

    # set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if params.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    model_loss_predict = {
        'cnn'             : (ConvNet, cnn_loss, class_pred, recog_acc),
        'capsule'         : (CapsuleNet, capsule_loss, class_pred, recog_acc),
        'darknet_d'       : (DarkNet, dark_loss, dark_pred, detect_acc),
        'darknet_r'       : (DarkNet, dark_loss, dark_pred, detect_and_recog_mAP),
        'darkcapsule'     : (DarkCapsuleNet, darkcapsule_loss, None, detect_and_recog_mAP),
    }

    model, loss_fn, predict_fn, metric = model_loss_predict[args.model]
    model = model(params).to(device=params.device)

    if args.summary:
        summary(model, config.input_shape[args.model])

    if args.fine_tune > 0:
        model.load_weights('./darknet19_weights.npz', 18)
        for name, param in model.named_parameters():
            layer_type, index = name.split('.')[1].split('_')
            if int(index) <= params.fine_tune:
                param.requires_grad = False

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if args.mode == 'train':
        train_and_evaluate(model, optimizer, loss_fn, metric, params,
            data_dir, model_dir, restore_file=args.restore)
        
    if args.mode == 'overfit':
        utils.make_small_data(data_dir, 3)
        train_and_evaluate(
            model, optimizer, loss_fn, metric, params,
            data_dir, model_dir, is_small=True, restore_file=args.restore)

    if args.mode == 'predict':
        if args.restore is None:
            print('Must give restore file last/bast')
            sys.exit()
        # x_tr, y_tr, x_ev, y_ev = utils.load_data(data_dir)
        # x = x_ev[0:100]
        # y = y_ev[0:100]
        x, y = pickle.load(open(data_dir + '/train.p', 'rb'))
        x = x[0:3]
        y = y[0:3]

        if args.combine is None:
            y_hat, output = predict_fn(x, model, model_dir, params, args.restore)
            # pickle.dump((y, y_hat), open('./debug/{}.p'.format(args.model), 'wb'))
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

            # pickle.dump((y, dark_y_hat, class_y_hat), open('./debug/{}-{}.p'.format(args.model, args.combine), 'wb'))

        if args.save:
            save_dir = os.path.join(model_dir, 'result')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if args.model in ('darknet_d', 'darknet_r'):
                for i, image in enumerate(output):
                    cv2.imwrite(os.path.join(save_dir, str(i) + '.jpg'), image)

        if args.show:
            if args.model in ('darknet_d', 'darknet_r'):
                print(y_hat[0, 0, 0, 0])

                for i, image in enumerate(output):
                    cv2.imshow(str(i), image)
                cv2.waitKey(0)
            else:
                print("acc", np.mean(y == output))