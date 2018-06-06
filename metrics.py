import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils

from sklearn.metrics import auc, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

def recog_acc(y, y_hat, params):
    y_hat_pred = np.argmax(y_hat, axis=1)
    return np.sum(y == y_hat_pred) / y.shape[0]

def recog_auc(y, y_hat, params, show=False, save=False):
    n_classes = params.n_classes
    y = np.eye(n_classes)[y]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], th = roc_curve(y[:, i], y_hat[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr['micro'], tpr['micro'], _ = roc_curve(y.ravel(), y_hat.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    if show or save:
        plt.figure(1)
        plt.step(fpr['micro'], tpr['micro'],
            color='darkorange', alpha=0.2, where='post')

        plt.fill_between(fpr['micro'], tpr['micro'],
            step='post', alpha=0.2, color='darkorange')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Average auc score, micro-averaged over' \
            'all classes: auc={0:0.2f}'.format(roc_auc['micro']))

    if show:
        plt.show()

    if save:
        plt.savefig(config.model_dir[params.model]+'/r_auc.png')

    return roc_auc['micro']


def recog_pr(y, y_hat, params, show=False, save=False):
    n_classes = params.n_classes
    y = np.eye(n_classes)[y]

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y[:, i], y_hat[:, i])
        average_precision[i] = average_precision_score(
            y[:, i], y_hat[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y.ravel(), y_hat.ravel())
    average_precision["micro"] = average_precision_score(
        y, y_hat, average="micro")

    if show or save:
        plt.figure(2)
        plt.step(recall['micro'], precision['micro'],
            color='b', alpha=0.2, where='post')

        plt.fill_between(recall["micro"], precision["micro"],
            step='post', alpha=0.2, color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Average precision score, micro-averaged over' \
            'all classes: AP={0:0.2f}'.format(average_precision['micro']))

    if show:
        plt.show()

    if save:
        plt.savefig(config.model_dir[params.model]+'/r_pr.png')

    return average_precision['micro']


def calc_iou_individual(gt_box, pred_box):
    """
    Calculate IoU of single predicted and ground truth box
    @args:
        gt_box:   [xmin, ymin, xmax, ymax]
        pred_box: [xmin, ymin, xmax, ymax]
    @return:
        float: IoU
    @exception:
        AssertionError: if the box is obviously malformed
    """

    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x) * (far_y - near_y)
    true_box_area = (x2_t - x1_t) * (y2_t - y1_t)
    pred_box_area = (x2_p - x1_p) * (y2_p - y1_p)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou


def single_img_confusion(y_, y_hat_, iou_th):
    n1, n2 = y_.shape[0], y_hat_.shape[0]
    gt_hit, pred_hit = set(), set()
    for i in range(n1):     # loop thru ground truth boxes
        for j in range(n2): # loop thru predicted boxes
            iou = calc_iou_individual(y_[i], y_hat_[j])
            if iou > iou_th:
                gt_hit.add(i)
                pred_hit.add(j)
    n_gt_hit, n_pred_hit = len(gt_hit), len(pred_hit)
    tp, fp, fn = n_gt_hit, n2-n_pred_hit, n1-n_gt_hit
    return tp, fp, fn


def precision_and_recall(tp, fp, fn):
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return precision, recall


def plot_pr_curve(p, r, label=None, color=None, ax=None, name='default'):
    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    if color is None:
        color = config.colors[0]

    ax.scatter(r, p, label=label, s=20, color=color)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('Precision-Recall curve for {}'.format(name))
    ax.set_xlim([0.0, 1.1])
    ax.set_ylim([0.0, 1.1])
    return ax


def average_precision(p, r):
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(r >= recall_level).flatten()
            prec = max(p[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_p = np.mean(prec_at_rec)
    return avg_p


def detect_AP(y, y_hat, params, show=False, save=False, save_dir = None):
    iou_ths = np.linspace(0.5, 0.95, 10)
    conf_ths = np.linspace(0, 1, 100)

    im_indices = np.arange(y.shape[0])
    avg_ps = []
    ax = None

    for i, iou_th in enumerate(iou_ths):
        precisions = []
        recalls = []
        for conf_th in conf_ths:
            y_im_idx, y_bx, _ = utils.y_to_boxes_vec(
                y, params, conf_th=conf_th)
            y_hat_im_idx, y_hat_bx, _ = utils.y_to_boxes_vec(
                y_hat, params, conf_th=conf_th)

            TP = FP = FN = 0
            for j in im_indices:
                y_, y_hat_ = y_bx[y_im_idx == j], y_hat_bx[y_hat_im_idx == j]
                tp, fp, fn = single_img_confusion(y_, y_hat_, iou_th)
                TP += tp
                FP += fp
                FN += fn
            p, r = precision_and_recall(TP, FP, FN)
            precisions.append(p)
            recalls.append(r)

        p, r = np.array(precisions), np.array(recalls)
        avg_p = average_precision(p, r)

        if show or save:
            ax = plot_pr_curve(
                precisions, recalls, label='iou={:.2f}'.format(iou_th),
                color=config.colors[i*2], ax=ax, name=params.model)

        avg_ps.append(avg_p)

    if show:
        plt.legend()
        plt.show()

    if save:
        plt.legend()
        if save_dir is None:
            save_dir = config.model_dir[params.model]
        plt.savefig(save_dir+'/d_AP.png')

    avg_ps = np.array(avg_ps)
    return np.mean(avg_ps)


def detect_acc(y, y_hat, params):
    conf_th = 0.5
    iou_th = 0.5
    y_im_idx, y_bx, _ = utils.y_to_boxes_vec(y, params, conf_th=conf_th)
    y_hat_im_idx, y_hat_bx, _ = utils.y_to_boxes_vec(y_hat, params, conf_th=conf_th)
    im_indices = np.arange(y.shape[0])

    TP = FP = FN = 0
    for j in im_indices:
        y_, y_hat_ = y_bx[y_im_idx == j], y_hat_bx[y_hat_im_idx == j]
        tp, fp, fn = single_img_confusion(y_, y_hat_, iou_th)
        TP += tp
        FP += fp
        FN += fn
    
    p, r = precision_and_recall(TP, FP, FN)
    avg_pr = (p + r) / 2
    return avg_pr


def detect_and_recog_acc(y, y_hat, params, show=False, save=False):
    conf_th = 0.5
    iou_th = 0.5
    y_im_idx, y_bx, y_cls = utils.y_to_boxes_vec(y, params, conf_th=conf_th)
    y_hat_im_idx, y_hat_bx, y_hat_cls = utils.y_to_boxes_vec(
        y_hat, params, conf_th=conf_th)
    im_indices = np.arange(y.shape[0])
    TP = FP = FN = 0
    for c in range(params.n_classes):
        for j in im_indices:
            y_ = y_bx[(y_im_idx == j) * (y_cls == c)]
            y_hat_ = y_hat_bx[(y_hat_im_idx == j) * (y_hat_cls == c)]
            tp, fp, fn = single_img_confusion(y_, y_hat_, iou_th)
            TP += tp
            FP += fp
            FN += fn
    p, r = precision_and_recall(TP, FP, FN)
    return 2*p*r/(p+r+1e-8)


def detect_and_recog_mAP(y, y_hat, params, show=False, save=False, save_dir=None):
    params.n_classes = 43
    iou_ths = np.linspace(0.5, 0.95, 10)
    conf_ths = np.linspace(0, 1, 100)

    im_indices = np.arange(y.shape[0])
    avg_ps = []

    for c in range(params.n_classes):
        plt.figure(c, figsize=(10, 8))
        ax = plt.gca()
        for i, iou_th in enumerate(iou_ths):
            precisions = []
            recalls = []
            for conf_th in conf_ths:
                y_im_idx, y_bx, y_cls = utils.y_to_boxes_vec(
                    y, params, conf_th=conf_th)
                y_hat_im_idx, y_hat_bx, y_hat_cls = utils.y_to_boxes_vec(
                    y_hat, params, conf_th=conf_th)

                TP = FP = FN = 0
                for j in im_indices:
                    y_ = y_bx[(y_im_idx == j) * (y_cls == c)]
                    y_hat_ = y_hat_bx[(y_hat_im_idx == j) * (y_hat_cls == c)]
                    tp, fp, fn = single_img_confusion(y_, y_hat_, iou_th)
                    TP += tp
                    FP += fp
                    FN += fn
                        
                p, r = precision_and_recall(TP, FP, FN)
                precisions.append(p)
                recalls.append(r)
            p, r = np.array(precisions), np.array(recalls)
            avg_p = average_precision(p, r)

            if show or save:
                ax = plot_pr_curve(
                    precisions, recalls, label='iou={:.2f}'.format(iou_th),
                    color=config.colors[i*2], ax=ax, name=params.model)
            avg_ps.append(avg_p)

        if save:
            plt.legend()
            if save_dir is None:
                save_dir = config.model_dir[params.model]
            plt.savefig(save_dir+'/d&r_mAP_class_{}.png'.format(c))

        if show:
            plt.legend()
            plt.show()

    classes = np.sign(y[:, :, :, 5:].reshape(-1, 43).sum(axis=0)) # 43
    mask = classes > 0
    avg_ps = np.array(avg_ps).reshape(params.n_classes, -1)[mask]

    return np.mean(avg_ps)


def darkcapsule_acc(y, y_hat, params):
    conf_th = 0.5
    iou_th = 0.5
    y_hat = y_hat * np.sqrt(2)
    y_cls = y[:,:,:,5:]
    cap_other = y_hat[:,:,:,:,5:]
    y_hat_cls = np.argmax(np.sum((cap_other ** 2), -1) ** 0.5, -1)
    B, g, _ = y_hat_cls.shape
    y_hat_cls = np.eye(params.n_classes)[y_hat_cls.reshape(-1)].reshape(
        B,g,g,-1)

    y_hat = np.sum((y_hat * np.expand_dims(y_cls, 4)), 3).squeeze()
    y_hat = np.concatenate((y_hat[:,:,:,:5], y_hat_cls), 3)
    
    y_im_idx, y_bx, _ = utils.y_to_boxes_vec(y, params, conf_th=conf_th)
    y_hat_im_idx, y_hat_bx, _ = utils.y_to_boxes_vec(
        y_hat, params, conf_th=conf_th)
    im_indices = np.arange(y.shape[0])

    TP = FP = FN = 0
    for j in im_indices:
        y_, y_hat_ = y_bx[y_im_idx == j], y_hat_bx[y_hat_im_idx == j]
        tp, fp, fn = single_img_confusion(y_, y_hat_, iou_th)
        TP += tp
        FP += fp
        FN += fn
    
    p, r = precision_and_recall(TP, FP, FN)
    return int(p*100) + r


if __name__ == "__main__":
    # please see the following tests to understand how to
    # use those metrics
    y, y_hat = np.array([0,1,2,3]), np.eye(4)
    # test recognition auc
    assert recog_auc(y, y_hat) == 1.0
    # test recognition precision-recall
    assert recog_pr(y, y_hat) == 1.0
    # test detection average precision (AP)
    y, y_hat = pickle.load(open('debug/darknet_d.p', 'rb'))
    AP = detect_AP(y, y, params, show=True)
    # test detection and classification mean Average Precision (mAP)
    y, y_hat = pickle.load(open('debug/darknet_d.p', 'rb'))
    mAP = detect_and_recog_mAP(y, y, params, show=True)