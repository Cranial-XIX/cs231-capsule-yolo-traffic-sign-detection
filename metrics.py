import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import auc, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

def recog_auc(y, y_hat):
    y, y_hat = y.numpy(), y_hat.numpy()
    n_classes = y.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_hat[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_hat.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure(1)
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc[2]


def recog_pr(y, y_hat):
    y, y_hat = y.numpy(), y_hat.numpy()
    n_classes = y.shape[1]
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
        'all classes: AP={0:0.2f}'.format(average_precision["micro"]))
    plt.show()
    return average_precision["micro"]


def dummy_convert(y):
    return np.random.rand(3, 4), np.array([0,1,2]), np.array([0, 1, 0])


def detect_pr(y, y_hat):
    y_box, y_i, y_cls = dummy_convert(y)
    y_hat_box, y_hat_i, y_hat_cls = dummy_convert(y_hat)


if __name__ == "__main__":
    y, y_hat = torch.eye(4,4), torch.eye(4,4)
    # test for recog auc
    recog_pr(y, y_hat)
    # test for recog pr
    recog_pr(y, y_hat)
