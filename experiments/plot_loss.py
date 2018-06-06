import matplotlib.pyplot as plt
import numpy as np
import sys
data_dir = sys.argv[1]
train_losses = np.load(data_dir + '/losses_tr.npy')
val_losses = np.load(data_dir + '/losses_ev.npy')

train_metrics = np.load(data_dir + '/metrics_tr.npy')
val_metrics = np.load(data_dir + '/metrics_ev.npy')
print("train min:", np.min(train_losses))
print("val min:", np.min(val_losses))
print("train max ev:", np.max(train_metrics))
print("val max ev:", np.max(val_metrics))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()