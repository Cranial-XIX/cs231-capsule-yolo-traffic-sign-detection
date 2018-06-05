import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
from collections import OrderedDict

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, c, w, h):
        super(UnFlatten, self).__init__()
        self.c, self.w, self.h = c, w, h

    def forward(self, x):
        return x.view(-1, self.c, self.w, self.h)


class ConvNet(nn.Module):
    def __init__(self, params):
        super(ConvNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(params.dropout),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(params.dropout),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, params.n_classes)
        )

    def forward(self, x):
        scores = self.cnn(x)
        return scores


class CapsuleLayer(nn.Module):
    def __init__(self, params, n_caps, n_nodes, in_C, out_C,
        kernel=None, stride=None, n_iter=3):
        super(CapsuleLayer, self).__init__()

        self.params = params
        self.n_iter = n_iter
        self.n_nodes = n_nodes
        self.n_caps = n_caps

        if n_nodes != -1: # caps -> caps layer
            self.route_weights = nn.Parameter(0.1 *
                torch.randn(1, n_nodes, n_caps, in_C, out_C))
        else:   # conv -> caps layer
            self.capsules = nn.ModuleList([nn.Conv2d(
                in_C, out_C, kernel, stride=stride
            ) for _ in range(n_caps)])

    def squash(self, v):
        squared_norm = (v ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * v / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.n_nodes != -1:
            priors = (x[:, :, None, None, :] @ self.route_weights).squeeze(4)
            logits = torch.zeros(*priors.size()).to(device=self.params.device)
            # dynamic routing
            for i in range(self.n_iter):
                probs = F.softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=1, keepdim=True))
                if i != self.n_iter - 1:
                    delta = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta
        else:
            outputs = [cap(x).view(x.size(0), -1, 1) for cap in self.capsules]
            outputs = self.squash(torch.cat(outputs, dim=-1))
        return outputs


class CapsuleNet(nn.Module):
    def __init__(self, params):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 256, 9)
        self.primary_capsules = CapsuleLayer(params,
            n_caps=8, n_nodes=-1, in_C=256, out_C=16, kernel=8, stride=2)
        self.traffic_sign_capsules = CapsuleLayer(params,
            n_caps=params.n_classes, n_nodes=16 * 9 * 9, in_C=8, out_C=16)

        self.decoder = nn.Sequential(
            nn.Linear(16, 16 * 4 * 4),
            nn.ReLU(),
            UnFlatten(16, 4, 4),
            nn.Upsample((8, 8)),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
            nn.Upsample((16, 16)),
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.Upsample((32, 32)),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, y=None, recon=False):
        x = F.relu(self.conv1(x))
        x = self.primary_capsules(x)
        x = self.traffic_sign_capsules(x).squeeze()
        scores = (x ** 2).sum(dim=-1) ** 0.5

        if not recon:
            return scores
        else:
            t = torch.gather(x, 1, y.repeat(16, 1).t().unsqueeze(1)).squeeze()
            decoded = self.decoder(t)
            return scores, decoded

class DarkNet(nn.Module):
    def __init__(self, params):
        super(DarkNet, self).__init__()

        self.params = params
        self.model = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(3, 32, 3, padding=1, bias=False)),
            ('bn_1', nn.BatchNorm2d(32, momentum=0.01)),
            ('relu_1', nn.LeakyReLU(0.1)),
            ('maxpool_1', nn.MaxPool2d(2)),

            ('conv_2', nn.Conv2d(32, 64, 3, padding=1, bias=False)),
            ('bn_2', nn.BatchNorm2d(64, momentum=0.01)),
            ('relu_2', nn.LeakyReLU(0.1)),
            ('maxpool_2', nn.MaxPool2d(2)),

            ('conv_3', nn.Conv2d(64, 128, 3, padding=1, bias=False)),
            ('bn_3', nn.BatchNorm2d(128, momentum=0.01)),
            ('relu_3', nn.LeakyReLU(0.1)),

            ('conv_4', nn.Conv2d(128, 64, 1, bias=False)),
            ('bn_4', nn.BatchNorm2d(64, momentum=0.01)),
            ('relu_4', nn.LeakyReLU(0.1)),

            ('conv_5', nn.Conv2d(64, 128, 3, padding=1, bias=False)),
            ('bn_5', nn.BatchNorm2d(128, momentum=0.01)),
            ('relu_5', nn.LeakyReLU(0.1)),
            ('maxpool_3', nn.MaxPool2d(2)),

            ('conv_6', nn.Conv2d(128, 256, 3, padding=1, bias=False)),
            ('bn_6', nn.BatchNorm2d(256, momentum=0.01)),
            ('relu_6', nn.LeakyReLU(0.1)),

            ('conv_7', nn.Conv2d(256, 128, 1, bias=False)),
            ('bn_7', nn.BatchNorm2d(128, momentum=0.01)),
            ('relu_7', nn.LeakyReLU(0.1)),

            ('conv_8', nn.Conv2d(128, 256, 3, padding=1, bias=False)),
            ('bn_8', nn.BatchNorm2d(256, momentum=0.01)),
            ('relu_8', nn.LeakyReLU(0.1)),
            ('maxpool_4', nn.MaxPool2d(2)),

            ('conv_9', nn.Conv2d(256, 512, 3, padding=1, bias=False)),
            ('bn_9', nn.BatchNorm2d(512, momentum=0.01)),
            ('relu_9', nn.LeakyReLU(0.1)),

            ('conv_10', nn.Conv2d(512, 256, 1, bias=False)),
            ('bn_10', nn.BatchNorm2d(256, momentum=0.01)),
            ('relu_10', nn.LeakyReLU(0.1)),

            ('conv_11', nn.Conv2d(256, 512, 3, padding=1, bias=False)),
            ('bn_11', nn.BatchNorm2d(512, momentum=0.01)),
            ('relu_11', nn.LeakyReLU(0.1)),

            ('conv_12', nn.Conv2d(512, 256, 1, bias=False)),
            ('bn_12', nn.BatchNorm2d(256, momentum=0.01)),
            ('relu_12', nn.LeakyReLU(0.1)),

            ('conv_13', nn.Conv2d(256, 512, 3, padding=1, bias=False)),
            ('bn_13', nn.BatchNorm2d(512, momentum=0.01)),
            ('relu_13', nn.LeakyReLU(0.1)),
            ('maxpool_5', nn.MaxPool2d(2)),

            ('conv_14', nn.Conv2d(512, 1024, 3, padding=1, bias=False)),
            ('bn_14', nn.BatchNorm2d(1024, momentum=0.01)),
            ('relu_14', nn.LeakyReLU(0.1)),

            ('conv_15', nn.Conv2d(1024, 512, 1, bias=False)),
            ('bn_15', nn.BatchNorm2d(512, momentum=0.01)),
            ('relu_15', nn.LeakyReLU(0.1)),

            ('conv_16', nn.Conv2d(512, 1024, 3, padding=1, bias=False)),
            ('bn_16', nn.BatchNorm2d(1024, momentum=0.01)),
            ('relu_16', nn.LeakyReLU(0.1)),

            ('conv_17', nn.Conv2d(1024, 512, 1, bias=False)),
            ('bn_17', nn.BatchNorm2d(512, momentum=0.01)),
            ('relu_17', nn.LeakyReLU(0.1)),

            ('conv_18', nn.Conv2d(512, 1024, 3, padding=1, bias=False)),
            ('bn_18', nn.BatchNorm2d(1024, momentum=0.01)),
            ('relu_18', nn.LeakyReLU(0.1)),

            ('conv_19', nn.Conv2d(1024,
                5 * params.n_boxes + params.n_classes, 1, bias=False))
        ]))

    def forward(self, x):
        out = self.model(x).permute(0, 2, 3, 1)
        split = 5 * self.params.n_boxes
        y_box = F.sigmoid(out[:,:,:,:split])

        if self.params.n_classes == 0:
            y = y_box
        else:
            y_cls = F.softmax(out[:,:,:,split:], dim=-1)
            y = torch.cat((y_box, y_cls), dim=-1)
        return y

    def load_weights(self, weights_dir, n_load_layer):
        model_dict = self.state_dict()
        pretr_dict = np.load(weights_dir)
        name_dict = {
            'kernel:0': ('conv', 'weight'),
            'biases:0': ('bn', 'bias'),
            'gamma:0': ('bn', 'weight'),
            'moving_mean:0': ('bn', 'running_mean'), 
            'moving_variance:0': ('bn', 'running_var'),
        }

        print('Load weights from ' + weights_dir)
        load_dict = {}
        for key, v in pretr_dict.items():
            index, layer = key.split('-')
            index = int(index) + 1

            if index > n_load_layer:
                continue

            _, name = layer.split('/')
            layer_type, param_type = name_dict[name]

            param = torch.from_numpy(v)
            if layer_type == 'conv' and param_type == 'weight':
                param = param.permute(3, 2, 0, 1)

            new_key = "model.{}_{}.{}".format(layer_type, index, param_type)  
            load_dict[new_key] = param

        model_dict.update(load_dict)
        self.load_state_dict(model_dict)

class DarkCapsuleNet2(nn.Module):
    def __init__(self, params):
        super(DarkCapsuleNet2, self).__init__()

        self.params = params
        self.conv = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(3, 32, 4, 2, padding=1)),
            ('bn_1', nn.BatchNorm2d(32)),
            ('relu_1', nn.LeakyReLU(0.1)),
            ('drop_1', nn.Dropout(params.dropout)),

            ('conv_2', nn.Conv2d(32, 64, 4, 2, padding=1)),
            ('bn_2', nn.BatchNorm2d(64)),
            ('relu_2', nn.LeakyReLU(0.1)),
            ('drop_2', nn.Dropout(params.dropout)),

            ('conv_3', nn.Conv2d(64, 128, 4, 2, padding=1)),
            ('bn_3', nn.BatchNorm2d(128)),
            ('relu_3', nn.LeakyReLU(0.1)),
            ('drop_3', nn.Dropout(params.dropout)),

            ('conv_4', nn.Conv2d(128, 256, 4, 2, padding=1)),
            ('bn_4', nn.BatchNorm2d(256)),
            ('relu_4', nn.LeakyReLU(0.1)),
            ('drop_4', nn.Dropout(params.dropout))]))

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(3, 32, 4, 2, padding=1)),
            ('bn_1', nn.BatchNorm2d(32)),
            ('relu_1', nn.LeakyReLU(0.1)),
            ('drop_1', nn.Dropout(params.dropout)),

            ('conv_2', nn.Conv2d(32, 64, 4, 2, padding=1)),
            ('bn_2', nn.BatchNorm2d(64)),
            ('relu_2', nn.LeakyReLU(0.1)),
            ('drop_2', nn.Dropout(params.dropout)),

            ('conv_3', nn.Conv2d(64, 128, 4, 2, padding=1)),
            ('bn_3', nn.BatchNorm2d(128)),
            ('relu_3', nn.LeakyReLU(0.1)),
            ('drop_3', nn.Dropout(params.dropout)),

            ('conv_4', nn.Conv2d(128, 256, 4, 2, padding=1)),
            ('bn_4', nn.BatchNorm2d(256)),
            ('relu_4', nn.LeakyReLU(0.1)),
            ('drop_4', nn.Dropout(params.dropout)),

            ('conv_5', nn.Conv2d(256, 512, 4, 2, padding=1)),
            ('bn_5', nn.BatchNorm2d(512)),
            ('relu_5', nn.LeakyReLU(0.1)),
            ('drop_5', nn.Dropout(params.dropout)),
        ]))

        self.primary_capsules = CapsuleLayer(params,
            n_caps=8, n_nodes=-1, in_C=512, out_C=16, kernel=1, stride=1)

        self.traffic_sign_capsules = CapsuleLayer(params,
            n_caps=params.n_grid**2,
            n_nodes=16 * 7 * 7, in_C=8, out_C=5+params.n_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv2(x)
        x = self.primary_capsules(x)
        capsules = self.traffic_sign_capsules(x).squeeze().view(
            batch_size, self.params.n_grid, self.params.n_grid, -1)
        return capsules


class DarkCapsuleNet(nn.Module):
    def __init__(self, params):
        super(DarkCapsuleNet, self).__init__()

        self.params = params

        self.conv = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(3, 128, 3, padding=1)),
            ('bn_1', nn.BatchNorm2d(128)),
            ('relu_1', nn.LeakyReLU(0.1)),

            ('conv_2', nn.Conv2d(128, 256, 3, padding=1)),
            ('bn_2', nn.BatchNorm2d(256)),
            ('relu_2', nn.LeakyReLU(0.1)),

            ('conv_3', nn.Conv2d(256, 64, 4, 2, padding=1)),
            ('bn_3', nn.BatchNorm2d(64)),
            ('relu_3', nn.LeakyReLU(0.1)),

            ('conv_4', nn.Conv2d(64, 128, 4, 2, padding=1)),
            ('bn_4', nn.BatchNorm2d(128)),
            ('relu_4', nn.LeakyReLU(0.1)),

            ('conv_5', nn.Conv2d(128, 256, 4, 2, padding=1)),
            ('bn_5', nn.BatchNorm2d(256)),
            ('relu_5', nn.LeakyReLU(0.1)),
        ]))

        self.traffic_sign_capsules = CapsuleLayer(params,
            n_caps=params.n_classes,
            n_nodes=16 * 32, in_C=8, out_C=5+16)

        self.decoder = nn.Sequential(
            nn.Linear(16, 16 * 4 * 4),
            nn.ReLU(),
            UnFlatten(16, 4, 4),
            nn.Upsample((8, 8)),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.ReLU(),
            nn.Upsample((16, 16)),
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.Upsample((32, 32)),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        B, _, H, W = x.size()
        g = self.params.n_grid
        x = self.conv(x)
        x = torch.chunk(x.view(B, 256, 4, 4*g**2), 
            self.params.n_grid**2, 3)  # n_grid^2 * [B, 256, 4, 4]

        x = [xx.permute(0, 2, 3, 1).contiguous().view(B, -1, 8).unsqueeze(0) for xx in x] # n_grid^2 * [B, 512, 8]
        x = torch.cat(x, 0) # n_grid^2 * B, 512 * 8

        x = self.traffic_sign_capsules(x.view(-1, 512, 8)).squeeze() # [n_grid^2 *B, 43, 21]
        x = x.view(g, g, B, self.params.n_classes, 21).permute(2, 0, 1, 3, 4)
        return x