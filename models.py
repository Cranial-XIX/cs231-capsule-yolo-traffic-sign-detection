import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvNet(nn.Module):
    def __init__(self, params):
        super(ConvNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
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

        self.softmax = nn.Softmax(dim=2)

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
                probs = self.softmax(logits)
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

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Conv2d(3, 256, 9)
        self.primary_capsules = CapsuleLayer(params,
            n_caps=8, n_nodes=-1, in_C=256, out_C=32, kernel=8, stride=2)
        self.traffic_sign_capsules = CapsuleLayer(params,
            n_caps=params.n_classes, n_nodes=32 * 9 * 9, in_C=8, out_C=16)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.primary_capsules(x)
        x = self.traffic_sign_capsules(x).squeeze()
        scores = (x ** 2).sum(dim=-1) ** 0.5
        return scores


class DarkNetD(nn.Module):
    def __init__(self, params):
        super(DarkNetD, self).__init__()

    def forward(self, x):
        # only one input x
        pass


class DarkNetR(nn.Module):
    def __init__(self, params):
        super(DarkNetR, self).__init__()

    def forward(self, x):
        # only one input x
        pass


class DarkCapsuleNet(nn.Module):
    def __init__(self, params):
        super(DarkCapsuleNet, self).__init__()

    def forward(self, x):
        # only one input x
        pass