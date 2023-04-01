import torch
from torch.nn.init import ones_, zeros_
from torch.nn import Parameter
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
from torch_geometric.nn import BatchNorm, LayerNorm
import copy
import numpy as np


class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))


def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss = - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def mask(g, x, mask_rate=0.5):
    num_nodes = g.num_nodes()
    perm = torch.randperm(num_nodes, device=x.device)
    num_mask_nodes = int(mask_rate * num_nodes)
    mask_nodes = perm[: num_mask_nodes]

    return mask_nodes


class Encoder1(nn.Module):
    def __init__(self, in_dim, out_dim, p1, hidden, num_layers):
        super(Encoder1, self).__init__()
        self.num_layers = num_layers
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.act = nn.ModuleList()
        for layer in range(num_layers):  # excluding the input layer
            self.act.append(nn.PReLU())
            if layer == 0 and num_layers == 1:
                self.conv.append(GraphConv(in_dim, out_dim))
                self.bn.append(BatchNorm(out_dim))
            elif layer == 0:
                self.conv.append(GraphConv(in_dim, hidden))
                self.bn.append(BatchNorm(hidden))
            else:
                self.conv.append(GraphConv(hidden, out_dim))
                self.bn.append(BatchNorm(out_dim))

        self.dp = nn.Dropout(p1)

    def forward(self, graph, feat):
        h = self.dp(feat)
        for i, layer in enumerate(self.conv):
            h = layer(graph, h)
            h = self.bn[i](h)
            h = self.act[i](h)

        return h

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.conv[i].reset_parameters()
            self.bn[i].reset_parameters()


class CG(nn.Module):
    def __init__(self, in_dim, out_dim, p1, rate, hidden, layers, t, alpha):
        super(CG, self).__init__()
        self.online_encoder = Encoder1(in_dim, out_dim, p1, hidden, layers)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.reset_parameters()
        self.rate = rate
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.criterion = self.setup_loss_fn("sce", 1)
        self.t = t
        self.alpha = alpha
        self.decoder = Encoder1(out_dim, in_dim, 0.0, hidden, 1)
        # self.decoder = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, in_dim))
        self.proj_head = nn.Sequential(nn.Linear(out_dim, 256), nn.ReLU(inplace=True),
                                       nn.Linear(256, 128))
        self.pred_head = nn.Sequential(nn.Linear(out_dim, 256), nn.ReLU(inplace=True),
                                       nn.Linear(256, 128))

        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.pred_head.parameters()) + \
               list(self.proj_head.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.
        Args:
            mm (float): Momentum used in moving average update.
        """
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def cl_loss(self, x, x_aug, t=0.2):

        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / t)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss

    def forward(self, graph, feat):

        mask_nodes = mask(graph, feat, mask_rate=self.rate)
        x = feat.clone()
        x[mask_nodes] = 0.0
        x[mask_nodes] += self.enc_mask_token

        h1 = self.online_encoder(graph, x)
        h11 = h1.clone()
        h11[mask_nodes] = 0.0
        re_x = self.decoder(graph, h11)
        loss1 = self.criterion(re_x[mask_nodes], feat[mask_nodes].detach())

        # Contrastive
        with torch.no_grad():
            h2 = self.target_encoder(graph, feat)
        c_h = self.proj_head(h2)
        c_m = self.pred_head(h1)
        cl_loss = self.cl_loss(c_h[mask_nodes], c_m[mask_nodes], self.t)

        return loss1 + cl_loss * self.alpha

    def get_embed(self, graph, feat):
        h1 = self.online_encoder(graph, feat)

        return h1.detach()
