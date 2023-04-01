import torch
from torch.optim import AdamW, Adam
from model import CG, CosineDecayScheduler
from torch_geometric import seed_everything
import numpy as np
import warnings
import yaml
import argparse
from eval import label_classification
from dataset import load_data

warnings.filterwarnings('ignore')


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    configs = configs[args.dataname]

    for k, v in configs.items():
        if "lr" in k or "w" in k:
            v = float(v)
        setattr(args, k, v)
    return args


parser = argparse.ArgumentParser(description="SimGOP")
parser.add_argument("--dataname", type=str, default="Cora")
parser.add_argument("--cuda", type=int, default=0)
args = parser.parse_args()
args = load_best_configs(args, "config.yaml")
dataname = args.dataname
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

label_type = args.label
graph, feat, label, train_mask, val_mask, test_mask = load_data(dataname)
n_feat = feat.shape[1]
n_classes = np.unique(label).shape[0]
graph = graph.to(device)
label = label.to(device)
feat = feat.to(device)


def train():
    seed_everything(0)
    model = CG(n_feat, 256, args.p1, args.rate, 256, args.layer, args.t, args.alpha).to(device)
    # scheduler
    lr_scheduler = CosineDecayScheduler(args.lr, 100, args.epochs)
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, args.epochs)
    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=0.001)

    for epoch in range(1, args.epochs + 1):
        model.train()

        # update learning rate
        lr = lr_scheduler.get(epoch - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(epoch - 1)
        # mm = 0.99
        loss = model(graph, feat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        # print(f"Epoch: {epoch}, Loss: {loss.item()}")

    model.eval()
    z1 = model.get_embed(graph, feat)
    acc = label_classification(z1, train_mask, val_mask, test_mask,
                               label, label_type, name=dataname, device=device)
    print(f" Acc: {acc['Acc']['mean']}, Std: {round(acc['Acc']['std'], 4)}")


train()

# Cora
# train({'alpha': 0.6, 'epochs': 400, 'layer': 2, 'lr': 0.005, 'p1': 0.8, 'rate': 0.5, 't': 0.2, 'w': 0.001, 'acc': 0.844})
# Photo
# train({'alpha': 1, 'epochs': 5000, 'layer': 1, 'lr': 0.001, 'p1': 0.8, 'rate': 0.5, 't': 0.2, 'w': 0.001, 'acc': 0.932+-0.3})
# PubMed
# train({'alpha': 0.7, 'epochs': 1000, 'layer': 2, 'lr': 0.005, 'p1': 0.8, 'rate': 0.1, 't': 0.2, 'w': 0.001, 'acc': 0.827})
# Com
# train({'alpha': 0.7, 'epochs': 5000, 'layer': 1, 'lr': 0.001, 'p1': 0.8, 'rate': 0.5, 't': 0.2, 'w': 0.001, 'acc': 0.898+-0.2})
# CiteSeer
# train({'alpha': 0.2, 'epochs': 500, 'layer': 2, 'lr': 0.001, 'p1': 0.8, 'rate': 0.2, 't': 0.3, 'acc': 0.708})
# CS
# train({'alpha': 0.7, 'epochs': 400, 'layer': 1, 'lr': 0.005, 'p1': 0.8, 'rate': 0.3, 't': 0.2, 'w': 0.001, 'acc': 0.927})
# arxiv
# train({'alpha': 0.4, 'epochs': 3000, 'layer': 3, 'lr': 0.005, 'p1': 0.5, 'rate': 0.15, 't': 0.2, 'w': 0.001, 'acc': 0.695})
# WikiCS
# train({'alpha': 0.7, 'epochs': 200, 'layer': 2, 'lr': 0.001, 'p1': 0.5, 'rate': 0.3, 't': 0.1, 'w': 0.001, 'acc': 0.784})
# Phy
# train({'alpha': 0.7, 'epochs': 400, 'layer': 1, 'lr': 0.001, 'p1': 0.8, 'rate': 0.1, 't': 0.2, 'w': 0.001, 'acc': 0.955})
