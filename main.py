import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import normalize
from torch.optim import Adam
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import dgl
import networkx as nx
from model import CMAE, CosineDecayScheduler, LogReg
from torch_geometric import seed_everything
import numpy as np
import warnings
import yaml
import argparse
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from dataset import load_graph_classification_dataset, train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')


def graph_show(graph, index):
    g = nx.Graph(dgl.to_networkx(graph))
    nx.draw(g, pos=nx.kamada_kawai_layout(g),
            node_color=graph.ndata["node_labels"].numpy(),
            node_size=300, width=4, with_labels=True)
    # nx.draw(g, pos=nx.spring_layout(g), node_color=graph.ndata["node_labels"].numpy(), node_size=50)
    plt.savefig(f'./T/{index}.png', dpi=800)
    # plt.show()
    plt.close()


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    configs = configs[args.dataname]

    for k, v in configs.items():
        if "lr" in k or "w" in k:
            v = float(v)
        setattr(args, k, v)
    return args


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []

    kf = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        f1 = f1_score(y_test, preds, average="micro")
        result.append(f1)
    test_f1 = np.mean(result)
    test_std = np.std(result)

    return test_f1, test_std


def plot_embeddings(embeddings, y):
    # emb_list = []
    # for k in range(Y.shape[0]):
    #     emb_list.append(embeddings[k])

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(embeddings)

    color_idx = {}
    for i in range(y.shape[0]):
        color_idx.setdefault(y[i], [])
        color_idx[y[i]].append(i)
    plt.figure()
    # ax = Axes3D(fig)
    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, s=10, alpha=0.7)
    # plt.legend([p3, p4], ['label', 'label1'], loc='lower right', scatterpoints=1)
    # plt.savefig("./DBLP.svg", format='svg')
    plt.show()


def evaluate_cluster(embeds, y, n_label):
    Y_pred = KMeans(n_label, random_state=0).fit(embeds).predict(embeds)
    nmi = normalized_mutual_info_score(y, Y_pred)
    ari = adjusted_rand_score(y, Y_pred)
    sil = silhouette_score(embeds, Y_pred)
    ch = calinski_harabasz_score(embeds, Y_pred)
    db = davies_bouldin_score(embeds, Y_pred)
    return nmi, ari, sil, ch, db


def collate_fn(batch):
    # graphs = [x[0].add_self_loop() for x in batch]
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return batch_g, labels


seed_everything(35536)
parser = argparse.ArgumentParser(description="CMAE")
parser.add_argument("--dataname", type=str, default="MUTAG")
parser.add_argument("--cuda", type=int, default=0)
args = parser.parse_args()
args = load_best_configs(args, "config.yaml")
dataname = args.dataname

graphs, (n_feat, num_classes), Y = load_graph_classification_dataset(dataname)
train_idx = torch.arange(len(graphs))
batch_size = 256
train_sampler = SubsetRandomSampler(train_idx)
train_loader = GraphDataLoader(graphs, collate_fn=collate_fn,
                               batch_size=batch_size, shuffle=True)
eval_loader = GraphDataLoader(graphs, collate_fn=collate_fn, batch_size=batch_size,
                              shuffle=True)

device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')


def train():
    model = CMAE(n_feat, 32, args.rate, args.out_hidden, args.t, args.layer, args.alpha).to(device)
    optimizer = Adam(model.trainable_parameters(), lr=args.lr, weight_decay=args.w)
    lr_scheduler = CosineDecayScheduler(args.lr, args.warmup, args.epochs)
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, args.epochs)
    for epoch in range(1, args.epochs + 1):
        model.train()
        # update momentum
        mm = 1 - mm_scheduler.get(epoch - 1)
        # mm = 0.99
        # update learning rate
        lr = lr_scheduler.get(epoch - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for g, _ in train_loader:
            g = g.to(device)
            loss = model(g, g.ndata["attr"])
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target_network(mm)
    # torch.save(model.state_dict(), './COLLAB.pth')

    x_list = []
    y_list = []
    # model = CMAE(n_feat, 32, args.rate, args.out_hidden, args.t, args.layer, args.alpha).to(device)
    # model.load_state_dict(torch.load("./COLLAB.pth"))
    model.eval()
    # i = 0
    for g, label in eval_loader:
        g = g.to(device)
        # print(i)
        # i += 1
        # graph_show(g.cpu().remove_self_loop(), i)
        # if i == 156:
        #     loss = model(g, g.ndata["attr"])
        z1 = model.get_embed(g, g.ndata['attr'])
        y_list.append(label.numpy())
        x_list.append(z1.detach().cpu().numpy())
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    # plot_embeddings(x, y)
    test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
    print(args)
    print(f"Acc: {test_f1}, Std: {test_std}")

    # accs = []
    # for i in range(10):
    #     b_xent = torch.nn.CrossEntropyLoss()
    #     mask = train_test_split(
    #         y, seed=i, train_examples_per_class=20,
    #         val_size=0, test_size=None)
    #     idx_train, idx_test = mask['train'].astype(bool), mask['test'].astype(bool)
    #     train_embs = torch.from_numpy(x)[idx_train].to(device)
    #     test_embs = torch.from_numpy(x)[idx_test].to(device)
    #
    #     train_lbls = torch.from_numpy(y)[idx_train].to(device)
    #     test_lbls = torch.from_numpy(y)[idx_test].to(device)
    #     log = LogReg(train_embs.shape[1], train_lbls.max().cpu().numpy() + 1).to(device)
    #     opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=5e-4)
    #
    #     for _ in range(10):
    #         log.train()
    #         opt.zero_grad()
    #         logits = log(train_embs)
    #         loss = b_xent(logits, train_lbls)
    #         loss.backward()
    #         opt.step()
    #
    #     log.eval()
    #     logits = log(test_embs)
    #     preds = torch.argmax(logits, dim=1)
    #     f1 = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')  # f1 score
    #     accs.append(f1.item() * 100)
    #
    # accs = np.array(accs)
    #
    # best_acc = accs.mean().item()
    # best_std = accs.std().item()
    # if accs.mean().item() > best_acc:
    #     best_acc = accs.mean().item()
    #     best_std = accs.std().item()
    #
    # print('avg_f1: {0:.2f}, f1_std: {1:.2f}\n'.format(best_acc, best_std))


train()

# x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# DD = np.array([[78.4, 79.5, 79.4, 79.2, 79.1, 79.1, 79, 79, 78.4]])
# PROTEINS = np.array([[74.7, 74.8, 74.3, 75.65, 75.74, 75.3, 75.3, 75.1, 75.4]])
# IMDB = np.array([[75.8, 76, 75.9, 75.2, 75.4, 75.3, 74.8, 74.8, 75]])
# z = pd.DataFrame(np.concatenate([DD, PROTEINS, IMDB], axis=0),
#                  columns=x, index=['DD', 'PROTEINS', 'IMDB-B'])
# # z1 = pd.DataFrame(np.concatenate([O_I, M_I], axis=0),
# #                   columns=x, index=['CMAE', 'GraphMAE'])
# sns.set_palette("deep")
# # sns.set(style='whitegrid')
# sns.lineplot(data=z.transpose(), markers=['p', 'D', 'v'], linewidth=3, ms='10')
# # sns.lineplot(data=z1.transpose(), markers=True, linewidth=6, ms='15')
# plt.xlabel(r"$\alpha$")
# plt.ylabel("Accuracy")
# # plt.title("Graph Classification")
# plt.grid()
# plt.savefig("./M4.svg", dpi=800)
# # sns.despine()
# plt.show()
