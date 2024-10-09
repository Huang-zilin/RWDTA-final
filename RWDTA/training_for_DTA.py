import os
import pickle
import networkx as nx
from node2vec import Node2Vec
from models.Model import *
from utils import *
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from lifelines.utils import concordance_index
import argparse


def train(model, device, train_loader, optimizer, ppi_adj, new_ppi_features, loss_fn, args, epoch):
    """
    Training function, which records the training-related logic.
    :param model: The model that we aim to train.
    :param device: The GPU device selected to train our model
    :param train_loader: The dataloader for train dataset
    :param optimizer: Adam optimizer
    :param ppi_adj: The adjacency matrix of the PPI network. Note that the adjacency matrix here is sparse, with dimensions of [2, E].
    :param new_ppi_features: The feature matrix of the PPI network and use node2vec.
    :param loss_fn: MSEloss.
    :param args: The parameter namespace object.
    :param epoch: Train epoch
    :return: None
    """
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        mol_data = data[0].to(device)
        pro_data = data[1].to(device)
        optimizer.zero_grad()
        output= model(mol_data, pro_data, ppi_adj, new_ppi_features)
        loss = loss_fn(output, mol_data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * args.batch,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def predicting(model, device, loader, ppi_adj, new_ppi_features):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            mol_data = data[0].to(device)
            pro_data = data[1].to(device)
            output = model(mol_data, pro_data, ppi_adj, new_ppi_features)

            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, mol_data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def main(args):
    dataset = args.dataset
    model_dict_ = {'RWNet': RWNet}
    modeling = model_dict_[args.model]
    model_st = modeling.__name__
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")

    df_train = pd.read_csv(f'data/{dataset}/train.csv')
    df_test = pd.read_csv(f'data/{dataset}/test.csv')
    train_smile, train_seq, train_label = list(df_train['compound_iso_smiles']), list(df_train['target_sequence']), list(df_train['affinity'])
    test_smile, test_seq, test_label = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']), list(df_test['affinity'])

    #node2vec
    if not os.path.exists(f'data/{dataset}/PPI/ppi_data_wv.emb'):
        with open(f'data/{dataset}/PPI/ppi_data.pkl', 'rb') as file_graph:
            graph_data = pickle.load(file_graph)
        adj_matrix = graph_data[0]
        G = nx.from_numpy_array(adj_matrix)
        # Generate node walking sequences
        node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
        ppi_model = node2vec.fit(window=10, min_count=1, batch_words=4)
        ppi_model.wv.save_word2vec_format(f'data/{dataset}/PPI/ppi_data_wv.emb')



    with open(f"data/{dataset}/mol_data.pkl", 'rb') as file:
        mol_data = pickle.load(file)

    wv_feature = {}
    with open(f'data/{dataset}/PPI/ppi_data_wv.emb', 'r') as file2:
        for i, line in enumerate(file2):
            if i == 0:
                continue
            else:
                temp = line.strip().split(' ')
                wv_feature[temp[0]] = [eval(i) for i in temp[1:]]
    new_ppi_features = []
    for i in range(len(wv_feature)):
        new_ppi_features.append(wv_feature[str(i)])
    new_ppi_features = np.array(new_ppi_features)
    with open(f'data/{dataset}/PPI/ppi_data.pkl', 'rb') as file3:
        ppi_adj, ppi_features, ppi_index = pickle.load(file3)
    row, col = np.diag_indices_from(ppi_adj)
    ppi_adj[row, col] = 1
    with open(f'data/{dataset}/PPI/ppi_data_wv.pkl', 'wb') as file4:
        pickle.dump((ppi_adj, new_ppi_features, ppi_index), file4)
    with open(f'data/{dataset}/PPI/ppi_data_wv.pkl', 'rb') as file5:
        ppi_adj, new_ppi_features, ppi_index = pickle.load(file5)

    ppi_adj = torch.LongTensor(np.argwhere(ppi_adj == 1).transpose(1, 0)).to(device)
    new_ppi_features = torch.Tensor(new_ppi_features).to(device)

    train_dataset = DTADataset(train_smile, train_seq, train_label, mol_data = mol_data, ppi_index = ppi_index, new_ppi_features = new_ppi_features)
    test_dataset = DTADataset(test_smile, test_seq, test_label, mol_data = mol_data, ppi_index = ppi_index, new_ppi_features = new_ppi_features)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate,num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    # training the model

    model = modeling().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    model_file_name = f'results/{dataset}/'  + f'train_{model_st}.model'
    result_file_name = f'results/{dataset}/' + f'train_{model_st}.csv'
    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, ppi_adj, new_ppi_features, loss_fn, args, epoch + 1)
        G, P = predicting(model, device, test_loader, ppi_adj, new_ppi_features)
        rm2_score = get_rm2(G, P)
        ret = [mse(G, P), concordance_index(G, P), rm2_score]
        if ret[0] < best_mse:
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
            best_epoch = epoch + 1
            best_mse = ret[0]
            best_ci = ret[1]
            best_rm2 = ret[-1]
            print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci,best_rm2:', best_mse, best_ci, best_rm2, dataset, model_st)
        else:
            print(ret[0], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci,best_rm2:', best_mse, best_ci, best_rm2, dataset, model_st)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'RWNet', choices=['RWNet'])
    parser.add_argument('--epochs', type = int, default = 2000)
    parser.add_argument('--batch', type = int, default = 512)
    parser.add_argument('--LR', type = float, default = 0.0005)
    parser.add_argument('--log_interval', type = int, default = 20)
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--dataset', type = str, default = 'davis', choices=['davis', 'kiba'])
    parser.add_argument('--num_workers', type= int, default = 0)
    # parser.add_argument('--output', type=str, default='ppi_graph.pkl',help = 'The best performance of current model')
    args = parser.parse_args()
    #torch.multiprocessing.set_start_method('spawn')
    main(args)







