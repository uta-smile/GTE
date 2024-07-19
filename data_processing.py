import torch
import pickle
import numpy as np
from torch_geometric.data import Data
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from arg_parser import parse_args
import os
import yaml


def TEINet_embeddings_5fold(config_path):
    args = parse_args()

    with open(config_path) as file:
        config = yaml.safe_load(file)

    with open(config["embeddings_path"], 'rb') as f:
        embedding_dict = pickle.load(f)

    train_file_list = config[args.split]['train_data']['file_list']
    test_file_list = config[args.split]['test_data']['file_list']
    file_path = config['path']

    train_data = []
    for file_name in train_file_list:
        data = pd.read_csv(os.path.join(file_path, file_name))
        train_data.append(data)
    train_data = pd.concat(train_data)


    test_data = []
    for file_name in test_file_list:
        data = pd.read_csv(os.path.join(file_path, file_name))
        test_data.append(data)
    test_data = pd.concat(test_data) 

    all_data = []
    for data in [train_data, test_data]:
        node_index = {} 
        num_nodes = 0
        edge_list = []
        X = []
        y_list = []
        for _, row in data.iterrows():
            label = float(row["Label"])
            nodes = [row["Epitope"], row["CDR3.beta"]]
            for node in nodes:
                if node not in node_index:
                    node_index[node] = num_nodes
                    num_nodes += 1
                    X.append(embedding_dict[node])
            y_list.append(label)
            edge_list.append((node_index[nodes[0]], node_index[nodes[1]]))

        
        X = torch.tensor(np.array(X), dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        y = torch.tensor(y_list, dtype=torch.float)

        all_data.append(Data(x=X, edge_index=edge_index, y=y, num_nodes=num_nodes))

    return all_data




def esm_embeddings_5fold(config_path):
    args = parse_args()

    with open(config_path) as file:
        config = yaml.safe_load(file)

    # with open(config["embeddings_path"], 'rb') as f:
    #     embedding_dict = pickle.load(f)

    train_file_list = config[args.split]['train_data']['file_list']
    test_file_list = config[args.split]['test_data']['file_list']
    file_path = config['path']

    train_data = []
    for file_name in train_file_list:
        data = pd.read_csv(os.path.join(file_path, file_name))
        train_data.append(data)
    train_data = pd.concat(train_data)


    test_data = []
    for file_name in test_file_list:
        data = pd.read_csv(os.path.join(file_path, file_name))
        test_data.append(data)
    test_data = pd.concat(test_data) 

    all_data = []
    for data in [train_data, test_data]:
        node_index = {} 
        num_nodes = 0
        edge_list = []
        X = []
        y_list = []
        for _, row in data.iterrows():
            label = float(row["Label"])
            nodes = [row["Epitope"], row["CDR3.beta"]]
            for node in nodes:
                if node not in node_index:
                    node_index[node] = num_nodes
                    num_nodes += 1
                    X.append(torch.load(f'{config["embeddings_path"]}/{node}.pt')["mean_representations"][36])
            y_list.append(label)
            edge_list.append((node_index[nodes[0]], node_index[nodes[1]]))

        
        X = torch.tensor(np.array(X), dtype=torch.float)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        y = torch.tensor(y_list, dtype=torch.float)

        all_data.append(Data(x=X, edge_index=edge_index, y=y, num_nodes=num_nodes))

    return all_data
