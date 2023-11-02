import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch.nn as nn
import argparse



class GraphNet(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1):
        super(GraphNet, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.Linear(mlp_hidden_channels, num_classes)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        return edge_prediction.view(-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=1, type=int, help="GPU id to use. Default is 0.")
    parser.add_argument(
        "--split",
        default="RandomTCR",
        type=str,
        choices=["RandomTCR", "StrictTCR"],
        help="Choose split method: RandomTCR or StrictTCR."
    )
    parser.add_argument(
        "--dataset",
        default="pMTnet",
        type=str,
        choices=["McPAS", "pMTnet", "VDJdb", "TEINet"],
        help="Choose from McPAS, pMTnet, VDJdb, TEINet."
    )
    
    return parser.parse_args()


def compute_aupr(preds, y_true):
    probs = torch.sigmoid(preds)
    probs_numpy = probs.detach().cpu().numpy()
    y_true_numpy = y_true.detach().cpu().numpy()
    return average_precision_score(y_true_numpy, probs_numpy)


def compute_auc(preds, y_true):
    probs = torch.sigmoid(preds)
    y_true_numpy = y_true.detach().cpu().numpy()
    probs_numpy = probs.detach().cpu().numpy()
    return roc_auc_score(y_true_numpy, probs_numpy)


def get_test_data(test_path,embedding_path):

    with open(embedding_path, 'rb') as f:
        embedding_dict = pickle.load(f)

    node_index = {} 
    num_nodes = 0
    edge_list = []
    X = []
    y_list = []
    data = pd.read_csv(test_path)
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


    return Data(x=X, edge_index=edge_index, y=y, num_nodes=num_nodes)



args = parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
dataset = args.dataset
split = args.split

print(f"You chose the dataset: {dataset}")
print(f"The split method is: {split}")



for i in range(5):
    train_folds = ''.join([str(j) for j in range(5) if j != i])
    if dataset == "TEINet" and split == "RandomTCR":
        # Using the pre-processed RandomTCR data provided by the TEINet baseline.
        file_path = f"processed_data/{dataset}/{split}/test_fold_{i}_random.csv"
    else:
        file_path = f"processed_data/{dataset}/{split}/{split}_fold_{i}.csv"

    model_path = f"models/{dataset}/{split}/{dataset}_{train_folds}_{i}.pth"
    embedding_path = f"models/{dataset}/{dataset}_embeddings.pkl"

    test_data = get_test_data(file_path, embedding_path).to(device)
    test_data_df = pd.read_csv(file_path)

    GTE = GraphNet(num_node_features=test_data.num_node_features).to(device)
    GTE.load_state_dict(torch.load(model_path))
    GTE.eval()

    with torch.no_grad():
        preds_test = GTE(test_data.x, test_data.edge_index)
        y_true_test = test_data.y.to(device)

        roc_auc_test = compute_auc(preds_test, y_true_test)
        test_aupr = compute_aupr(preds_test, y_true_test)

        # save results
        probabilities = torch.sigmoid(preds_test)
        binary_predictions = (probabilities > 0.5).type(torch.int).detach().cpu().numpy()
        df = pd.DataFrame({
            "CDR3.beta":test_data_df["CDR3.beta"].values,
            "Epitope":test_data_df["Epitope"].values,
            'Label': y_true_test.detach().cpu().numpy().astype(int),
            'Prediction': probabilities.detach().cpu().numpy(),
            
        })
        df.to_csv(f'results/{dataset}_{split}_{train_folds}_{i}.csv', index=False)

    print(f"Fold: {i}, AUC: {roc_auc_test:.4f}, AUPR: {test_aupr:.4f}")

