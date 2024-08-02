import torch
import torch.nn.functional as F
from data_processing import TEINet_embeddings_5fold, esm_embeddings_5fold
from model import GraphNet
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from arg_parser import parse_args
import numpy as np
import collections
from torch_geometric.data import Data
import random
from sklearn.model_selection import train_test_split
import yaml


seed = 18
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def compute_accuracy(preds, y_true):
    return ((preds > 0).float() == y_true).sum().item() / preds.size(0)


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



args = parse_args()

with open(args.configs_path) as file:
    configs = yaml.safe_load(file)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

data_list = TEINet_embeddings_5fold(args.configs_path)
# data_list = esm_embeddings_5fold(args.configs_path)
data_list = [data.to(device) for data in data_list]

train_data = data_list[0]
test_data = data_list[1]


model = GraphNet(num_node_features=train_data.num_node_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

margin = 4.0
epoch_decay = 0.0046
weight_decay = 0.006
aucm_optimizer = PESG(model.parameters(),
                 loss_fn=AUCMLoss(),
                 lr=args.lr,
                 momentum=0.4,
                 margin=margin,
                 device=device,
                 epoch_decay=epoch_decay,
                 weight_decay=weight_decay)


num_epochs = args.epochs
best_valid_roc = 0
best_valid_acc = 0
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    aucm_optimizer.zero_grad()

    out = model(train_data.x, train_data.edge_index)
    preds = out
    y_true = train_data.y.to(device)

    num_positive_samples = (y_true == 1).sum()
    num_negative_samples = (y_true == 0).sum()
    weight_factor = num_negative_samples.float() / num_positive_samples.float()
    pos_weight = torch.ones([y_true.size(0)],device=device) * weight_factor * args.positive_weights
    bce_loss = F.binary_cross_entropy_with_logits(preds, y_true, pos_weight=pos_weight)


    aucm_module = AUCMLoss()
    aucm_loss = aucm_module(torch.sigmoid(preds), y_true)
    total_loss = args.w_celoss * bce_loss + args.w_aucloss * aucm_loss.to(device)
    total_loss.backward()
    optimizer.step()
    aucm_optimizer.step()

    accuracy = compute_accuracy(preds, y_true)
    roc_auc = compute_auc(preds, y_true)
    aupr = compute_aupr(preds, y_true)


    # Validation part
    model.eval()
    with torch.no_grad():
        out_valid = model(test_data.x, test_data.edge_index)
        preds_valid = out_valid
        y_true_valid = test_data.y.to(device)


        valid_acc = compute_accuracy(preds_valid, y_true_valid)
        roc_auc_valid = compute_auc(preds_valid, y_true_valid)
        valid_aupr = compute_aupr(preds_valid, y_true_valid)

        if roc_auc_valid > best_valid_roc:
            best_valid_roc = roc_auc_valid
            torch.save(model.state_dict(), configs['save_model'])
    
    print("Epoch: {}/{}, Loss: {:.7f}, Train Acc: {:.4f}, Test Acc: {:.4f}, Train AUC: {:.4f}, Train APUR: {:.4f}, Test AUC: {:.4f}, Test AUPR: {:.4f}".format(epoch+1, num_epochs, total_loss.item(), accuracy, valid_acc, roc_auc, aupr, roc_auc_valid, valid_aupr))
    
# Load the best model
best_model = GraphNet(num_node_features=test_data.num_node_features).to(device)
best_model.load_state_dict(torch.load(configs['save_model']))



# Evaluate on test test_data
best_model.eval()
with torch.no_grad():
    out_test = best_model(test_data.x, test_data.edge_index)
    preds_test = out_test
    y_true_test = test_data.y.to(device)

    test_acc = compute_accuracy(preds_test, y_true_test)
    roc_auc_test = compute_auc(preds_test, y_true_test)
    test_aupr = compute_aupr(preds_test, y_true_test)

    # save results
    probabilities = torch.sigmoid(preds_test)
    binary_predictions = (probabilities > 0.5).type(torch.int).detach().cpu().numpy()
    df = pd.DataFrame({
        'prediction': binary_predictions,
        'label': y_true_test.detach().cpu().numpy().astype(int)
    })
    df.to_csv(f'results/{configs["dataset_name"]}.csv', index=False)


print("Test Acc: {:.4f}, Test AUC: {:.4f}, Test AUPR: {:.4f}".format(test_acc, roc_auc_test, test_aupr))