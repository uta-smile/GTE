import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use. Default is 0.")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate. Default is 0.001.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs. Default is 100.")
    parser.add_argument("--w_celoss", default=1, type=float, help="CE loss 1.")
    parser.add_argument("--w_aucloss", default=0, type=float, help="auc loss 0.")
    parser.add_argument('--add_same_type_edges', action='store_true', help='Flag for adding edges of same type.')
    parser.add_argument('--dynamic_graph', action='store_true', help='dynamic_graph')
    parser.add_argument('--dynamic_ratio', default=0.05, type=float, help='Add dynamic_ratio')
    parser.add_argument('--droup_out', default=0.1, type=float, help='droup_out')
    parser.add_argument('--positive_weights', default=1, type=float, help='Add positive weights')
    parser.add_argument("--dynamic_epochs", default=30, type=int, help="Number of training dynamic_epochs.")
    parser.add_argument('--distance_threshold', type=float, default=10, help='Set the threshold for the embedding distance.')
    parser.add_argument("--configs_path", default="configs/TEINet.yml", type=str, help="Path to training data file.")
    # parser.add_argument("--split", default="RandomTCR", type=str, help="Path to training data file.")
    parser.add_argument(
        "--split",
        default="StrictTCR",
        type=str,
        choices=["RandomTCR", "StrictTCR", "UniformEpitope"],
        help="Choose split method: RandomTCR or StrictTCR or UniformEpitope."
    )
    parser.add_argument(
        "--dataset",
        default="pMTnet",
        type=str,
        choices=["McPAS", "pMTnet", "VDJdb", "TEINet"],
        help="Choose from McPAS, pMTnet, VDJdb, TEINet."
    )
    args = parser.parse_args()
    return args