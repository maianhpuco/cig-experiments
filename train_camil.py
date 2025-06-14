import pandas as pd
import sys
import os

sys.path.append(os.path.join("src/externals/camil-iclr-clone")) 

from args import parse_args
from models.camil import CAMIL
import os
from flushed_print import print
import tensorflow as tf
import numpy as np
import gc
import random
from dataset_utils.camelyon16 import Camelyon16Dataset
import yaml

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DISABLE_SPARSE_SOFTMAX_XENT_WITH_LOGITS_OP_DETERMINISM_EXCEPTIONS"] = "1"

def create_dataset(col_case, col_label):
    case_ids = df[col_case].dropna().tolist()
    labels = df[col_label].dropna().astype(int).tolist()
    return case_ids, labels

if __name__ == "__main__":
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args.feature_path = config['feature_path']
    args.save_dir = config['save_dir']
    args.split_folder = config['split_folder']
    args.test = config['test']
    args.epochs = config['epochs']
    args.batch_size = config['batch_size']
    args.init_lr = config['init_lr']
    args.drop_out = config['drop_out']
    args.bag_loss = config['bag_loss']
    args.model_size = config['model_size']
    args.subtyping = config['subtyping']
    args.B = config['B']
    args.n_classes = config['n_classes']
    args.seed = config['seed']
    args.log_data = config['log_data']
    args.testing = config['testing']
    args.early_stopping = config['early_stopping']
    args.opt = config['opt']
    

    print("Called with args:", args)

    adj_dim = None
    set_seed(12321)

    acc = []
    recall = []
    f_score = []
    auc = []
    precision = []

    i = 1
    split_csv_path = os.path.join(args.split_folder, f'fold_{i}.csv')
    fold_id = str(i)

    df = pd.read_csv(split_csv_path)

    # Create train, validation and test bags
    train_cases, train_labels = create_dataset('train', 'train_label')
    val_cases, val_labels = create_dataset('val', 'val_label')
    test_cases, test_labels = create_dataset('test', 'test_label')

    # Convert to h5 file paths
    train_bags = [os.path.join(args.feature_path, case + ".h5") for case in train_cases]
    val_bags = [os.path.join(args.feature_path, case + ".h5") for case in val_cases]
    test_bags = [os.path.join(args.feature_path, case + ".h5") for case in test_cases]

    args.label_maps = {}

    for (train_bag, train_label) in zip(train_bags, train_labels):
        args.label_maps[train_bag] = train_label

    for (val_bag, val_label) in zip(val_bags, val_labels):
        args.label_maps[val_bag] = val_label

    for (test_bag, test_label) in zip(test_bags, test_labels):
        args.label_maps[test_bag] = test_label

    if not args.test:
        train_net = CAMIL(args)
        train_net.train(train_bags, Camelyon16Dataset, fold_id, val_bags, args)
    else:
        test_net = CAMIL(args)
        test_acc, test_auc = test_net.predict(test_bags, Camelyon16Dataset, fold_id, args, test_model=test_net.model)
