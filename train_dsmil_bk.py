import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, accuracy_score, hamming_loss
from sklearn.model_selection import KFold
from collections import OrderedDict
import json
from tqdm import tqdm
from src.datasets.classification.camelyon16 import return_splits_custom 

sys.path.append(os.path.join("src/externals/dsmil-wsi"))

import dsmil as mil
import yaml

'''
Note: DSMIl model was fromm mil.MILNet(i_classifier, b_classifier).cuda() 
'''
def train(args, data_loader, label_dict, milnet, criterion, optimizer):
    milnet.train()

    total_loss = 0
    Tensor = torch.cuda.FloatTensor
    for batch_idx, (data, label) in enumerate(data_loader):
        optimizer.zero_grad()

        label = label.astype(int)

        bag_label = torch.zeros(args.num_classes, dtype=torch.float32).cuda()
        bag_label[label] = 1
        bag_label = bag_label.unsqueeze(0)
        bag_feats = data.cuda()
        bag_feats = dropout_patches(bag_feats, 1-args.dropout_patch)
        bag_feats = bag_feats.view(-1, args.feats_size)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
        # print('ins_prediction:', ins_prediction)
        # print('bag_prediction:', bag_prediction) 
        max_prediction, _ = torch.max(ins_prediction, 0)        
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (batch_idx, len(data_loader), loss.item()))
    return total_loss / len(data_loader)

def dropout_patches(feats, p):
    num_rows = feats.size(0)
    num_rows_to_select = int(num_rows * p)
    random_indices = torch.randperm(num_rows)[:num_rows_to_select]
    selected_rows = feats[random_indices]
    return selected_rows

def test(args, data_loader, label_dict, milnet, criterion, thresholds=None, return_predictions=False):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(data_loader):
            label = label.astype(int)
            bag_label = torch.zeros(args.num_classes, dtype=torch.float32).cuda()
            bag_label[label] = 1
            bag_label = bag_label.unsqueeze(0)
            bag_feats = data.cuda()
            bag_feats = dropout_patches(bag_feats, 1-args.dropout_patch)
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)  
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (batch_idx, len(data_loader), loss.item()))
            test_labels.extend([bag_label.squeeze().cpu().numpy().astype(int)])
            if args.average:
                test_predictions.extend([(torch.sigmoid(max_prediction)+torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else: test_predictions.extend([torch.sigmoid(bag_prediction).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if thresholds: thresholds_optimal = thresholds
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_labels)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_labels)
    
    if return_predictions:
        return total_loss / len(test_labels), avg_score, auc_value, thresholds_optimal, test_predictions, test_labels
    return total_loss / len(test_labels), avg_score, auc_value, thresholds_optimal

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=-1)
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        # c_auc = roc_auc_score(label, prediction)
        try:
            c_auc = roc_auc_score(label, prediction)
            print("ROC AUC score:", c_auc)
        except ValueError as e:
            if str(e) == "Only one class present in y_true. ROC AUC score is not defined in that case.":
                print("ROC AUC score is not defined when only one class is present in y_true. c_auc is set to 1.")
                c_auc = 1
            else:
                raise e

        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs):
    if args.dataset.startswith('TCGA-lung'):
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, normal: %.4f, tumor: %.4f' % 
                (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
    else:
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 

def get_current_score(avg_score, aucs):
    current_score = (sum(aucs) + avg_score)/2
    return current_score

def save_model(args, fold, run, save_path, model, thresholds_optimal):
    # Construct the filename including the fold number
    save_name = os.path.join(save_path, f'fold_{fold}_{run+1}.pth')
    torch.save(model.state_dict(), save_name)
    print_save_message(args, save_name, thresholds_optimal)
    file_name = os.path.join(save_path, f'fold_{fold}_{run+1}.json')
    with open(file_name, 'w') as f:
        json.dump([float(x) for x in thresholds_optimal], f)

def print_save_message(args, save_name, thresholds_optimal):
    if args.dataset.startswith('TCGA-lung'):
        print('Best model saved at: ' + save_name + ' Best thresholds: normal %.4f, tumor %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
    else:
        print('Best model saved at: ' + save_name)
        print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))

def main(args):
    def apply_sparse_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_model(args):
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
        milnet.apply(lambda m: apply_sparse_init(m))
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
        return milnet, criterion, optimizer, scheduler
    
    import time
    start_time = time.time()
    label_dict = {'normal': 0, 'tumor': 1}
    # bags_path = bags_path.sample(n=50, random_state=42)
    fold_results = []
    
    
    all_save_paths = []
    for iteration in range(args.k_start, args.k_end + 1):
        
        os.makedirs(args.save_path, exist_ok=True)
        save_path = os.path.join(args.save_path, f"fold_{iteration}")
        os.makedirs(save_path, exist_ok=True)
        
        # save_path = os.path.join(args.save_path, datetime.date.today().strftime("%Y%m%d"))
        all_save_paths.append(save_path) 
       
        print("---------------------") 
        print(f"Saving results to: {save_path}")
        
        run = len(glob.glob(os.path.join(save_path, '*.pth'))) 
        print(f"Starting iteration {iteration}.")
        milnet, criterion, optimizer, scheduler = init_model(args)

        split_csv_path = os.path.join(args.split_folder, f'fold_{iteration}.csv')

        train_dataset, val_dataset, test_dataset = return_splits_custom(
            csv_path=split_csv_path,
            data_dir=args.data_dir,
            label_dict= label_dict,  # This won't affect direct labels
            seed=42,
            print_info=False
        )
        print("---------------------")
        print(f"Train len: {len(train_dataset)} | Val len: {len(val_dataset)} | Test len: {len(test_dataset)}")
        

        fold_best_score = 0
        best_ac = 0
        best_auc = 0
        counter = 0

        for epoch in range(1, args.num_epochs + 1):
            counter += 1
            train_loss_bag = train(args, train_dataset, label_dict, milnet, criterion, optimizer) # iterate all bags
            test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, val_dataset, label_dict, milnet, criterion)
            
            print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs)
            scheduler.step()

            current_score = get_current_score(avg_score, aucs)
            if current_score > fold_best_score:
                counter = 0
                fold_best_score = current_score
                best_ac = avg_score
                best_auc = aucs
                
                save_model(args, iteration, run, save_path, milnet, thresholds_optimal)
                
                best_model = copy.deepcopy(milnet)
            if counter > args.stop_epochs: break
         
        
        test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, test_dataset, label_dict, best_model, criterion)
        fold_results.append((best_ac, best_auc))
    mean_ac = np.mean(np.array([i[0] for i in fold_results]))
    mean_auc = np.mean(np.array([i[1] for i in fold_results]), axis=0)
    # Print mean and std deviation for each class
    print(f"Final results: Mean Accuracy: {mean_ac}")
    for i, mean_score in enumerate(mean_auc):
        print(f"Class {i}: Mean AUC = {mean_score:.4f}")
    elapsed = time.time() - start_time
    
    print(f"\n All folds complete. Results saved under: {all_save_paths}")
    print(f" Total run time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n") 
    
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    # parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    # parser.add_argument('--feats_size', default=1024, type=int, help='Dimension of the feature size [512]')
    # parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0001]')
    # # parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [100]')
    # # parser.add_argument('--stop_epochs', default=10, type=int, help='Skip remaining epochs if training has not improved after N epochs [10]')
    # parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    # parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay [1e-3]')
    # parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    # parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    # parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    # parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    # parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    # parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    # parser.add_argument('--average', type=bool, default=False, help='Average the score of max-pooling and bag aggregating')
    # parser.add_argument('--eval_scheme', default='5-fold-cv', type=str, help='Evaluation scheme [5-fold-cv | 5-fold-cv-standalone-test | 5-time-train+valid+test ]')
    # parser.add_argument('--k_start', default=1, type=int, help='Start fold number')
    # parser.add_argument('--k_end', default=1, type=int, help='End fold number')
    # parser.add_argument("--config", default="./configs_simea/dsmil_camelyon16.yaml", type=str)
    # parser.add_argument("--split_folder", default="/home/mvu9/processing_datasets/processing_camelyon16/splits_csv", type=str)
    # parser.add_argument("--data_dir", default="/home/mvu9/processing_datasets/processing_camelyon16/features_fp", type=str)
    # # parser.add_argument("--save_path", default="/home/mvu9/processing_datasets/processing_camelyon16/results", type=str)
    # parser.add_argument("--save_path", default="/home/mvu9/processing_datasets/processing_camelyon16/dsmil_camelyon16_results", type=str)

    # args = parser.parse_args()
    # print(args.eval_scheme)

    # gpu_ids = tuple(args.gpu_index)
    # os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)

    # config_path = args.config
    # with open(config_path, 'r') as f:
    #     config = yaml.safe_load(f)

    # # Merge YAML config into args
    # for key, value in config.items():
    #     setattr(args, key, value)
        
    # main(args)
    
if __name__ == '__main__':
    import yaml
    parser = argparse.ArgumentParser(description='Train DSMIL using a full YAML config.')
    parser.add_argument('--num_epochs', type=int, help='Number of total training epochs [100]') 
    parser.add_argument('--k_start', default=1, type=int, help='Start fold number')
    parser.add_argument('--k_end', default=1, type=int, help='End fold number')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file.')
    
    args = parser.parse_args()

    # Load full config from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Convert config dictionary to an argparse.Namespace object
    for key, val in config.items():
        if key == 'paths':
            args.paths = val
        else:
            setattr(args, key, val)
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_index))

    # Start training
    main(args) 