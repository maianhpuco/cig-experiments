import torch

torch.multiprocessing.set_sharing_strategy("file_system")
import argparse
import json
import os
import sys
from torch.utils.tensorboard import SummaryWriter
import random
from torch.nn.parallel import DataParallel

sys.path.append(os.path.join("src/externals/dtfd-mil"))

from Model.Attention import Attention_Gated as Attention
from Model.Attention import Attention_with_Classifier
from utils import get_cam_1d
import torch.nn.functional as F
from Model.network import Classifier_1fc, DimReduction
import numpy as np
from utils import eval_metric
import pandas as pd
import yaml


torch.manual_seed(32)
torch.cuda.manual_seed(32)
np.random.seed(32)
random.seed(32)

data_dir = "/home/mvu9/processing_datasets/processing_camelyon16/features_fp"

def main(args):
    torch.autograd.set_detect_anomaly(True)  # Add this line for better error reporting
    epoch_step = json.loads(args.epoch_step)
    writer = SummaryWriter(os.path.join(args.log_dir, "LOG", args.name))

    in_chn = 1024

    classifier = Classifier_1fc(args.mDim, args.num_cls, args.droprate).to(
        args.device
    )
    attention = Attention(args.mDim).to(args.device)
    dimReduction = DimReduction(
        in_chn, args.mDim, numLayer_Res=args.numLayer_Res
    ).to(args.device)
    attCls = Attention_with_Classifier(
        L=args.mDim, num_cls=args.num_cls, droprate=args.droprate_2
    ).to(args.device)

    if args.isPar:
        classifier = DataParallel(classifier)
        attention = DataParallel(attention)
        dimReduction = DataParallel(dimReduction)
        attCls = DataParallel(attCls)

    ce_cri = torch.nn.CrossEntropyLoss(reduction="none").to(args.device)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_dir = os.path.join(args.log_dir, "log.txt")
    save_dir = os.path.join(args.log_dir, "best_model.pth")
    z = vars(args).copy()
    with open(log_dir, "a") as f:
        f.write(json.dumps(z))
    log_file = open(log_dir, "a")

    trainable_parameters = []
    trainable_parameters = trainable_parameters + list(classifier.parameters())
    trainable_parameters = trainable_parameters + list(attention.parameters())
    trainable_parameters = trainable_parameters + list(dimReduction.parameters())

    optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_adam1 = torch.optim.Adam(attCls.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, epoch_step, gamma=args.lr_decay_ratio)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam1, epoch_step, gamma=args.lr_decay_ratio)

    best_auc = 0
    best_epoch = -1
    test_auc = 0

    for iteration in range(args.k_start, args.k_end + 1):
        csv_path = os.path.join(args.split_folder, f'fold_{iteration}.csv')

        df = pd.read_csv(csv_path)
        train_case_ids = df["train"].dropna().tolist()
        val_case_ids = df["val"].dropna().tolist()
        test_case_ids = df["test"].dropna().tolist()

        train_labels = df["train_label"].dropna().astype(int).tolist()
        val_labels = df["val_label"].dropna().astype(int).tolist()
        test_labels = df["test_label"].dropna().astype(int).tolist()

        print_log(
            f"training slides: {len(train_case_ids)}, validation slides: {len(val_case_ids)}, test slides: {len(test_case_ids)}",
            log_file,
        )

        for ii in range(args.EPOCH):
            for param_group in optimizer_adam1.param_groups:
                curLR = param_group["lr"]
                print_log(f" current learn rate {curLR}", log_file)

            train_attention_preFeature_DTFD(
                classifier=classifier,
                dimReduction=dimReduction,
                attention=attention,
                UClassifier=attCls,
                mDATA_list=(train_case_ids, train_labels),
                ce_cri=ce_cri,
                optimizer0=optimizer_adam0,
                optimizer1=optimizer_adam1,
                epoch=ii,
                args=args,
                f_log=log_file,
                writer=writer,
                numGroup=args.numGroup,
                total_instance=args.total_instance,
                distill=args.distill_type,
            )
            print_log(f">>>>>>>>>>> Validation Epoch: {ii}", log_file)
            auc_val = test_attention_DTFD_preFeat_MultipleMean(
                classifier=classifier,
                dimReduction=dimReduction,
                attention=attention,
                UClassifier=attCls,
                mDATA_list=(val_case_ids, val_labels),
                criterion=ce_cri,
                epoch=ii,
                args=args,
                f_log=log_file,
                writer=writer,
                numGroup=args.numGroup_test,
                total_instance=args.total_instance_test,
                distill=args.distill_type,
            )
            print_log(" ", log_file)
            print_log(f">>>>>>>>>>> Test Epoch: {ii}", log_file)
            tauc = test_attention_DTFD_preFeat_MultipleMean(
                classifier=classifier,
                dimReduction=dimReduction,
                attention=attention,
                UClassifier=attCls,
                mDATA_list=(test_case_ids, test_labels),
                criterion=ce_cri,
                epoch=ii,
                args=args,
                f_log=log_file,
                writer=writer,
                numGroup=args.numGroup_test,
                total_instance=args.total_instance_test,
                distill=args.distill_type,
            )
            print_log(" ", log_file)

            if ii > int(args.EPOCH * 0.8):
                if auc_val > best_auc:
                    best_auc = auc_val
                    best_epoch = ii
                    test_auc = tauc
                    if args.isSaveModel:
                        tsave_dict = {
                            "classifier": classifier.state_dict(),
                            "dim_reduction": dimReduction.state_dict(),
                            "attention": attention.state_dict(),
                            "att_classifier": attCls.state_dict(),
                        }
                        torch.save(tsave_dict, save_dir)

                print_log(f" test auc: {test_auc}, from epoch {best_epoch}", log_file)

            scheduler0.step()
            scheduler1.step()


def test_attention_DTFD_preFeat_MultipleMean(
    mDATA_list,
    classifier,
    dimReduction,
    attention,
    UClassifier,
    epoch,
    criterion=None,
    args=None,
    f_log=None,
    writer=None,
    numGroup=3,
    total_instance=3,
    distill="MaxMinS",
):

    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    case_ids, labels = mDATA_list
    instance_per_group = total_instance // numGroup

    test_loss0 = AverageMeter()
    test_loss1 = AverageMeter()

    gPred_0 = torch.FloatTensor().to(args.device)
    gt_0 = torch.LongTensor().to(args.device)
    gPred_1 = torch.FloatTensor().to(args.device)
    gt_1 = torch.LongTensor().to(args.device)

    with torch.no_grad():

        numSlides = len(case_ids)
        numIter = numSlides // args.batch_size_v
        tIDX = list(range(numSlides))

        for idx in range(numIter):

            tidx_slide = tIDX[
                idx * args.batch_size_v : (idx + 1) * args.batch_size_v
            ]
            slide_names = [case_ids[sst] for sst in tidx_slide]
            tlabel = [labels[sst] for sst in tidx_slide]
            label_tensor = torch.LongTensor(tlabel).to(args.device)

            for tidx, tfeat in enumerate(slide_names):
                tslideName = slide_names[tidx]
                tslideLabel = label_tensor[tidx].unsqueeze(0).to(args.device)

                full_path = os.path.join(data_dir, 'pt_files', f"{tslideName}.pt")
                features = torch.load(full_path, weights_only=True)
                tfeat = features.to(args.device)

                midFeat = dimReduction(tfeat)


                AA = attention(midFeat, isNorm=False).squeeze(0)  ## N

                allSlide_pred_softmax = []

                for jj in range(args.num_MeanInference):

                    feat_index = list(range(tfeat.shape[0]))
                    random.shuffle(feat_index)
                    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                    index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                    slide_d_feat = []
                    slide_sub_preds = []
                    slide_sub_labels = []

                    for tindex in index_chunk_list:
                        slide_sub_labels.append(tslideLabel)
                        idx_tensor = torch.LongTensor(tindex).to(args.device)
                        tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                        tAA = AA.index_select(dim=0, index=idx_tensor)
                        tAA = torch.softmax(tAA, dim=0)
                        tattFeats = torch.einsum("ns,n->ns", tmidFeat, tAA)  ### n x fs
                        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(
                            0
                        )  ## 1 x fs

                        tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                        slide_sub_preds.append(tPredict)

                        patch_pred_logits = get_cam_1d(
                            classifier, tattFeats.unsqueeze(0)
                        ).squeeze(
                            0
                        )  ###  cls x n
                        patch_pred_logits = torch.transpose(
                            patch_pred_logits, 0, 1
                        )  ## n x cls
                        patch_pred_softmax = torch.softmax(
                            patch_pred_logits, dim=1
                        )  ## n x cls

                        _, sort_idx = torch.sort(
                            patch_pred_softmax[:, -1], descending=True
                        )

                        if distill == "MaxMinS":
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx_min = sort_idx[-instance_per_group:].long()
                            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == "MaxS":
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx = topk_idx_max
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == "AFS":
                            slide_d_feat.append(tattFeat_tensor)

                    slide_d_feat = torch.cat(slide_d_feat, dim=0)
                    slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                    slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

                    gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
                    gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)
                    loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
                    test_loss0.update(loss0.item(), numGroup)

                    gSlidePred = UClassifier(slide_d_feat)
                    print("--test pred", gSlidePred)
                    allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))

                allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
                allSlide_pred_softmax = torch.mean(
                    allSlide_pred_softmax, dim=0
                ).unsqueeze(0)
                gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
                gt_1 = torch.cat([gt_1, tslideLabel], dim=0)

                loss1 = F.nll_loss(allSlide_pred_softmax, tslideLabel)
                test_loss1.update(loss1.item(), 1)

    gPred_0 = torch.softmax(gPred_0, dim=1)
    gPred_0 = gPred_0[:, -1]
    gPred_1 = gPred_1[:, -1]

    macc_0, mprec_0, mrecal_0, mspec_0, mF1_0, auc_0 = eval_metric(gPred_0, gt_0)
    macc_1, mprec_1, mrecal_1, mspec_1, mF1_1, auc_1 = eval_metric(gPred_1, gt_1)

    print_log(
        f"  First-Tier acc {macc_0}, precision {mprec_0}, recall {mrecal_0}, specificity {mspec_0}, F1 {mF1_0}, AUC {auc_0}",
        f_log,
    )
    print_log(
        f"  Second-Tier acc {macc_1}, precision {mprec_1}, recall {mrecal_1}, specificity {mspec_1}, F1 {mF1_1}, AUC {auc_1}",
        f_log,
    )

    writer.add_scalar(f"auc_0 ", auc_0, epoch)
    writer.add_scalar(f"auc_1 ", auc_1, epoch)

    return auc_1


def train_attention_preFeature_DTFD(
    mDATA_list,
    classifier,
    dimReduction,
    attention,
    UClassifier,
    optimizer0,
    optimizer1,
    epoch,
    ce_cri=None,
    args=None,
    f_log=None,
    writer=None,
    numGroup=3,
    total_instance=3,
    distill="MaxMinS",
):

    case_ids, labels = mDATA_list

    classifier.train()
    dimReduction.train()
    attention.train()
    UClassifier.train()

    instance_per_group = total_instance // numGroup

    Train_Loss0 = AverageMeter()
    Train_Loss1 = AverageMeter()

    numSlides = len(case_ids)
    numIter = numSlides // args.batch_size

    tIDX = list(range(numSlides))
    random.shuffle(tIDX)


    for idx in range(numIter):

        tidx_slide = tIDX[idx * args.batch_size : (idx + 1) * args.batch_size]

        tslide_name = [case_ids[sst] for sst in tidx_slide]
        tlabel = [labels[sst] for sst in tidx_slide]
        label_tensor = torch.LongTensor(tlabel).to(args.device)

        for tidx, (tslide, slide_idx) in enumerate(zip(tslide_name, tidx_slide)):
            tslideLabel = label_tensor[tidx].unsqueeze(0)

            slide_pseudo_feat = []
            slide_sub_preds = []
            slide_sub_labels = []

            full_path = os.path.join(data_dir, 'pt_files', f"{tslide}.pt")
            features = torch.load(full_path, weights_only=True)

            tfeat_tensor = features
            tfeat_tensor = tfeat_tensor.to(args.device)

            feat_index = list(range(tfeat_tensor.shape[0]))
            random.shuffle(feat_index)
            index_chunk_list = np.array_split(np.array(feat_index), numGroup)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list]

            for tindex in index_chunk_list:
                slide_sub_labels.append(tslideLabel)

                # Create fresh tensor with no history
                with torch.no_grad():
                    indices = torch.LongTensor(tindex).to(args.device)
                    subFeat_tensor_raw = tfeat_tensor.index_select(0, indices).detach()

                # Create completely new tensor
                subFeat_tensor = (
                    subFeat_tensor_raw.clone().detach().requires_grad_(True)
                )

                # Forward pass through dimReduction
                with torch.enable_grad():
                    tmidFeat = dimReduction(subFeat_tensor)

                # Get attention scores without gradient tracking
                with torch.no_grad():
                    tAA = attention(tmidFeat.detach())
                    if tAA.dim() > 1 and tAA.size(0) == 1:
                        tAA = tAA.squeeze(0)

                # Continue computation with fresh tensors
                with torch.enable_grad():
                    # Create a completely fresh copy of tmidFeat
                    tmidFeat_fresh = tmidFeat.detach().clone().requires_grad_(True)
                    tattFeats = torch.einsum("ns,n->ns", tmidFeat_fresh, tAA)
                    tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)
                    tPredict = classifier(tattFeat_tensor)
                    slide_sub_preds.append(tPredict)

                # CAM computation with separate tensor
                with torch.enable_grad():
                    tattFeats_cam = tattFeats.detach().clone().requires_grad_(True)
                    patch_pred_logits = get_cam_1d(
                        classifier, tattFeats_cam.unsqueeze(0)
                    ).squeeze(0)
                    patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)
                    patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)

                _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx_min = sort_idx[-instance_per_group:].long()
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

                MaxMin_inst_feat = tmidFeat.index_select(
                    dim=0, index=topk_idx
                )  ##########################
                max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
                af_inst_feat = tattFeat_tensor

                if distill == "MaxMinS":
                    slide_pseudo_feat.append(MaxMin_inst_feat)
                elif distill == "MaxS":
                    slide_pseudo_feat.append(max_inst_feat)
                elif distill == "AFS":
                    slide_pseudo_feat.append(af_inst_feat)

            slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs

            ## optimization for the first tier
            slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  ### numGroup x fs
            slide_sub_labels = torch.cat(slide_sub_labels, dim=0)  ### numGroup
            loss0 = ce_cri(slide_sub_preds, slide_sub_labels).mean()
            optimizer0.zero_grad()
            loss0.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                dimReduction.parameters(), args.grad_clipping
            )
            torch.nn.utils.clip_grad_norm_(attention.parameters(), args.grad_clipping)
            torch.nn.utils.clip_grad_norm_(
                classifier.parameters(), args.grad_clipping
            )
            optimizer0.step()

            ## optimization for the second tier
            gSlidePred = UClassifier(slide_pseudo_feat)
            print("--train pred", gSlidePred)
            loss1 = ce_cri(gSlidePred, tslideLabel).mean()
            optimizer1.zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(
                UClassifier.parameters(), args.grad_clipping
            )
            optimizer1.step()

            Train_Loss0.update(loss0.item(), numGroup)
            Train_Loss1.update(loss1.item(), 1)

        if idx % args.train_show_freq == 0:
            tstr = "epoch: {} idx: {}".format(epoch, idx)
            tstr += f" First Loss : {Train_Loss0.avg}, Second Loss : {Train_Loss1.avg} "
            print_log(tstr, f_log)

    writer.add_scalar(f"train_loss_0 ", Train_Loss0.avg, epoch)
    writer.add_scalar(f"train_loss_1 ", Train_Loss1.avg, epoch)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        cur_sum = self.sum
        cur_count = self.count
        self.sum = cur_sum + val * n
        self.count = cur_count + n
        self.avg = self.sum / self.count


def print_log(tstr, f):
    # with open(dir, 'a') as f:
    f.write("\n")
    f.write(tstr)
    print(tstr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="abc")
    parser.add_argument("--name", default="abc", type=str)
    parser.add_argument("--num_epochs", default=200, type=int)
    parser.add_argument("--epoch_step", default="[100]", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--isPar", default=False, type=bool)
    parser.add_argument("--log_dir", default="./debug_log", type=str)  ## log file path
    parser.add_argument("--config", default="./configs_simea/dtfd_camelyon16.yaml", type=str)
    parser.add_argument("--train_show_freq", default=40, type=int)
    parser.add_argument("--droprate", default="0", type=float)
    parser.add_argument("--droprate_2", default="0", type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--lr_decay_ratio", default=0.2, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--batch_size_v", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--num_cls", default=2, type=int)
    parser.add_argument("--numGroup", default=4, type=int)
    parser.add_argument("--total_instance", default=4, type=int)
    parser.add_argument("--numGroup_test", default=4, type=int)
    parser.add_argument("--total_instance_test", default=4, type=int)
    parser.add_argument("--mDim", default=512, type=int)
    parser.add_argument("--grad_clipping", default=5, type=float)
    parser.add_argument("--isSaveModel", action="store_false")
    parser.add_argument("--debug_DATA_dir", default="", type=str)
    parser.add_argument("--numLayer_Res", default=0, type=int)
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument("--num_MeanInference", default=1, type=int)
    parser.add_argument("--distill_type", default="AFS", type=str)  ## MaxMinS, MaxS, AFS
    parser.add_argument("--k_start", default=1, type=int)
    parser.add_argument("--k_end", default=5, type=int)
    parser.add_argument("--split_folder", default="", type=str)
    parser.add_argument("--data_dir", default="", type=str)

    args = parser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Merge YAML config into args
    for key, value in config.items():
        setattr(args, key, value)
    main(args)
