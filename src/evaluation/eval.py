import torch
import torch.nn as nn
import numpy as np
import csv
import argparse
import os
os.sys.path.append(os.path.dirname(os.path.abspath('..')))
from util import model_utils
from util.test_methods import PICTestFunctions as PIC
from util.test_methods import RISETestFunctions as RISE
from util.attribution_methods import saliencyMethods as attribution
from util.attribution_methods import GIGBuilder as GIG_Builder
from util.attribution_methods import AGI as AGI

model = None
'''
- Data Loading: Replaced Image.open with torch.load for .pt files and np.load for .np files.
- Normalization: Removed ImageNet normalization since features are not images. If your model requires normalized features, you may need to add standardization.
- Saliency Map: Aggregated across the feature dimension (np.abs(np.sum(saliency_map, axis=1))) to get a 1D saliency score per patch.
- Transforms: Removed image-specific transforms (resize, crop, transform_IG) as they are not needed for features.
- Random Mask: Adapted to use num_patches instead of img_hw for the PIC random mask.
- Baseline for Insertion: Replaced Gaussian blur with a mean feature vector baseline (mean_feature).
- File Naming: Changed references from "images" to "WSIs" for clarity.
- Model Input: The model now takes [1, N, 512] tensors instead of [1, 3, H, W].
'''

def run_and_save_tests(num_patches, random_mask, saliency_thresholds, image_count, function, function_steps, batch_size, model, model_name, deletion, insertion, device, feature_path):
    if function == "AGI":
        label = function + "_"
    else:
        label = function + "_" + str(function_steps) + "_steps_"

    label = label + str(image_count) + "_wsis_"

    # Track WSIs that are classified correctly
    correctly_classified = np.loadtxt("../../util/class_maps/WSI/correctly_classified_" + model_name + ".txt").astype(np.int64)

    num_classes = 1000  # Adjust based on your dataset
    wsis_per_class = int(np.ceil(image_count / num_classes))
    classes_used = [0] * num_classes

    fields = ["attr", "SIC", "AIC", "Ins", "Del"]
    scores = [function, 0, 0, 0, 0]

    wsis = sorted(os.listdir(feature_path))
    wsis_used = 0

    for wsi in wsis:
        if wsis_used == image_count:
            print("method finished")
            break

        # Extract WSI number from filename (assuming format like "WSI_00000001.pt")
        wsi_num = int((wsi.split("_")[1]).split(".")[0]) - 1
        if correctly_classified[wsi_num] == 0:
            continue

        # Load features and saliency scores
        feature_tensor = torch.load(os.path.join(feature_path, wsi))  # Shape: [N, 512]
        saliency_file = os.path.join(feature_path, wsi.replace(".pt", ".np"))
        saliency_map = np.load(saliency_file)  # Shape: [N,]

        num_patches = feature_tensor.shape[0]
        if len(saliency_map) != num_patches:
            continue

        feature_tensor = torch.unsqueeze(feature_tensor, 0)  # Shape: [1, N, 512]

        target_class = model_utils.getClass(feature_tensor, model, device)

        if classes_used[target_class] == wsis_per_class:
            continue
        else:
            classes_used[target_class] += 1

        print(model_name + " Function " + function + ", WSI: " + wsi)

        # Compute attribution map
        if function == "IG":
            integrated_gradients = IntegratedGradients(model)
            attr = integrated_gradients.attribute(feature_tensor.to(device), baselines=0, target=target_class, n_steps=function_steps, internal_batch_size=batch_size)
            saliency_map = attr.squeeze().detach().cpu().numpy()  # Shape: [N, 512]
        elif function == "LIG":
            saliency_map = attribution.IG(feature_tensor, model, function_steps, batch_size, 0.9, 0, device, target_class)
            saliency_map = saliency_map.squeeze().detach().cpu().numpy()
        elif function == "IDG":
            saliency_map = attribution.IDG(feature_tensor, model, function_steps, batch_size, 0, device, target_class)
            saliency_map = saliency_map.squeeze().detach().cpu().numpy()
        elif function == "GIG":
            call_model_args = {'class_idx_str': target_class.item()}
            guided_ig = GIG_Builder.GuidedIG()
            baseline = torch.zeros_like(feature_tensor)
            gig = guided_ig.GetMask(feature_tensor, model, device, GIG_Builder.call_model_function, call_model_args, x_baseline=baseline, x_steps=50, max_dist=1.0, fraction=0.5)
            saliency_map = gig.squeeze().detach().cpu().numpy()
        elif function == "AGI":
            # AGI requires model-specific modifications; assuming feature-based AGI
            epsilon = 0.05
            max_iter = 20
            topk = 1
            selected_ids = range(0, 999, int(1000 / topk))
            agi_features = feature_tensor.cpu().numpy()  # Shape: [1, N, 512]
            example = AGI.test(model, device, agi_features, epsilon, topk, selected_ids, max_iter)
            AGI_map = example[2]
            if type(AGI_map) is not np.ndarray:
                print("AGI failure, skipping WSI")
                classes_used[target_class] -= 1
                continue
            saliency_map = AGI_map

        # Aggregate saliency map across feature dimensions
        saliency_map = np.abs(np.sum(saliency_map, axis=1))  # Shape: [N,]

        if np.sum(saliency_map) == 0:
            print("Skipping WSI due to 0 attribution")
            classes_used[target_class] -= 1
            continue

        # Compute PIC metrics
        sic_score = PIC.compute_pic_metric(feature_tensor.squeeze().cpu().numpy(), saliency_map, random_mask, saliency_thresholds, 0, model, device)
        aic_score = PIC.compute_pic_metric(feature_tensor.squeeze().cpu().numpy(), saliency_map, random_mask, saliency_thresholds, 1, model, device)

        if sic_score == 0 or aic_score == 0:
            print("WSI: " + wsi + " thrown out due to 0 score")
            classes_used[target_class] -= 1
            continue

        scores[1] += sic_score.auc
        scores[2] += aic_score.auc

        # Compute insertion and deletion
        _, ins_sum = insertion.single_run(feature_tensor, saliency_map, device, batch_size)
        _, del_sum = deletion.single_run(feature_tensor, saliency_map, device, batch_size)
        scores[3] += RISE.auc(ins_sum)
        scores[4] += RISE.auc(del_sum)

        wsis_used += 1
        print("Total used: " + str(wsis_used) + " / " + str(image_count))

    for i in range(1, len(scores)):
        scores[i] /= wsis_used
        scores[i] = round(scores[i], 3)

    folder = "../test_results/" + model_name + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    label = function + "_" + str(image_count) + "_wsis"
    with open(folder + label + ".csv", 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerow(scores)

def main(FLAGS):
    device = 'cuda:' + str(FLAGS.cuda_num) if torch.cuda.is_available() else 'cpu'

    if FLAGS.model == "R101":
        model = models.resnet101(weights="ResNet101_Weights.IMAGENET1K_V2")
        batch_size = 50
    elif FLAGS.model == "R152":
        model = models.resnet152(weights="ResNet152_Weights.IMAGENET1K_V2")
        batch_size = 25
    elif FLAGS.model == "RESNXT":
        model = models.resnext101_64x4d(weights="ResNeXt101_64X4D_Weights.IMAGENET1K_V1")
        batch_size = 25

    function_steps = 50
    model = model.eval()
    model.to(device)

    # Initialize random mask for PIC (based on patches, not pixels)
    num_patches = 1000  # Adjust based on your data
    random_mask = PIC.generate_random_mask(num_patches, fraction=0.01)
    saliency_thresholds = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.13, 0.21, 0.34, 0.5, 0.75]

    # Initialize RISE mean feature baseline
    mean_feature = lambda x: torch.mean(x, dim=1, keepdim=True).expand(-1, x.shape[1], -1)

    insertion = RISE.CausalMetric(model, num_patches, 'ins', substrate_fn=mean_feature)
    deletion = RISE.CausalMetric(model, num_patches, 'del', substrate_fn=torch.zeros_like)

    run_and_save_tests(num_patches, random_mask, saliency_thresholds, FLAGS.image_count, FLAGS.function, function_steps, batch_size, model, FLAGS.model, deletion, insertion, device, FLAGS.feature_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Attribution Test Script for Features.')
    parser.add_argument('--function', type=str, default="IG", help='Name of the attribution method to test: IG, LIG, GIG, AGI, IDG.')
    parser.add_argument('--image_count', type=int, default=5000, help='How many WSIs to test with.')
    parser.add_argument('--model', type=str, default="R101", help='Classifier to use: R101, R152, or RESNXT')
    parser.add_argument('--cuda_num', type=int, default=0, help='The number of the GPU to use.')
    parser.add_argument('--feature_path', type=str, default="Features", help='The relative path to your WSI feature files (.pt and .np).')

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
    
    # python evaluation.py --function IG --image_count 5000 --model R101 --cuda_num 0 --feature_path /path/to/features