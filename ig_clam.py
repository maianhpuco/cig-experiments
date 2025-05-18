import os
import sys 
import argparse
import time
import numpy as np
import h5py
import torch
import yaml
import pickle
from torch import nn
import argparse
from attr_method._common import (
    sample_random_features,
    call_model_function
)

ig_path = os.path.abspath(os.path.join("src/models/attr_method"))
clf_path = os.path.abspath(os.path.join("src/classifers"))
sys.path.append(ig_path)   
sys.path.append(clf_path)  
from clam import load_clam_model  

# from data.ig_dataset import IG_dataset
# from utils.utils import load_pkl
# from utils.models.model_utils import get_clam_model
# from saliency.core.base import CoreSaliency
 
# # base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) 






# from LoadClamModel import load_clam_model  # adjust if the function is in a different file

def get_dummy_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop_out', type=float, default=0.25)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small')
    args = parser.parse_args(args=[])  # empty args for testing
    return args

if __name__ == "__main__":
    args = get_dummy_args()

    # Update with your actual checkpoint path
    ckpt_path = "/home/mvu9/processing_datasets/processing_camelyon16/result_2025-05-18_22-23-56/s_1_checkpoint.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = load_clam_model(args, ckpt_path, device=device)

    print("Model loaded and ready for inference.")




# def load_clam_model(config):
#     model_type = config['model_type']
#     model_size = config['model_size']
    
#     checkpoint_path = config['paths']['for_ig_checkpoint_path']
    
#     model = get_clam_model(
#         model_type=model_type,
#         model_size=model_size,
#         drop_out=config['drop_out'],
#         n_classes=2,
#         instance_loss=config['inst_loss'],
#         subtyping=config['subtyping'],
#         B=config['B'],
#         device='cuda' if torch.cuda.is_available() else 'cpu'
#     )
#     ckpt = torch.load(checkpoint_path, map_location='cpu')
#     model.load_state_dict(ckpt['model_state_dict'])
#     model.eval()
#     return model
 
# def main(args): 
#     '''
#     Input: h5 file
#     Output: save scores into a json folder
#     '''
#     #---------------------------------------------------- 
#     if args.ig_name=='integrated_gradient':
#         from attr_method.integrated_gradient import IntegratedGradients as AttrMethod 
       
#     elif args.ig_name=='vanilla_gradient':
#         from attr_method.vanilla_gradient import VanillaGradients as AttrMethod 
    
#     elif args.ig_name=='contrastive_gradient':
#         from attr_method.contrastive_gradient import ContrastiveGradients as AttrMethod 

#     elif args.ig_name=='expected_gradient':
#        from attr_method.expected_gradient import ExpectedGradients as AttrMethod   
    
#     elif args.ig_name=='integrated_decision_gradient':
#        from attr_method.integrated_decision_gradient import IntegratedDecisionGradients as AttrMethod     
  
#     elif args.ig_name=='optim_square_integrated_gradient':
#        from attr_method.optim_square_integrated_gradient import OptimSquareIntegratedGradients as AttrMethod
    
#     elif args.ig_name=='square_integrated_gradient':
#        from attr_method.square_integrated_gradient import SquareIntegratedGradients as AttrMethod     
#     # LIME
#     # KernelSHAP
#     # DeepSHAP 
    
#     print(f"Running for {args.ig_name} Attribution method") 
    
    
#     #----------------------------------------------------    
#     attribution_method = AttrMethod()   
    
#     score_save_path = os.path.join(args.attribution_scores_folder, f'{args.ig_name}') 
#     # print("score_save_path", score_save_path)
#     # if os.path.exists(score_save_path):
#     #     shutil.rmtree(score_save_path)  # Delete the existing directory
#     # os.makedirs(score_save_path)    
    

#     checkpoint_path = os.path.join(args.checkpoints_dir, f'{CHECK_POINT_FILE}')
#     mil_model = load_model(checkpoint_path)
    
#     if args.dry_run==1:
#         dataset = IG_dataset(
#             args.features_h5_path,
#             args.slide_path,
#             basenames=['tumor_026', 'tumor_031', 'tumor_032','tumor_036']
#         )   
        
#     else:
#         basenames = [] 
#         for basename in os.listdir(args.slide_path):
#             basename = basename.split(".")[0]
#             if basename.startswith('normal_'): 
#             # if basename.startswith(('tumor_', 'test_')):  # Check if it starts with either prefix
#                 basenames.append(basename)
        
#         dataset = IG_dataset(
#             args.features_h5_path,
#             args.slide_path,
#             basenames=basenames
#             )
        
#     if args.do_normalizing: 
#         with h5py.File(args.feature_mean_std_path, "r") as f:
#             mean = f["mean"][:]
#             std = f["std"][:]
            
#     print(">>>>>>>>>----- Total number of sample in dataset:", len(dataset)) 
    
#     for idx, data in enumerate(dataset):
#         total_file = len(dataset)
#         print(f"Processing the file numner {idx+1}/{total_file}")
#         basename = data['basename']
#         features = data['features']  # Shape: (batch_size, num_patches, feature_dim)
#         label = data['label']
#         start = time.time() 
            
    
#         if args.do_normalizing:   
#             print("----- normalizing")
#             features = (features - mean) / (std + 1e-8)  
        
#         # randomly sampling #file to create the baseline 
#         stacked_features_baseline, selected_basenames =  sample_random_features(
#             dataset, num_files=20) 
#         stacked_features_baseline = stacked_features_baseline.numpy() 
        
#         # if args.ig_name=='ig':
#         kwargs = {
#             "x_value": features,  
#             "call_model_function": call_model_function,  
#             "model": mil_model,  
#             "baseline_features": stacked_features_baseline,  # Optional
#             "memmap_path": args.memmap_path, 
#             "x_steps": 50,  
#         }  
 
#         attribution_values = attribution_method.GetMask(**kwargs) 
#         scores = attribution_values.mean(1)
#         _save_path = os.path.join(score_save_path, f'{basename}.npy')
#         np.save(_save_path, scores)
#         print(f"Done save result numpy file at shape {scores.shape} at {_save_path}")
    
         
# if __name__=="__main__":
#     # get config 
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dry_run', type=int, default=0)
#     parser.add_argument('--config_file', default='ma_exp002')
#     parser.add_argument('--ig_name', 
#                     default='integrated_gradients', 
#                     choices=[
#                         'integrated_gradient', 
#                         'expected_gradient', 
#                         'integrated_decision_gradient', 
#                         'contrastive_gradient', 
#                         'vanilla_gradient', 
#                         'square_integrated_gradient', 
#                         'optim_square_integrated_gradient'
#                         ],
#                     help='Choose the attribution method to use.') 
    
#     args = parser.parse_args()
    
#     if os.path.exists(f'./testbest_config/{args.config_file}.yaml'):
#         config = load_config(f'./testbest_config/{args.config_file}.yaml')
#         args.use_features = config.get('use_features', True)
        
#         args.slide_path = config.get('SLIDE_PATH')
#         args.features_h5_path = config.get("FEATURES_H5_PATH") # save all the features
#         args.checkpoints_dir = config.get("CHECKPOINT_PATH")
#         if args.dry_run==1:
#             args.attribution_scores_folder = config.get("SCORE_FOLDER_DRYRUN") 
#             args.plot_path = config.get("PLOT_PATH_DRYRUN")    
#             print("----")
            
#             print("args.attribution_scores_folder", args.attribution_scores_folder)
#             print("args.plot_path", args.plot_path)
            
#         else: 
#             args.attribution_scores_folder = config.get("SCORE_FOLDER")    
#             args.plot_path = config.get("PLOT_PATH") 
#         print("Attribution folder path", args.attribution_scores_folder)
        
#         os.makedirs(args.features_h5_path, exist_ok=True)  
#         os.makedirs(args.attribution_scores_folder, exist_ok=True) 
#         args.batch_size = config.get('batch_size')
#         args.feature_extraction_model = config.get('feature_extraction_model')
#         args.device = "cuda" if torch.cuda.is_available() else "cpu"
#         args.feature_mean_std_path=config.get("FEATURE_MEAN_STD_PATH")
#         # args.ig_name = "integrated_gradients"
#         args.do_normalizing = True
#         args.memmap_path = config.get("MEMMAP_PATH")
        
#     # CHECK_POINT_FILE = 'mil_checkpoint_draft.pth' 
#     main(args) 
   
   
   
 