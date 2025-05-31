import torch
import os
from src.externals.CLAM.utils.utils import load_pkl
from src.externals.CLAM.utils.core_utils import load_clam_model

# === Required paths ===
feature_path = "/home/mvu9/processing_datasets/processing_camelyon16/features_fp/pt_files/normal_1.pt"  # your .pt file
checkpoint_path = "/home/mvu9/processing_datasets/processing_camelyon16/clam_result/result_final_ep200/s_1_checkpoint.pt"

# === Load model ===
model_dict = load_clam_model(checkpoint_path)
model = model_dict['model']
model.eval()
model.cuda()  # or .to(device)

# === Load features ===
features = torch.load(feature_path, map_location='cuda')  # shape: [N, D]
features = features.unsqueeze(0)  # shape: [1, N, D]

# === Forward pass ===
with torch.no_grad():
    logits, Y_prob, Y_hat, _, _ = model(features)

print(f"Logits: {logits}")
print(f"Probability: {Y_prob}")
print(f"Prediction: {Y_hat}")
