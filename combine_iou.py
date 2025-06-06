import os
import pandas as pd
from collections import defaultdict

# Root directory containing the result CSVs
root_dir = "/home/mvu9/cig_attributions/dice_iou/camelyon16"

# Classifiers and IG methods
classifiers = ['clam', 'mlp']
ig_methods = ['eg', 'ig', 'g', 'cig', 'idg']

# Store results
results = defaultdict(list)

# Loop through each classifier and method
for clf in classifiers:
    for method in ig_methods:
        csv_path = os.path.join(root_dir, clf, method, 'dice_iou_fold_1.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # You can change column name if needed (e.g., 'iou' or 'dice')
                if 'iou' in df.columns:
                    metric = df['iou'].mean()
                elif 'dice' in df.columns:
                    metric = df['dice'].mean()
                else:
                    raise ValueError(f"No 'iou' or 'dice' column in {csv_path}")
                results[clf].append((method, metric))
            except Exception as e:
                print(f"[ERROR] Failed to process {csv_path}: {e}")
        else:
            print(f"[WARN] File not found: {csv_path}")

# Print summary
print("\n=== Average IoU/Dice per Classifier and Method ===")
for clf in classifiers:
    print(f"\nClassifier: {clf}")
    for method, avg in results[clf]:
        print(f"  {method.upper():<5} : {avg:.4f}")
