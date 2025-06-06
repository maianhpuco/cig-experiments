import os
import pandas as pd
from collections import defaultdict

# Root directory containing the result CSVs
root_dir = "/home/mvu9/cig_attributions/dice_iou/camelyon16"
classifiers = ['clam', 'mlp']
methods = ['eg', 'ig', 'g', 'cig', 'idg']
metric_keys = ['dice_tumor', 'iou_tumor', 'dice_normal', 'iou_normal']

# Store results
results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Traverse folders and load metrics
for clf in classifiers:
    for method in methods:
        csv_path = os.path.join(root_dir, clf, method, "dice_iou_fold_1.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                for metric in metric_keys:
                    if metric in df.columns:
                        results[clf][method][metric].extend(df[metric].dropna().tolist())
            except Exception as e:
                results[clf][method]["error"] = str(e)

# Prepare summary with mean ± std
summary = []
for clf in classifiers:
    for method in methods:
        entry = {"classifier": clf, "method": method}
        for metric in metric_keys:
            values = results[clf][method].get(metric, [])
            if values:
                mean = pd.Series(values).mean()
                std = pd.Series(values).std()
                entry[metric] = f"{mean:.4f} ± {std:.4f}"
            else:
                entry[metric] = "---"
        summary.append(entry)

# Convert to DataFrame for display
summary_df = pd.DataFrame(summary)

