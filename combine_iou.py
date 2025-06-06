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

# Prepare summary with mean ± std and sorting
summary = []
for clf in classifiers:
    method_summaries = []
    for method in methods:
        entry = {"classifier": clf, "method": method}
        dice_values = results[clf][method].get('dice_tumor', [])
        for metric in metric_keys:
            values = results[clf][method].get(metric, [])
            if values:
                mean = pd.Series(values).mean()
                std = pd.Series(values).std()
                entry[metric] = f"{mean:.4f} ± {std:.4f}"
            else:
                entry[metric] = "---"
        avg_dice = pd.Series(dice_values).mean() if dice_values else -1
        entry["avg_dice"] = avg_dice  # for sorting
        method_summaries.append(entry)
    
    # Sort methods for this classifier by average dice descending
    method_summaries.sort(key=lambda x: x["avg_dice"], reverse=True)
    summary.extend(method_summaries)

# Convert to DataFrame and drop the helper column
summary_df = pd.DataFrame(summary)
summary_df.drop(columns=["avg_dice"], inplace=True)

import ace_tools as tools; tools.display_dataframe_to_user(name="Sorted Dice-IoU Summary", dataframe=summary_df)
