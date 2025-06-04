import os
import pandas as pd
import glob

# Define the parent directory containing subdirectories for each method
base_dir = "/home/mvu9/processing_datasets/processing_camelyon16/clam_metrics"

# Pattern to match all result CSVs

csv_files = glob.glob(os.path.join(base_dir, "*", "topk_pic_results_fold_1.csv"))
print(csv_files)
# Load and concatenate all CSVs
all_dfs = []
for file in csv_files:
    print(f"processing {file}")
    df = pd.read_csv(file)
    df['method'] = df['IG'].str.lower()  # Normalize method name
    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)

# Group by method and compute mean and std for AIC and SIC
summary = combined_df.groupby('method')[['AIC', 'SIC']].agg(['mean', 'std']).reset_index()
summary_sorted = summary.sort_values(by=('SIC', 'mean'), ascending=False)
print(summary)
print(summary_sorted)