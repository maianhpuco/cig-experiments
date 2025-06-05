import os
import pandas as pd
import glob

# All base directories to search
base_dirs = [
    "/home/mvu9/processing_datasets/processing_camelyon16/clam_metrics",
    "/home/mvu9/processing_datasets/processing_tcga_256/clam_tcga_renal_metrics",
    "/home/mvu9/processing_datasets/processing_tcga_256/clam_tcga_lung_metrics",
    "/home/mvu9/processing_datasets/processing_camelyon16/mlp_metrics",
    "/home/mvu9/processing_datasets/processing_tcga_256/mlp_tcga_lung_metrics",
    "/home/mvu9/processing_datasets/processing_tcga_256/mlp_tcga_renal_metrics",
]

all_dfs = []

# Read all CSVs from each base directory
for base_dir in base_dirs:
    csv_files = glob.glob(os.path.join(base_dir, "*", "topk_pic_results_fold_1.csv"))
    folder_name = os.path.basename(base_dir)
    print(f"[INFO] {folder_name}: found {len(csv_files)} files")

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['method'] = df['IG'].str.lower()
            df['source_folder'] = folder_name
            all_dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {file}: {e}")

# Combine and group
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Group by source_folder, method, pred_label
    grouped = (
        combined_df
        .groupby(['source_folder', 'method', 'pred_label'])[['AIC', 'SIC']]
        .agg(['mean', 'std'])
        .reset_index()
    )

    # Flatten MultiIndex columns
    grouped.columns = ['source_folder', 'method', 'pred_label', 'AIC_mean', 'AIC_std', 'SIC_mean', 'SIC_std']

    # Sort by source_folder, pred_label, then SIC_mean (descending)
    grouped = grouped.sort_values(by=['source_folder', 'pred_label', 'SIC_mean'], ascending=[True, True, False])

    print("\n=== Grouped by source_folder, method, pred_label (sorted) ===")
    print(grouped.to_string(index=False))
else:
    print("No valid result files found.")
