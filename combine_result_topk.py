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

    # Split into separate tables per folder
    for folder, df_group in combined_df.groupby('source_folder'):
        print(f"\n=== {folder.upper()} ===")

        # For camelyon (clam_metrics), drop pred_label == 0
        if folder == "clam_metrics":
            df_group = df_group[df_group['pred_label'] != 0]

        grouped = (
            df_group
            .groupby(['method', 'pred_label'])[['AIC', 'SIC']]
            .agg(['mean', 'std'])
            .reset_index()
        )

        # Flatten MultiIndex columns
        grouped.columns = ['method', 'pred_label', 'AIC_mean', 'AIC_std', 'SIC_mean', 'SIC_std']

        # Add folder name
        grouped.insert(0, 'source_folder', folder)

        # Sort
        grouped = grouped.sort_values(by=['pred_label', 'SIC_mean'], ascending=[True, False])

        print(grouped.to_string(index=False))
else:
    print("No valid result files found.")
