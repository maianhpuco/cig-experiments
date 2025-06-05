import os
import pandas as pd
import glob

base_dirs = [
    "/home/mvu9/processing_datasets/processing_camelyon16/clam_metrics",
    "/home/mvu9/processing_datasets/processing_tcga_256/clam_tcga_renal_metrics",
    "/home/mvu9/processing_datasets/processing_tcga_256/clam_tcga_lung_metrics",
    "/home/mvu9/processing_datasets/processing_camelyon16/mlp_metrics",
    "/home/mvu9/processing_datasets/processing_tcga_256/mlp_tcga_lung_metrics",
    "/home/mvu9/processing_datasets/processing_tcga_256/mlp_tcga_renal_metrics",
]

all_dfs = []

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

if not all_dfs:
    print("No data found.")
else:
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Group by pred_label and folder name
    grouped = combined_df.groupby(['source_folder', 'pred_label'])[['AIC', 'SIC']].agg(['mean', 'std']).reset_index()

    # Optional: flatten multi-level columns
    grouped.columns = ['source_folder', 'pred_label', 'AIC_mean', 'AIC_std', 'SIC_mean', 'SIC_std']

    print("\n=== Grouped by source folder and predicted label ===")
    print(grouped.to_string(index=False))
