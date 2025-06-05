import os
import pandas as pd
import glob

# Define base directories
base_dirs = [
    "/home/mvu9/processing_datasets/processing_camelyon16/clam_metrics",
    "/home/mvu9/processing_datasets/processing_tcga_256/clam_tcga_renal_metrics",
    "/home/mvu9/processing_datasets/processing_tcga_256/clam_tcga_lung_metrics",
    "/home/mvu9/processing_datasets/processing_camelyon16/mlp_metrics",
    "/home/mvu9/processing_datasets/processing_tcga_256/mlp_tcga_lung_metrics",
    "/home/mvu9/processing_datasets/processing_tcga_256/mlp_tcga_renal_metrics",
]

def parse_folder(folder_name):
    folder_name = folder_name.lower()
    if "camelyon16" in folder_name:
        dataset = "camelyon16"
    elif "renal" in folder_name:
        dataset = "tcga_renal"
    elif "lung" in folder_name:
        dataset = "tcga_lung"
    else:
        dataset = "unknown"

    if "clam" in folder_name:
        classifier = "clam"
    elif "mlp" in folder_name:
        classifier = "mlp"
    else:
        classifier = "unknown"

    return dataset, classifier

all_dfs = []

# Read all CSVs from each base directory
for base_dir in base_dirs:
    csv_files = glob.glob(os.path.join(base_dir, "*", "topk_pic_results_fold_1.csv"))
    folder_name = os.path.basename(base_dir)
    dataset, classifier = parse_folder(folder_name)
    print(f"[INFO] {folder_name}: found {len(csv_files)} files")

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['method'] = df['IG'].str.lower()
            df['source_folder'] = folder_name
            df['dataset'] = dataset
            df['classifier'] = classifier
            all_dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {file}: {e}")

# Combine and group
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Loop over dataset/classifier pairs
    for (dataset, classifier), folder_df in combined_df.groupby(['dataset', 'classifier']):
        print(f"\n=== DATASET: {dataset.upper()} | CLASSIFIER: {classifier.upper()} ===")

        # Special rule: exclude class 0 for camelyon16 using clam
        if dataset == "camelyon16" and classifier == "clam":
            folder_df = folder_df[folder_df['pred_label'] != 0]

        # Group by method and pred_label
        grouped = (
            folder_df
            .groupby(['method', 'pred_label'])[['AIC', 'SIC']]
            .agg(['mean', 'std'])
            .reset_index()
        )

        # Flatten MultiIndex columns
        grouped.columns = ['method', 'pred_label', 'AIC_mean', 'AIC_std', 'SIC_mean', 'SIC_std']

        # Add metadata
        grouped.insert(0, 'classifier', classifier)
        grouped.insert(0, 'dataset', dataset)

        # Print separate tables per prediction label
        for pred_label, pred_df in grouped.groupby('pred_label'):
            print(f"\n--- Prediction Class: {pred_label} ---")
            pred_df_sorted = pred_df.sort_values(by='SIC_mean', ascending=False)
            print(pred_df_sorted.to_string(index=False))
else:
    print("No valid result files found.")
