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

# Method order for sorting
method_order = {'random': 0, 'g': 1, 'ig': 2, 'eg': 3, 'idg': 4, 'cig': 5}

def parse_folder(folder_name):
    folder_name = folder_name.lower()
    if "camelyon16" in folder_name or "clam_metrics" in folder_name or "mlp_metrics" in folder_name:
        dataset = "Camelyon16"
    elif "renal" in folder_name:
        dataset = "TCGA-RCC"
    elif "lung" in folder_name:
        dataset = "TCGA-Lung"
    else:
        dataset = "Camelyon16"

    if "clam" in folder_name:
        classifier = "CLAM"
    elif "mlp" in folder_name:
        classifier = "MLP"
    else:
        classifier = "unknown"

    return dataset, classifier

all_dfs = []

# Read all CSVs
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

# Process and print
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)

    for folder, folder_df in combined_df.groupby('source_folder'):
        dataset = folder_df['dataset'].iloc[0]
        classifier = folder_df['classifier'].iloc[0]

        print(f"\n=== TABLE FOR: {folder.upper()} ===")

        # Special rule: skip class 0 for Camelyon16 + CLAM
        if dataset.lower() == "camelyon16" and classifier.lower() == "clam":
            folder_df = folder_df[folder_df['pred_label'] != 0]

        # Group and aggregate
        grouped = (
            folder_df
            .groupby(['method', 'pred_label'])[['AIC', 'SIC']]
            .agg(['mean', 'std'])
            .reset_index()
        )

        # Flatten MultiIndex columns
        grouped.columns = ['method', 'pred_label', 'AIC_mean', 'AIC_std', 'SIC_mean', 'SIC_std']

        # Add metadata columns
        grouped.insert(0, 'classifier', classifier)
        grouped.insert(0, 'dataset', dataset)

        # Sort method by predefined order
        grouped['method_order'] = grouped['method'].map(lambda m: method_order.get(m, 999))
        grouped = grouped.sort_values(by=['pred_label', 'method_order']).drop(columns='method_order')

        # Reorder columns
        grouped = grouped[['dataset', 'classifier', 'pred_label', 'method', 'AIC_mean', 'AIC_std', 'SIC_mean', 'SIC_std']]

        print(grouped.to_string(index=False))
else:
    print("No valid result files found.")
