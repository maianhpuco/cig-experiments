import os
import pandas as pd
import glob

# List of base metric directories to check
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
    # Find all result CSVs inside any IG method folder
    csv_files = glob.glob(os.path.join(base_dir, "*", "topk_pic_results_fold_1.csv"))
    print(f"[INFO] Found {len(csv_files)} files in {base_dir}")

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            print(df.head(3))
            method_name = df['IG'].iloc[0].lower()  # Assumes consistent IG name in each file
            df['method'] = method_name
            all_dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed to process {file}: {e}")

if not all_dfs:
    print("No files loaded.")
else:
    # Combine all data
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Count how many files per method
    file_counts = combined_df.groupby("method").size().reset_index(name="file_count")

    # Aggregate AIC and SIC
    summary_stats = combined_df.groupby("method")[["AIC", "SIC"]].agg(['mean', 'std']).reset_index()

    # Merge counts with stats
    summary = pd.merge(summary_stats, file_counts, on="method")
    summary_sorted = summary.sort_values(by=('SIC', 'mean'), ascending=False)

    print("\n=== Summary (unsorted) ===")
    print(summary.to_string(index=False))

    print("\n=== Summary (sorted by SIC mean) ===")
    print(summary_sorted.to_string(index=False))
