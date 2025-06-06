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
    if "camelyon16" in folder_name or "clam_metrics" in folder_name or "mlp_metrics" in folder_name:
        dataset = "Camelyon16"

    elif "renal" in folder_name:
        dataset = "TCGA-RCC"
    elif "lung" in folder_name:
        dataset = "TCGA-Lung"
    else:
        dataset = "Camelyon16"  # fallback if unrecognized

    if "clam" in folder_name:
        classifier = "CLAM"
    elif "mlp" in folder_name:
        classifier = "MLP"
    else:
        classifier = "unknown"

    return dataset, classifier


all_dfs = []

# Read all CSVs from each base directory
for base_dir in base_dirs:
    csv_files = glob.glob(os.path.join(base_dir, "*", "topknpc_pic_results_fold_1.csv"))
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
# --------------------

import pandas as pd

# # Dummy structure to allow the function definition (data not available in this reset state)
# combined_df = pd.DataFrame(columns=[
#     'dataset', 'classifier', 'method', 'pred_label',
#     'AIC_mean', 'AIC_std', 'SIC_mean', 'SIC_std'
# ])

# Mapping for LaTeX formatting
latex_method_names = {
    "g": "Gradient",
    "ig": "IG",
    "idg": "IDG",
    "eg": "EG",
    "cig": "\\textbf{CIG (ours)}"
}

def format_val(val):
    return f"{val:.3f}" if isinstance(val, float) else "---"

def format_table(df, dataset_name):
    classes = sorted(df['pred_label'].unique())
    classifiers = ['CLAM', 'MLP']
    methods = ['g', 'ig', 'idg', 'eg', 'cig']

    lines = []
    if dataset_name == "Camelyon16":
        lines.append("\\begin{table}[t]")
    elif dataset_name == "TCGA-RCC":
        lines.append("\\begin{table*}[t]")
    else:
        lines.append("\\begin{table}[t]")

    lines += [
        "    \\centering",
        "    \\renewcommand{\\arraystretch}{1.2}",
        "    \\setlength{\\tabcolsep}{4pt}"
    ]

    col_span = {
        1: "\\begin{tabular}{c|l|cc}",
        2: "\\begin{tabular}{c|l|cc|cc}",
        3: "\\begin{tabular}{c|l|cc|cc|cc}"
    }[len(classes)]

    lines.append(f"    {col_span}")
    lines.append("        \\toprule")

    if len(classes) == 1:
        lines.append("        \\textbf{Classifier} & \\textbf{Attribution Method} & \\textbf{AIC}~↑ & \\textbf{SIC}~↑ \\\\")
    elif len(classes) == 2:
        lines.append("        \\multirow{2}{*}{\\textbf{Classifier}} & \\multirow{2}{*}{\\textbf{Attribution Method}} "
                     "& \\multicolumn{2}{c|}{\\textbf{Class 0}} & \\multicolumn{2}{c}{\\textbf{Class 1}} \\\\")
        lines.append("        \\cmidrule(lr){3-4} \\cmidrule(lr){5-6}")
        lines.append("        & & AIC~↑ & SIC~↑ & AIC~↑ & SIC~↑ \\\\")
    else:
        lines.append("        \\multirow{2}{*}{\\textbf{Classifier}} & \\multirow{2}{*}{\\textbf{Attribution Method}} "
                     "& \\multicolumn{2}{c|}{\\textbf{Class 0}} & \\multicolumn{2}{c|}{\\textbf{Class 1}} & \\multicolumn{2}{c}{\\textbf{Class 2}} \\\\")
        lines.append("        \\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}")
        lines.append("        & & AIC~↑ & SIC~↑ & AIC~↑ & SIC~↑ & AIC~↑ & SIC~↑ \\\\")

    lines.append("        \\midrule")
    lines.append("        \\multicolumn{2}{c|}{\\textbf{Random}} " + "& --- & --- " * len(classes) + "\\\\")
    lines.append("        \\midrule")

    for clf in classifiers:
        lines.append(f"        \\multirow{{5}}{{*}}{{{clf}}}")
        for i, method in enumerate(methods):
            row_vals = []
            method_df = df[(df['classifier'] == clf) & (df['method'] == method)]
            for c in classes:
                sub_df = method_df[method_df['pred_label'] == c]
                if not sub_df.empty:
                    aic = format_val(sub_df['AIC_mean'].values[0])
                    sic = format_val(sub_df['SIC_mean'].values[0])
                else:
                    aic, sic = "---", "---"
                row_vals += [aic, sic]
            method_name = latex_method_names.get(method, method.upper())
            lines.append(f"            & {method_name} & " + " & ".join(row_vals) + " \\\\")
        lines.append("        \\midrule")

    lines[-1] = "        \\bottomrule"

    lines.append("    \\end{tabular}")
    caption = {
        "Camelyon16": "\\caption{Attribution results on the \\textbf{Tumor class} for Camelyon16.}",
        "TCGA-Lung": "\\caption{Attribution results for each class in \\textbf{TCGA-Lung} (0: LUAD, 1: LUSC).}",
        "TCGA-RCC": "\\caption{Attribution results for each class in \\textbf{TCGA-RCC} (0: KIRP, 1: KIRC, 2: KICH).}"
    }[dataset_name]
    label = {
        "Camelyon16": "\\label{tab:camelyon16_tumor}",
        "TCGA-Lung": "\\label{tab:tcga_lung_classes}",
        "TCGA-RCC": "\\label{tab:tcga_rcc_classes}"
    }[dataset_name]
    lines.append(f"    {caption}")
    lines.append(f"    {label}")
    lines.append("\\end{table*}" if dataset_name == "TCGA-RCC" else "\\end{table}")

    return "\n".join(lines)

#--------- 
# for dataset in combined_df['dataset'].unique():
#     df_subset = combined_df[combined_df['dataset'] == dataset]
#     latex_code = format_table(df_subset, dataset)
#     print(latex_code)
#     print("\n\n")  # Optional spacing between tables 
    
for dataset in combined_df['dataset'].unique():
    df_subset = combined_df[combined_df['dataset'] == dataset]

    if df_subset.empty:
        print(f"[WARN] No data for dataset {dataset}")
        continue

    # Group and compute means and stds
    grouped = (
        df_subset
        .groupby(['classifier', 'method', 'pred_label'])[['AIC', 'SIC']]
        .agg(['mean', 'std'])
        .reset_index()
    )
    grouped.columns = ['classifier', 'method', 'pred_label', 'AIC_mean', 'AIC_std', 'SIC_mean', 'SIC_std']

    latex_code = format_table(grouped, dataset)
    print(latex_code)
    print("\n\n")