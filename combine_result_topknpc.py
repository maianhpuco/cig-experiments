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

def format_val_with_std(mean, std, bold=False):
    val = f"{mean:.3f} ± {std:.3f}"
    return f"\\textbf{{{val}}}" if bold else val

# Label mapping
class_name_map = {
    "Camelyon16": {1: "Tumor"},
    "TCGA-RCC": {0: "pRCC", 1: "ccRCC", 2: "chRCC"},
    "TCGA-Lung": {0: "LUAD", 1: "LUSC"}
}

def format_table(df, dataset_name):
    classes = sorted(df['pred_label'].unique())
    class_labels = class_name_map.get(dataset_name, {})
    classifiers = ['CLAM', 'MLP']
    methods = ['g', 'ig', 'idg', 'eg', 'cig']
    
    lines = []
    table_env = "table*" if dataset_name == "TCGA-RCC" else "table"
    lines.append(f"\\begin{{{table_env}}}[t]")
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
    else:
        class_headers = " & ".join([f"\\multicolumn{{2}}{{c|}}{{\\textbf{{{class_labels.get(c, f'Class {c}')}}}}}" for c in classes[:-1]])
        class_headers += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{class_labels.get(classes[-1], f'Class {classes[-1]}')}}}}}"
        lines.append("        \\multirow{2}{*}{\\textbf{Classifier}} & \\multirow{2}{*}{\\textbf{Attribution Method}} " + f"& {class_headers} \\\\")
        cmid_rules = " ".join([f"\\cmidrule(lr){{{3+2*i}-{4+2*i}}}" for i in range(len(classes))])
        lines.append(f"        {cmid_rules}")
        lines.append("        & & " + " & ".join(["AIC~↑ & SIC~↑"] * len(classes)) + " \\\\")

    lines.append("        \\midrule")
    lines.append("        \\multicolumn{2}{c|}{\\textbf{Random}} " + "& --- & --- " * len(classes) + "\\\\")
    lines.append("        \\midrule")

    # Process each classifier
    for clf in classifiers:
        clf_df = df[df['classifier'] == clf]
        if clf_df.empty:
            continue

        # Find max values for AIC and SIC for each class
        max_aic = {}
        max_sic = {}
        for c in classes:
            class_df = clf_df[clf_df['pred_label'] == c]
            if not class_df.empty:
                max_aic[c] = class_df['AIC_mean'].max()
                max_sic[c] = class_df['SIC_mean'].max()
            else:
                max_aic[c] = None
                max_sic[c] = None

        lines.append(f"        \\multirow{{5}}{{*}}{{{clf}}}")
        for method in methods:
            method_df = clf_df[clf_df['method'] == method]
            row_vals = []
            for c in classes:
                sub_df = method_df[method_df['pred_label'] == c]
                if not sub_df.empty:
                    aic_mean, aic_std = sub_df[['AIC_mean', 'AIC_std']].values[0]
                    sic_mean, sic_std = sub_df[['SIC_mean', 'SIC_std']].values[0]
                    row_vals.append((aic_mean, aic_std, sic_mean, sic_std))
                else:
                    row_vals.append((None, None, None, None))

            formatted_vals = []
            for i, (aic_m, aic_s, sic_m, sic_s) in enumerate(row_vals):
                if aic_m is None:
                    formatted_vals += ["---", "---"]
                else:
                    bold_aic = aic_m == max_aic[classes[i]] and aic_m is not None
                    bold_sic = sic_m == max_sic[classes[i]] and sic_m is not None
                    formatted_vals.append(format_val_with_std(aic_m, aic_s, bold_aic))
                    formatted_vals.append(format_val_with_std(sic_m, sic_s, bold_sic))

            method_name = latex_method_names.get(method, method.upper())
            lines.append(f"            & {method_name} & " + " & ".join(formatted_vals) + " \\\\")
        lines.append("        \\midrule")

    lines[-1] = "        \\bottomrule"
    lines.append("    \\end{tabular}")

    caption = {
        "Camelyon16": "\\caption{Attribution results on the \\textbf{Tumor class} for Camelyon16.}",
        "TCGA-Lung": "\\caption{Attribution results for each class in \\textbf{TCGA-Lung} (0: LUAD, 1: LUSC).}",
        "TCGA-RCC": "\\caption{Attribution results for each class in \\textbf{TCGA-RCC} (0: pRCC, 1: ccRCC, 2: chRCC).}"
    }[dataset_name]
    label = {
        "Camelyon16": "\\label{tab:camelyon16_tumor}",
        "TCGA-Lung": "\\label{tab:tcga_lung_classes}",
        "TCGA-RCC": "\\label{tab:tcga_rcc_classes}"
    }[dataset_name]
    lines.append(f"    {caption}")
    lines.append(f"    {label}")
    lines.append(f"\\end{{{table_env}}}")

    return "\n".join(lines)

# Generate LaTeX tables for each dataset
for dataset in combined_df['dataset'].unique():
    df_subset = combined_df[combined_df['dataset'] == dataset]
    if df_subset.empty:
        continue
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
    
    
print(combined_df)