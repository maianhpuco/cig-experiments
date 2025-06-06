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
        dataset = "Camelyon16"  # fallback

    if "clam" in folder_name:
        classifier = "CLAM"
    elif "mlp" in folder_name:
        classifier = "MLP"
    else:
        classifier = "unknown"

    return dataset, classifier

all_dfs = []

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

# LaTeX formatting utilities
latex_method_names = {
    "g": "Gradient",
    "ig": "IG",
    "idg": "IDG",
    "eg": "EG",
    "cig": "\\textbf{CIG (ours)}"
}

class_name_map = {
    "Camelyon16": {1: "Tumor"},
    "TCGA-RCC": {0: "pRCC", 1: "ccRCC", 2: "chRCC"},
    "TCGA-Lung": {0: "LUAD", 1: "LUSC"}
}

def format_val_with_std(mean, std, bold=False):
    val = f"{mean:.3f} ± {std:.3f}"
    return f"\\textbf{{{val}}}" if bold else val

def determine_bold_flags(df, classes):
    bold_flags = {}
    for clf in df['classifier'].unique():
        for c in classes:
            subset = df[(df['classifier'] == clf) & (df['pred_label'] == c)]
            if not subset.empty:
                max_aic = subset['AIC_mean'].max()
                max_sic = subset['SIC_mean'].max()
                for _, row in subset.iterrows():
                    key = (clf, row['method'], c)
                    bold_flags[key] = (
                        abs(row['AIC_mean'] - max_aic) < 1e-6,
                        abs(row['SIC_mean'] - max_sic) < 1e-6
                    )
    return bold_flags

def format_table(df, dataset_name):
    classes = sorted(df['pred_label'].unique())
    class_labels = class_name_map.get(dataset_name, {})
    classifiers = ['CLAM', 'MLP']
    methods = ['g', 'ig', 'idg', 'eg', 'cig']

    bold_flags = determine_bold_flags(df, classes)

    lines = []
    table_env = "table*" if dataset_name == "TCGA-RCC" else "table"
    lines.append(f"\\begin{{{table_env}}}[t]")
    lines += [
        "    \\centering",
        "    \\renewcommand{\\arraystretch}{1.2}",
        "    \\setlength{\\tabcolsep}{4pt}"
    ]

    col_span = {1: "c|l|cc", 2: "c|l|cc|cc", 3: "c|l|cc|cc|cc"}[len(classes)]
    lines.append(f"    \\begin{{tabular}}{{{col_span}}}")
    lines.append("        \\toprule")

    if len(classes) == 1:
        lines.append("        \\textbf{Classifier} & \\textbf{Attribution Method} & \\textbf{AIC}~↑ & \\textbf{SIC}~↑ \\")
    else:
        header = " & ".join([f"\\multicolumn{{2}}{{c|}}{{\\textbf{{{class_labels.get(c, f'Class {c}')}}}}}" for c in classes[:-1]])
        header += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{class_labels.get(classes[-1], f'Class {classes[-1]}')}}}}}"
        lines.append("        \\multirow{2}{*}{\\textbf{Classifier}} & \\multirow{2}{*}{\\textbf{Attribution Method}} & " + header + " \\")
        cmid = " ".join([f"\\cmidrule(lr){{{3+2*i}-{4+2*i}}}" for i in range(len(classes))])
        lines.append(f"        {cmid}")
        lines.append("        & & " + " & ".join(["AIC~↑ & SIC~↑"] * len(classes)) + " \\")

    lines.append("        \\midrule")
    lines.append("        \\multicolumn{2}{c|}{\\textbf{Random}} " + "& --- & --- " * len(classes) + "\\")
    lines.append("        \\midrule")

    for clf in classifiers:
        lines.append(f"        \\multirow{{5}}{{*}}{{{clf}}}")
        for method in methods:
            row_vals = []
            for c in classes:
                row = df[(df['classifier'] == clf) & (df['method'] == method) & (df['pred_label'] == c)]
                if not row.empty:
                    aic_m, aic_s = row[['AIC_mean', 'AIC_std']].values[0]
                    sic_m, sic_s = row[['SIC_mean', 'SIC_std']].values[0]
                    bold_aic, bold_sic = bold_flags.get((clf, method, c), (False, False))
                    row_vals.append(format_val_with_std(aic_m, aic_s, bold_aic))
                    row_vals.append(format_val_with_std(sic_m, sic_s, bold_sic))
                else:
                    row_vals += ["---", "---"]
            method_name = latex_method_names.get(method, method.upper())
            lines.append(f"            & {method_name} & " + " & ".join(row_vals) + " \\")
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

# Main execution
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
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
else:
    print("No valid result files found.")
