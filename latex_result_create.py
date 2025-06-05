import pandas as pd

# Simulated grouped data, in practice, this should be the output from your processing script
# Example input structure:
# columns: ['dataset', 'classifier', 'pred_label', 'method', 'AIC_mean', 'AIC_std', 'SIC_mean', 'SIC_std']
data = pd.read_csv("/mnt/data/sample_grouped_results.csv")  # Placeholder path if needed

# Ensure method order
method_order = ['random', 'g', 'ig', 'eg', 'idg', 'cig']
data['method'] = data['method'].str.lower()
data['method'] = pd.Categorical(data['method'], categories=method_order, ordered=True)
data = data.sort_values(by=['classifier', 'method'])

# Formatting function
def format_entry(row):
    aic = f"{row['AIC_mean']:.3f}"
    sic = f"{row['SIC_mean']:.3f}"
    return aic, sic

# Latex table rendering functions
def render_camelyon_tumor(df):
    lines = [
        "\\begin{table}[t]",
        "    \\centering",
        "    \\renewcommand{\\arraystretch}{1.2}",
        "    \\setlength{\\tabcolsep}{5pt}",
        "    \\begin{tabular}{c|l|cc}",
        "        \\toprule",
        "        \\textbf{Classifier} & \\textbf{Attribution Method} & \\textbf{AIC}~↑ & \\textbf{SIC}~↑ \\\\",
        "        \\midrule",
        "        \\multicolumn{2}{c|}{\\textbf{Random}} & --- & --- \\\\",
        "        \\midrule",
    ]

    for clf in ['CLAM', 'MLP']:
        df_clf = df[(df['classifier'] == clf)]
        lines.append(f"        \\multirow{{5}}{{*}}{{{clf}}}")
        for idx, method in enumerate(['g', 'ig', 'idg', 'eg', 'cig']):
            row = df_clf[df_clf['method'] == method]
            if not row.empty:
                aic, sic = format_entry(row.iloc[0])
                method_name = f"\\textbf{{CIG (ours)}}" if method == 'cig' else method.upper()
                prefix = "            &" if idx > 0 else " "
                lines.append(f"{prefix} {method_name:<12} & {aic} & {sic} \\\\")
        lines.append("        \\midrule")

    lines += [
        "    \\end{tabular}",
        "    \\caption{Attribution results on the \\textbf{Tumor class} for Camelyon16.}",
        "    \\label{tab:camelyon16_tumor}",
        "\\end{table}"
    ]
    return "\n".join(lines)

def render_tcga_table(df, dataset, class_labels):
    header = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\renewcommand{\\arraystretch}{1.2}",
        "    \\setlength{\\tabcolsep}{4pt}",
        "    \\begin{tabular}{c|l|" + "|".join(["cc" for _ in class_labels]) + "}",
        "        \\toprule",
        "        \\multirow{2}{*}{\\textbf{Classifier}} & \\multirow{2}{*}{\\textbf{Attribution Method}}"
    ]
    for i, cl in enumerate(class_labels):
        header.append(f"        & \\multicolumn{{2}}{{c}}{{\\textbf{{Class {i}}}}} " + ("\\\\" if i == len(class_labels) - 1 else ""))
    header.append("        \\cmidrule(lr){3-" + str(2 + 2*len(class_labels)) + "}")
    header.append("        & & " + " & ".join(["AIC~↑ & SIC~↑"] * len(class_labels)) + " \\\\")
    header.append("        \\midrule")
    header.append("        \\multicolumn{2}{c|}{\\textbf{Random}} & " + " & ".join(["--- & ---"] * len(class_labels)) + " \\\\")
    header.append("        \\midrule")

    body = []
    for clf in ['CLAM', 'MLP']:
        df_clf = df[(df['classifier'] == clf)]
        body.append(f"        \\multirow{{5}}{{*}}{{{clf}}}")
        for idx, method in enumerate(['g', 'ig', 'idg', 'eg', 'cig']):
            line = []
            for class_id in range(len(class_labels)):
                row = df_clf[(df_clf['method'] == method) & (df_clf['pred_label'] == class_id)]
                if not row.empty:
                    aic, sic = format_entry(row.iloc[0])
                else:
                    aic, sic = "---", "---"
                line.append(f"{aic} & {sic}")
            method_name = f"\\textbf{{CIG (ours)}}" if method == 'cig' else method.upper()
            prefix = "            &" if idx > 0 else " "
            body.append(f"{prefix} {method_name:<12} & " + " & ".join(line) + " \\\\")

        body.append("        \\midrule")

    footer = [
        "    \\end{tabular}",
        f"    \\caption{{Attribution results for each class in \\textbf{{{dataset}}}.}}",
        f"    \\label{{tab:{dataset.lower().replace('-', '_')}_classes}}",
        "\\end{table*}"
    ]

    return "\n".join(header + body + footer)

# Simulated example usage
camelyon_df = data[(data['dataset'] == 'Camelyon16') & (data['pred_label'] == 1)]
tcga_lung_df = data[(data['dataset'] == 'TCGA-Lung')]
tcga_rcc_df = data[(data['dataset'] == 'TCGA-RCC')]

latex_camelyon = render_camelyon_tumor(camelyon_df)
latex_tcga_lung = render_tcga_table(tcga_lung_df, "TCGA-Lung", ["LUAD", "LUSC"])
latex_tcga_rcc = render_tcga_table(tcga_rcc_df, "TCGA-RCC", ["KIRP", "KIRC", "KICH"])

# Output the result
latex_combined = latex_camelyon + "\n\n" + latex_tcga_lung + "\n\n" + latex_tcga_rcc
latex_combined[:1000]  # Show preview only

