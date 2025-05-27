import os
import numpy as np
import pandas as pd
import argparse
import yaml

def load_all_metrics(metrics_dir: str, fold_start: int, fold_end: int) -> pd.DataFrame:
    records = []

    for fold in range(fold_start, fold_end + 1):
        for fname in os.listdir(metrics_dir):
            if not fname.endswith(".npy"):
                continue
            if f"fold{fold}_" not in fname:
                continue

            path = os.path.join(metrics_dir, fname)
            slide_id = fname.replace(f"clam_fold{fold}_", "").replace(".npy", "")

            try:
                data = np.load(path, allow_pickle=True)
                for row in data:
                    method, aic, sic, insertion_auc, deletion_auc = row
                    records.append({
                        "Fold": fold,
                        "Slide": slide_id,
                        "Method": method,
                        "AIC": aic,
                        "SIC": sic,
                        "Insertion AUC": insertion_auc,
                        "Deletion AUC": deletion_auc
                    })
            except Exception as e:
                print(f"⚠️ Failed to load {fname}: {e}")

    return pd.DataFrame(records)

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    metrics_dir = config['paths']['metrics_dir']
    df = load_all_metrics(metrics_dir, args.fold_start, args.fold_end)

    if df.empty:
        print("⚠️ No metrics found.")
        return

    # Save CSV to current directory
    output_path = os.path.join(os.getcwd(), f"aggregated_metrics_folds_{args.fold_start}_to_{args.fold_end}.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")

    # Optional: print group summary
    print("\n🔍 Average scores per method:")
    summary = df.groupby("Method")[["AIC", "SIC", "Insertion AUC", "Deletion AUC"]].mean()
    print(summary.round(4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--fold_start', type=int, default=1)
    parser.add_argument('--fold_end', type=int, default=1)

    args = parser.parse_args()
    main(args)
