import pandas as pd
import os

# Paths
split_dir = "/home/mvu9/processing_datasets/processing_camelyon16/splits/task_1_tumor_vs_normal_100"
input_split_file = "/home/mvu9/folder_04_ma/wsi-data/camelyon16_csv_splits_camil/splits_test.csv"
output_split_dir = split_dir
slide_list_path = "/home/mvu9/processing_datasets/processing_camelyon16/process_list_autogen.csv"

# Ensure output directory exists
os.makedirs(output_split_dir, exist_ok=True)

# Read slide list for mapping
slide_df = pd.read_csv(slide_list_path)
all_slides = slide_df['slide_id'].str.replace('.tif', '').tolist()

# Read split file
df = pd.read_csv(input_split_file, index_col=0)

# Extract slides
train_slides = df['train'].dropna().tolist()
val_slides = df['val'].dropna().tolist()
test_slides = df['test'].dropna().tolist()
test_labels = df['test_label'].dropna().tolist()

# Map test_* to tumor_*/normal_*
train_val_slides = set(train_slides + val_slides)
available_slides = [s for s in all_slides if s not in train_val_slides]
if len(available_slides) < len(test_slides):
    raise ValueError(f"Not enough slides ({len(available_slides)}) for test set ({len(test_slides)})")
mapped_test_slides = []
for i, (test_id, test_label) in enumerate(zip(test_slides, test_labels)):
    slide_id = available_slides[i]
    mapped_test_slides.append(slide_id)

# Create CLAM-format split
split_df = pd.DataFrame({
    'train': pd.Series(train_slides),
    'val': pd.Series(val_slides),
    'test': pd.Series(mapped_test_slides)
})

# Save split
output_path = os.path.join(output_split_dir, "splits_0.csv")
split_df.to_csv(output_path, index=False)
print(f"Saved splits_0.csv to {output_path}")

# Placeholder for additional folds (optional)
for i in range(1, 10):
    output_path = os.path.join(output_split_dir, f"splits_{i}.csv")
    split_df.to_csv(output_path, index=False)
    print(f"Saved placeholder splits_{i}.csv to {output_path}")