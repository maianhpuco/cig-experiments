from src.datasets.classification.camelyon16 import Generic_MIL_Dataset, return_splits_custom


split_csv_path = "./camelyon16_csv_splits_camil/splits_0.csv"
# Run the split function
train_set, val_set, test_set = return_splits_custom(
    csv_path=split_csv_path,
    data_dir='/home/mvu/processing_datasets/processing_camelyon16/features_fp',
    label_dict={'normal': 0, 'tumor': 1},  # This won't affect direct labels
    seed=42,
    print_info=True
)

# Print a few samples
for split_name, dataset in zip(["Train", "Val", "Test"], [train_set, val_set, test_set]):
    print(f"\n{split_name} Split:")
    for i in range(len(dataset)):
        x, y = dataset[i]
        print(f"  Sample {i}: features shape = {x.shape}, label = {y}")
