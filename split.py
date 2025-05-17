# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
# # Load labels
# df = pd.read_csv("/home/mvu9/processing_datasets/processing_camelyon16/camelyon16_labels.csv")
# slide_ids = df['slide_id'].tolist()
# labels = df['label'].map({'normal': 0, 'tumor': 1}).tolist()
# # Generate 10 folds
# kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# for i, (train_val_idx, test_idx) in enumerate(kf.split(slide_ids, labels)):
#     train_val_slides = [slide_ids[j] for j in train_val_idx]
#     test_slides = [slide_ids[j] for j in test_idx]
#     n_val = len(train_val_slides) // 5
#     val_slides = train_val_slides[:n_val]
#     train_slides = train_val_slides[n_val:]
#     split_df = pd.DataFrame({
#         'train': pd.Series(train_slides),
#         'val': pd.Series(val_slides),
#         'test': pd.Series(test_slides)
#     })
#     split_df.to_csv(f"/home/mvu9/processing_datasets/processing_camelyon16/splits/task_1_tumor_vs_normal_100/splits_{i}.csv", index=False)