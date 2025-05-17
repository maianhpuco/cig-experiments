# import os
# import pandas as pd
# from torch.utils.data import Dataset

# class Generic_MIL_Dataset(Dataset):
#     def __init__(self, 
#                  csv_path=None, 
#                  data_dir=None, 
#                  label_dict=None, 
#                  case_ids=None, 
#                  labels=None,
#                  shuffle=False, 
#                  seed=1, 
#                  print_info=False,
#                  patient_strat=False,
#                  ignore=[]):
#         self.data_dir = data_dir
#         self.label_dict = label_dict
#         self.ignore = ignore
#         self.seed = seed

#         # Use provided case_ids and labels directly (custom split mode)
#         if case_ids is not None and labels is not None:
#             self.slides = case_ids
#             self.targets = labels
#         elif csv_path is not None:
#             df = pd.read_csv(csv_path)
#             self.slides = df['slide_id'].tolist()
#             self.targets = df['label'].tolist()
#         else:
#             raise ValueError("Must provide either (csv_path) or (case_ids and labels).")

#         if print_info:
#             print("Loaded {} samples from {}".format(len(self.slides), data_dir))

#     def __len__(self):
#         return len(self.slides)

#     def __getitem__(self, idx):
#         slide_id = self.slides[idx]
#         label = self.targets[idx]

#         # Features are assumed to be stored as <slide_id>.pt in data_dir
#         feat_path = os.path.join(self.data_dir, f"{slide_id}.pt")

#         # torch.load here or np.load depending on your format
#         import torch
#         features = torch.load(feat_path)  # Should be a tensor

#         return features, label


# def return_splits_custom(csv_path, data_dir, label_dict, seed=1, print_info=False):
#     df = pd.read_csv(csv_path)

#     def create_dataset(col_case, col_label):
#         case_ids = df[col_case].dropna().tolist()
#         labels = df[col_label].dropna().astype(int).tolist()
#         return Generic_MIL_Dataset(case_ids=case_ids, labels=labels,
#                                    data_dir=data_dir,
#                                    label_dict=label_dict,
#                                    shuffle=False, seed=seed,
#                                    print_info=print_info)

#     train_dataset = create_dataset('train', 'train_label')
#     val_dataset = create_dataset('val', 'val_label')
#     test_dataset = create_dataset('test', 'test_label')

#     return train_dataset, val_dataset, test_dataset 


import os
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset

class Generic_MIL_Dataset(Dataset):
    def __init__(self,
                 data_dir,
                 csv_path=None,
                 label_dict=None,
                 case_ids=None,
                 labels=None,
                 seed=1,
                 print_info=False,
                 ignore=[],
                 **kwargs):
        """
        Initialize the dataset for MIL classification, supporting .pt and .h5 feature files.
        
        Args:
            data_dir (str or dict): Directory containing feature files (.pt or .h5).
                                   If dict, maps source names to paths (e.g., {'camelyon16': 'path'}).
            csv_path (str, optional): Path to CSV with 'slide_id' and 'label' columns.
            label_dict (dict): Maps label strings to integers (e.g., {'normal': 0, 'tumor': 1}).
            case_ids (list, optional): List of slide IDs (used with return_splits_custom).
            labels (list, optional): List of labels (used with return_splits_custom).
            shuffle (bool): Shuffle data (not used).
            seed (int): Random seed.
            print_info (bool): Print dataset info.
            patient_strat (bool): Stratify by patient (not used).
            ignore (list): Slides to ignore.
            **kwargs: Additional arguments (for compatibility).
        """
        self.data_dir = data_dir
        self.label_dict = label_dict
        self.ignore = ignore
        self.seed = seed
        self.use_h5 = False
        self.kwargs = kwargs

        # Load slides and labels
        if case_ids is not None and labels is not None:
            self.slide_data = pd.DataFrame({
                'slide_id': case_ids,
                'label': labels
            })
        elif csv_path is not None:
            df = pd.read_csv(csv_path)
            self.slide_data = df[['slide_id', 'label']].copy()
        else:
            raise ValueError("Must provide either (csv_path) or (case_ids and labels).")

        # Filter out ignored slides
        self.slide_data = self.slide_data[~self.slide_data['slide_id'].isin(ignore)]

        if print_info:
            print("Loaded {} samples from {}".format(len(self.slide_data), data_dir))

    def load_from_h5(self, toggle):
        """Toggle between .pt and .h5 file loading."""
        self.use_h5 = toggle

    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'].iloc[idx]
        label = self.slide_data['label'].iloc[idx]

        # Determine data directory
        if isinstance(self.data_dir, dict):
            # Assume single source for simplicity; extend if needed
            source = self.kwargs.get('source', list(self.data_dir.keys())[0])
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not data_dir:
            return slide_id, label

        if not self.use_h5:
            full_path = os.path.join(data_dir, 'pt_files', f"{slide_id}.pt")
            features = torch.load(full_path)
            return features, label
        else:
            full_path = os.path.join(data_dir, 'h5_files', f"{slide_id}.h5")
            with h5py.File(full_path, 'r') as hdf5_file:
                features = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]
            features = torch.from_numpy(features)
            return features, label, coords

def return_splits_custom(csv_path, data_dir, label_dict, seed=1, print_info=False):
    """Create train, val, and test datasets from a custom split CSV."""
    df = pd.read_csv(csv_path)

    def create_dataset(col_case, col_label):
        case_ids = df[col_case].dropna().tolist()
        labels = df[col_label].dropna().astype(int).tolist()
        return Generic_MIL_Dataset(
            data_dir=data_dir,
            case_ids=case_ids,
            labels=labels,
            label_dict=label_dict,
            shuffle=False,
            seed=seed,
            print_info=print_info
        )

    train_dataset = create_dataset('train', 'train_label')
    val_dataset = create_dataset('val', 'val_label')
    test_dataset = create_dataset('test', 'test_label')

    return train_dataset, val_dataset, test_dataset


