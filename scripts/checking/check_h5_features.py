import os
import h5py

def list_h5_keys(h5_dir):
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]

    if not h5_files:
        print("No H5 files found.")
        return

    for h5_file in h5_files:
        file_path = os.path.join(h5_dir, h5_file)
        print(f"\ Keys in file: {h5_file}")
        try:
            with h5py.File(file_path, 'r') as f:
                keys = list(f.keys())
                for key in keys:
                    print(f"  - {key}")
        except Exception as e:
            print(f"  Error reading {h5_file}: {e}")

# Example usage

list_h5_keys("/home/mvu9/processing_datasets/processing_camelyon16/patches")

