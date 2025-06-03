import h5py

file_path = "/home/tyler/Desktop/data/h5_files/normal_001.h5"

with h5py.File(file_path, 'r') as f:
    for key in f.keys():
        print(f"Key '{key}' size: {f[key].shape}")