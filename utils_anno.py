import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from rtree import index  # R-tree for fast spatial lookup
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib  # Ensures tqdm updates correctly with joblib 
import os 
import time 
import multiprocessing
 
PATCH_SIZE = 256  

def parse_xml(file_path):
    """ Parse XML file and return root element. """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return root
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def downscale_coordinates(contour, scale_factor):
    """ Downscale contour coordinates for faster processing. """
    return [(x / scale_factor, y / scale_factor) for x, y in contour]

def upscale_coordinates(points, scale_factor):
    """ Upscale points back to original size. """
    return [(int(x * scale_factor), int(y * scale_factor)) for x, y in points]

def extract_coordinates(file_path, save_path):
    """
    Fast extraction of (X, Y) coordinates **inside** multiple contours using downscaling and R-tree.
    """
    root = parse_xml(file_path)
    if root is None:
        return None  # Skip if parsing failed

    print(f"Processing XML: {file_path}")

    # Extract multiple contours
    contours = []
    for annotation in root.findall(".//Annotation"):  # Adjust based on XML structure
        contour = []
        for coordinate in annotation.findall(".//Coordinate"):
            x = coordinate.attrib.get("X")
            y = coordinate.attrib.get("Y")
            if x and y:
                contour.append((float(x), float(y)))

        if contour:
            if contour[0] != contour[-1]:
                contour.append(contour[0])  # Ensure the contour is closed
            contours.append(contour)

    if not contours:
        return None  # No valid contours found

    # Downscale each contour separately
    downscaled_contours = [downscale_coordinates(contour, PATCH_SIZE) for contour in contours]

    # Create multiple polygons
    polygons = [Polygon(contour) for contour in downscaled_contours if len(contour) > 2]

    # Generate downscaled grid (treat each patch as a "pixel")
    all_bounds = [polygon.bounds for polygon in polygons]
    min_x = min(b[0] for b in all_bounds)
    min_y = min(b[1] for b in all_bounds)
    max_x = max(b[2] for b in all_bounds)
    max_y = max(b[3] for b in all_bounds)

    x_patches = np.arange(np.floor(min_x), np.ceil(max_x))
    y_patches = np.arange(np.floor(min_y), np.ceil(max_y))

    inside_points = []

    # Create an R-tree index to speed up spatial queries
    spatial_index = index.Index()
    for i, polygon in enumerate(polygons):
        spatial_index.insert(i, polygon.bounds)

    # Use tqdm to track progress
    with tqdm(total=len(x_patches) * len(y_patches), desc="Processing Patches", ncols=100) as pbar:
        for x_start in x_patches:
            for y_start in y_patches:
                patch_point = Point(x_start, y_start)

                # Only check polygons that intersect the bounding box
                possible_polygons = [polygons[i] for i in spatial_index.intersection((x_start, y_start, x_start, y_start))]
                
                if any(polygon.contains(patch_point) for polygon in possible_polygons):
                    inside_points.append((x_start, y_start))  # Save downscaled points
                
                pbar.update(1)

    # Upscale the points back to original resolution
    original_size_points = upscale_coordinates(inside_points, PATCH_SIZE)

    # Convert to DataFrame
    df_inside_points = pd.DataFrame({
        "File": file_path.split("/")[-1],
        "X": [p[0] for p in original_size_points],
        "Y": [p[1] for p in original_size_points]
    })
    
    basename = os.path.splitext(os.path.basename(file_path))[0]  # Correctly get file basename
    df_inside_points.to_csv(save_path, index=False)
    
    return df_inside_points

def check_coor(x, y, box):
    """
    Checks if (x, y) is inside the given bounding box.
    Format is [x, y], representing top-left corner of the patch.
    """
    px, py = box  # Each patch is a single point (top-left corner)
    return px <= x < px + 256 and py <= y < py + 256


def check_xy_in_coordinates(coordinates_xml, coordinates_h5):
    """
    Classic version: Iterate through XML and find corresponding H5 patches (using full loops + R-tree).
    coordinates_h5 should be Nx2 array: [x, y] for each patch.
    """
    length = coordinates_h5.shape[0]
    label = np.zeros(length, dtype=int)

    rtree_index = index.Index()
    for i, (x, y) in enumerate(coordinates_h5):
        rtree_index.insert(i, (x, y, x, y))

    for row in tqdm(coordinates_xml.itertuples(index=False), desc="Checking index:", total=len(coordinates_xml), ncols=100):
        x, y = row.X, row.Y
        possible_matches = list(rtree_index.intersection((x, y, x, y)))
        for idx in possible_matches:
            if check_coor(x, y, coordinates_h5[idx]):
                label[idx] = 1

    return label


def check_xy_in_coordinates_fast(coordinates_xml, coordinates_h5):
    """
    Optimized vectorized version for matching tumor coordinates from XML to H5 patches.
    H5 coordinates: [x, y] patch centers (or top-left corners), Nx2
    """
    if len(coordinates_h5) == 0:
        print("[WARN] Empty H5 coordinate list passed to R-tree.")
        return np.zeros(0, dtype=np.int8)

    label = np.zeros(len(coordinates_h5), dtype=np.int8)

    try:
        rtree_index = index.Index(
            (i, (x, y, x, y), None) for i, (x, y) in enumerate(coordinates_h5)
        )
    except Exception as e:
        print(f"[ERROR] Failed to create R-tree index: {e}")
        return label

    xy_pairs = np.column_stack((coordinates_xml["X"], coordinates_xml["Y"]))

    for x, y in xy_pairs:
        possible_matches = list(rtree_index.intersection((x, y, x, y)))
        for box_index in possible_matches:
            if check_coor(x, y, coordinates_h5[box_index]):
                label[box_index] = 1

    print(f"[INFO] Label distribution: 0s = {(label == 0).sum()}, 1s = {(label == 1).sum()}")
    return label