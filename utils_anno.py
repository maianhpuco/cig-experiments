import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from rtree import index
import os

PATCH_SIZE = 256  # Patch size in pixels (level 0 resolution)

def parse_xml(file_path):
    """Parse XML file and return root element."""
    try:
        tree = ET.parse(file_path)
        return tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def extract_coordinates(file_path, save_path):
    """
    Extract (X, Y) coordinates inside tumor contours from XML annotations.
    Works in original resolution (level 0) without downscaling.
    Saves coordinates to CSV and returns DataFrame.
    """
    root = parse_xml(file_path)
    if root is None:
        print(f"[WARN] Failed to parse {file_path}")
        return None

    print(f"Processing XML: {file_path}")

    # Extract contours
    contours = []
    for annotation in root.findall(".//Annotation"):
        contour = []
        for coordinate in annotation.findall(".//Coordinate"):
            x = coordinate.attrib.get("X")
            y = coordinate.attrib.get("Y")
            if x and y:
                try:
                    contour.append((float(x), float(y)))
                except ValueError:
                    continue
        if len(contour) > 2:  # Minimum 3 points for a polygon
            if contour[0] != contour[-1]:
                contour.append(contour[0])  # Close the contour
            contours.append(contour)

    if not contours:
        print(f"[WARN] No valid contours in {file_path}")
        return None

    # Debug: Print contour details
    print(f"Number of contours: {len(contours)}")
    for i, contour in enumerate(contours):
        poly = Polygon(contour)
        print(f"Contour {i}: points={len(contour)}, area={poly.area:.2f}, bounds={poly.bounds}")

    # Create polygons
    polygons = [Polygon(contour) for contour in contours]

    # Get bounds for grid generation
    all_bounds = [polygon.bounds for polygon in polygons]
    min_x = min(b[0] for b in all_bounds)
    min_y = min(b[1] for b in all_bounds)
    max_x = max(b[2] for b in all_bounds)
    max_y = max(b[3] for b in all_bounds)

    # Generate grid at patch resolution (level 0)
    x_patches = np.arange(np.floor(min_x / PATCH_SIZE) * PATCH_SIZE,
                         np.ceil(max_x / PATCH_SIZE) * PATCH_SIZE + PATCH_SIZE, PATCH_SIZE)
    y_patches = np.arange(np.floor(min_y / PATCH_SIZE) * PATCH_SIZE,
                         np.ceil(max_y / PATCH_SIZE) * PATCH_SIZE + PATCH_SIZE, PATCH_SIZE)

    inside_points = []
    spatial_index = index.Index()
    for i, polygon in enumerate(polygons):
        spatial_index.insert(i, polygon.bounds)

    # Check patch centers for containment
    total_patches = len(x_patches) * len(y_patches)
    with tqdm(total=total_patches, desc="Processing Patches", ncols=100) as pbar:
        for x_start in x_patches:
            for y_start in y_patches:
                # Use patch center to check if patch overlaps tumor
                patch_center = Point(x_start + PATCH_SIZE / 2, y_start + PATCH_SIZE / 2)
                possible_polygons = [polygons[i] for i in spatial_index.intersection(
                    (x_start, y_start, x_start + PATCH_SIZE, y_start + PATCH_SIZE))]
                if any(polygon.contains(patch_center) for polygon in possible_polygons):
                    inside_points.append((x_start, y_start))
                pbar.update(1)

    if not inside_points:
        print(f"[WARN] No points found inside contours for {file_path}")
        return None

    # Convert to DataFrame
    df_inside_points = pd.DataFrame({
        "File": os.path.basename(file_path),
        "X": [p[0] for p in inside_points],
        "Y": [p[1] for p in inside_points]
    })

    # Debug: Print coordinate range
    print(f"XML coordinates range: X=[{df_inside_points['X'].min()}, {df_inside_points['X'].max()}], "
          f"Y=[{df_inside_points['Y'].min()}, {df_inside_points['Y'].max()}]")

    df_inside_points.to_csv(save_path, index=False)
    return df_inside_points

def check_coor(x, y, box, patch_size=PATCH_SIZE):
    """
    Check if point (x, y) is inside a patch defined by top-left corner (px, py).
    """
    px, py = box
    return px <= x < px + patch_size and py <= y < py + patch_size

def check_xy_in_coordinates_fast(coordinates_xml, coordinates_h5, tolerance=10, patch_size=PATCH_SIZE):
    """
    Match XML coordinates to H5 patches using R-tree with tolerance.
    Returns binary mask (1 if patch contains tumor, 0 otherwise).
    """
    if len(coordinates_h5) == 0:
        print("[WARN] Empty H5 coordinate list")
        return np.zeros(0, dtype=np.int8)

    label = np.zeros(len(coordinates_h5), dtype=np.int8)

    # Debug: Print H5 coordinate range
    print(f"H5 coordinates range: X=[{coordinates_h5[:,0].min()}, {coordinates_h5[:,0].max()}], "
          f"Y=[{coordinates_h5[:,1].min()}, {coordinates_h5[:,1].max()}]")

    try:
        rtree_index = index.Index(
            (i, (x, y, x, y), None) for i, (x, y) in enumerate(coordinates_h5)
        )
    except Exception as e:
        print(f"[ERROR] Failed to create R-tree index: {e}")
        return label

    xy_pairs = np.column_stack((coordinates_xml["X"], coordinates_xml["Y"]))
    matches_found = 0
    for x, y in xy_pairs:
        # Search with tolerance
        search_bounds = (x - tolerance, y - tolerance, x + tolerance, y + tolerance)
        possible_matches = list(rtree_index.intersection(search_bounds))
        for box_index in possible_matches:
            if check_coor(x, y, coordinates_h5[box_index], patch_size):
                label[box_index] = 1
                matches_found += 1
                # Debug: Print first few matches
                if matches_found <= 5:
                    print(f"Match: XML ({x}, {y}) in patch ({coordinates_h5[box_index][0]}, {coordinates_h5[box_index][1]})")

    print(f"[INFO] Total matches: {matches_found}")
    print(f"[INFO] Label distribution: 0s = {(label == 0).sum()}, 1s = {(label == 1).sum()}")
    return label