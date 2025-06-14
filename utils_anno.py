import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from rtree import index
import os

PATCH_SIZE = 256  # Configurable patch size (try 224 if needed)

def parse_xml(file_path):
    """Parse XML file and return root element."""
    try:
        tree = ET.parse(file_path)
        return tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def extract_coordinates(file_path, save_path, h5_coords=None, patch_size=PATCH_SIZE):
    """
    Extract (X, Y) coordinates inside tumor contours from XML.
    Uses H5 patch coordinates as the grid to ensure alignment.
    Saves to CSV and returns DataFrame.
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
        if len(contour) > 2:
            if contour[0] != contour[-1]:
                contour.append(contour[0])
            contours.append(contour)

    if not contours:
        print(f"[WARN] No valid contours in {file_path}")
        return None

    # Debug: Contour details
    print(f"Number of contours: {len(contours)}")
    for i, contour in enumerate(contours):
        poly = Polygon(contour)
        print(f"Contour {i}: points={len(contour)}, area={poly.area:.2f}, bounds={poly.bounds}")

    # Create polygons
    polygons = [Polygon(contour) for contour in contours]

    # Use H5 coordinates as grid
    if h5_coords is not None and len(h5_coords) > 0:
        patch_coords = h5_coords  # Shape: (N, 2)
        print(f"Using H5 patch grid: {len(patch_coords)} patches")
    else:
        # Fallback: Generate grid from contour bounds
        all_bounds = [polygon.bounds for polygon in polygons]
        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)
        x_patches = np.arange(np.floor(min_x / patch_size) * patch_size,
                             np.ceil(max_x / patch_size) * patch_size + patch_size, patch_size)
        y_patches = np.arange(np.floor(min_y / patch_size) * patch_size,
                             np.ceil(max_y / patch_size) * patch_size + patch_size, patch_size)
        patch_coords = np.array([(x, y) for y in y_patches for x in x_patches])
        print(f"Generated grid: {len(patch_coords)} patches")

    inside_points = []
    spatial_index = index.Index()
    for i, polygon in enumerate(polygons):
        spatial_index.insert(i, polygon.bounds)

    # Check patches for tumor overlap
    with tqdm(total=len(patch_coords), desc="Processing Patches", ncols=100) as pbar:
        for x_start, y_start in patch_coords:
            # Check if patch region intersects any polygon
            patch_bbox = Polygon([
                (x_start, y_start),
                (x_start + patch_size, y_start),
                (x_start + patch_size, y_start + patch_size),
                (x_start, y_start + patch_size)
            ])
            possible_polygons = [polygons[i] for i in spatial_index.intersection(
                (x_start, y_start, x_start + patch_size, y_start + patch_size))]
            if any(polygon.intersects(patch_bbox) for polygon in possible_polygons):
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

    print(f"XML coordinates range: X=[{df_inside_points['X'].min()}, {df_inside_points['X'].max()}], "
          f"Y=[{df_inside_points['Y'].min()}, {df_inside_points['Y'].max()}]")
    print(f"Sample XML points: {df_inside_points[['X', 'Y']].head().to_dict('records')}")

    df_inside_points.to_csv(save_path, index=False)
    return df_inside_points

def check_coor(x, y, box, patch_size=PATCH_SIZE):
    """Check if point (x, y) is inside a patch defined by top-left corner (px, py)."""
    px, py = box
    return px <= x < px + patch_size and py <= y < py + patch_size

def check_xy_in_coordinates_fast(coordinates_xml, coordinates_h5, patch_size=PATCH_SIZE):
    """
    Match XML coordinates to H5 patches using R-tree.
    Returns binary mask (1 if patch contains tumor, 0 otherwise).
    """
    if len(coordinates_h5) == 0:
        print("[WARN] Empty H5 coordinate list")
        return np.zeros(0, dtype=np.int8)

    label = np.zeros(len(coordinates_h5), dtype=np.int8)

    print(f"H5 coordinates range: X=[{coordinates_h5[:,0].min()}, {coordinates_h5[:,0].max()}], "
          f"Y=[{coordinates_h5[:,1].min()}, {coordinates_h5[:,1].max()}]")
    print(f"Sample H5 coords: {coordinates_h5[:5]}")

    # Build R-tree with patch bounding boxes
    rtree_index = index.Index()
    for i, (px, py) in enumerate(coordinates_h5):
        rtree_index.insert(i, (px, py, px + patch_size, py + patch_size))

    xy_pairs = np.column_stack((coordinates_xml["X"], coordinates_xml["Y"]))
    matches_found = 0
    for x, y in xy_pairs:
        possible_matches = list(rtree_index.intersection((x, y, x, y)))
        for box_index in possible_matches:
            if check_coor(x, y, coordinates_h5[box_index], patch_size):
                label[box_index] = 1
                matches_found += 1
                if matches_found <= 5:
                    print(f"Match: XML ({x}, {y}) in patch ({coordinates_h5[box_index][0]}, {coordinates_h5[box_index][1]})")
            elif matches_found <= 5 and possible_matches:
                px, py = coordinates_h5[box_index]
                distance = np.sqrt((x - px)**2 + (y - py)**2)
                print(f"No match: XML ({x}, {y}), patch ({px}, {py}), distance={distance:.2f}")

    print(f"[INFO] Total matches: {matches_found}")
    print(f"[INFO] Label distribution: 0s = {(label == 0).sum()}, 1s = {(label == 1).sum()}")
    return label