import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from collections import deque
from PIL import Image


def line_segmenter(binary, distance = 5, prominence=50):
    h, w = binary.shape

    # Make a mask slightly larger than the image (required by floodFill)
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood fill from each corner
    cv2.floodFill(binary, mask, (0, 0), 0)          
    cv2.floodFill(binary, mask, (w - 1, 0), 0)      
    cv2.floodFill(binary, mask, (0, h - 1), 0)     
    cv2.floodFill(binary, mask, (w - 1, h - 1), 0) 

    vertical_density = np.sum(binary > 0, axis=0)

    peaks, _ = find_peaks(vertical_density, distance=distance, prominence=prominence)

    results_half = peak_widths(vertical_density, peaks, rel_height=0.3)

    left_ips = results_half[2].astype(int)
    right_ips = results_half[3].astype(int)

    line_regions = np.zeros_like(binary)
    visited = np.zeros_like(binary, dtype=bool)
    line_bounding_boxes = []

    def is_valid(y, x):
        return 0 <= x < w and 0 <= y < h and not visited[y, x] and binary[y, x] > 0

    for left, right in zip(left_ips, right_ips):
        coords = []
        for x in range(left, right + 1):
            for y in range(h):
                if binary[y, x] > 0 and not visited[y, x]:
                    queue = deque()
                    queue.append((y, x))

                    while queue:
                        cy, cx = queue.popleft()
                        if not is_valid(cy, cx):
                            continue
                        visited[cy, cx] = True
                        line_regions[cy, cx] = 255
                        coords.append((cy, cx))

                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                ny, nx = cy + dy, cx + dx
                                if is_valid(ny, nx):
                                    queue.append((ny, nx))

        if coords:
            ys, xs = zip(*coords)
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            line_bounding_boxes.append((x_min, y_min, x_max, y_max))

    return line_bounding_boxes


def segment_words_from_line(gray, binary, line_bbox, overlap_threshold=0.3, adiya = False, min_height_ratio=1, min_area=30):
    """
    Segments words from a line image using connected components and merges nearby bounding boxes.
    
    Args:
        gray (np.ndarray): Grayscale version of the original image.
        binary (np.ndarray): Binary version of the image.
        line_bbox (tuple): Bounding box of the line (x_min, y_min, x_max, y_max).
        overlap_threshold (float): Threshold to merge overlapping word boxes.

    Returns:
        List[Tuple[int, int, int, int]]: List of word bounding boxes in original image coordinates.
    """
    line_x_min, line_y_min, line_x_max, line_y_max = line_bbox
    line_width = line_x_max - line_x_min

    line_piece = binary[line_y_min:line_y_max, line_x_min:line_x_max]
    binary_line = (line_piece > 0).astype(np.uint8)

    # Get connected components with stats
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_line)

    # Create a clean mask
    clean_mask = np.zeros_like(binary_line)

    # Minimum area threshold (tune as needed)
    

    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == i] = 255
    line_piece = clean_mask
    num_labels, labels = cv2.connectedComponents(line_piece)

    height, width = line_piece.shape
    expanded_bboxes = []

    for label in range(1, num_labels):
        y_coords, x_coords = np.where(labels == label)
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        ratio =  (y_max - y_min) / (x_max - x_min) 
        # if ratio > min_height_ratio:
            # expanded_bboxes.append((x_min, y_min, x_max, y_max))
        expanded_bboxes.append((x_min, y_min, x_max, y_max))

    def get_overlap_area(box1, box2):
        x_min1, y_min1, x_max1, y_max1 = box1
        x_min2, y_min2, x_max2, y_max2 = box2

        overlap_x_min = max(x_min1, x_min2)
        overlap_x_max = min(x_max1, x_max2)
        overlap_y_min = max(y_min1, y_min2)
        overlap_y_max = min(y_max1, y_max2)

        if overlap_x_max > overlap_x_min and overlap_y_max > overlap_y_min:
            return (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
        return 0

    merged_bboxes = []
    for box in expanded_bboxes:
        overlap_found = False
        for i, merged_box in enumerate(merged_bboxes):
            overlap_area = get_overlap_area(merged_box, box)
            box_area = (box[2] - box[0]) * (box[3] - box[1])

            if overlap_area / box_area > overlap_threshold:
                new_box = (
                    min(merged_box[0], box[0]),
                    min(merged_box[1], box[1]),
                    max(merged_box[2], box[2]),
                    max(merged_box[3], box[3]),
                )
                merged_bboxes[i] = new_box
                overlap_found = True
                break

        if not overlap_found:
            # Expand box horizontally to whole line if no overlap found
            box = (0, box[1], line_width, box[3])
            merged_bboxes.append(box)

    # Convert bboxes to original image coordinates
    word_bboxes_in_full_image = [
        (box[0] + line_x_min, box[1] + line_y_min, box[2] + line_x_min, box[3] + line_y_min)
        for box in merged_bboxes
    ]
    if adiya:
        final_bboxes = []

        for i, box in enumerate(word_bboxes_in_full_image):
            box_height = box[3] - box[1]
            box_width = box[2] - box[0]
            
            # Merge with the previous box if height < 2*width
            if i > 0 and box_height < min_height_ratio * box_width:
                prev_box = final_bboxes[-1]
                new_box = (
                    min(prev_box[0], box[0]),
                    min(prev_box[1], box[1]),
                    max(prev_box[2], box[2]),
                    max(prev_box[3], box[3]),
                )
                final_bboxes[-1] = new_box
            else:
                final_bboxes.append(box)
            
        return final_bboxes

    return word_bboxes_in_full_image


