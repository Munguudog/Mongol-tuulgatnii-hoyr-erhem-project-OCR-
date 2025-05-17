import cv2
import os
import uuid
import tempfile
from PIL import Image
import numpy as np

def estimate_vertical_skew_and_draw(gray, show_debug=True):
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                            minLineLength=100, maxLineGap=10)
    
    angles = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
            
            # Consider only nearly vertical lines (angle near 90°)
            if 75 <= angle <= 105:
                angles.append(angle)
                # if show_debug:
                #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if not angles:
        print("No vertical lines detected.")
        return gray

    median_angle = np.median(angles)
    # print(angles)
    skew_from_vertical = 90 - median_angle 
    print(f"Detected skew: {skew_from_vertical:.2f}° (to vertical)")

    # Rotate back to vertical
    if skew_from_vertical == 0.0:
        return gray
    gray_pil = Image.fromarray(gray).convert("L")
    rotated = gray_pil.rotate(skew_from_vertical, resample=Image.BICUBIC, expand=True, fillcolor=0)

    return np.array(rotated)

def preprocess_image(image):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 2. Rotate 90 degrees counter-clockwise
    # gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 3. Correct skew using your custom function
    gray = estimate_vertical_skew_and_draw(gray)

    # 4. Binarize using Otsu’s method (text becomes white on black background)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 5. Save temp binary image for testing/debugging
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"binary_{uuid.uuid4().hex}.png")
    temp_path = "adiya_test.png"
    cv2.imwrite(temp_path, binary)
    print(f"Saved preprocessed image to {temp_path}")

    return binary
