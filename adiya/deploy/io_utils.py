from PIL import Image
import numpy as np

def load_image(file_storage):
    image = Image.open(file_storage).convert("RGB")
    return np.array(image)