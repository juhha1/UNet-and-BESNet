"""
There is no Edge images for this dataset.
Use edge detecion to crate edge images and save them.
"""

import os, glob, tqdm, cv2
import numpy as np
from PIL import Image

base_dir = os.path.dirname(os.path.realpath(__file__))
mask_files = glob.glob(os.path.join(base_dir, '*', '*', '*_mask.tif'))

for file in tqdm.tqdm(mask_files):
    image_mask = np.asarray(Image.open(file))
    image_edge = cv2.Canny(image_mask, 0, 1)
    image_edge = Image.fromarray(image_edge.astype(np.uint8))
    dir_name = os.path.dirname(file)
    base_name = os.path.basename(file)
    base_name = '_'.join(base_name.split('_')[:-1]) + '_edge.tif'
    image_edge.save(os.path.join(dir_name, base_name))