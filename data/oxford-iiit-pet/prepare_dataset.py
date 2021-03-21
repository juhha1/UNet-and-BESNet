import os, glob, tqdm
import numpy as np
from PIL import Image

base_dir = os.path.dirname(os.path.realpath(__file__))
mask_files = glob.glob(os.path.join(base_dir, 'annotations', 'trimaps', r'*.png'))

os.makedirs(os.path.join(base_dir, 'masks'), exist_ok = True)
os.makedirs(os.path.join(base_dir, 'edges'), exist_ok = True)

for file in tqdm.tqdm(mask_files):
    image_arr = np.array(Image.open(file))
    mask = Image.fromarray(((image_arr != 2) * 1).astype(np.uint8))
    edge = Image.fromarray(((image_arr == 3) * 1).astype(np.uint8))
    fname = os.path.basename(file)
    mask.save(os.path.join(base_dir, 'masks', fname))
    edge.save(os.path.join(base_dir, 'edges', fname))