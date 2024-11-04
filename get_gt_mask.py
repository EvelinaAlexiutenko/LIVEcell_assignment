import os
import numpy as np
from PIL import Image
from cfg import SIZE


def get_gt_mask(image_path, size=SIZE):
    mask_dirpath = os.path.join(os.path.dirname(image_path), "masks")
    gt_mask_path = os.path.join(
        mask_dirpath, f"{os.path.splitext(os.path.basename(image_path))[0]}_mask.tif"
    )
    gt_mask_image = Image.open(gt_mask_path).resize((size, size))
    gt_mask = np.array(gt_mask_image)
    return gt_mask
