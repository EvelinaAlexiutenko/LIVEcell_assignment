import argparse
import json
import numpy as np
import skimage
import tifffile
import os


def create_mask(image_info, annotations, output_folder):
    # Create an empty mask as a numpy array
    mask_np = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint16)
    # Counter for the object number
    object_number = 1

    for ann in annotations:
        if ann["image_id"] == image_info["id"]:
            for seg in ann["segmentation"]:
                # Convert polygons to a binary mask and add it to the main mask
                rr, cc = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)
                mask_np[rr, cc] = object_number
                object_number += 1

    mask_path = os.path.join(
        output_folder, image_info["file_name"].replace(".tif", "_mask.tif")
    )
    tifffile.imsave(mask_path, mask_np)

    print(f"Saved mask for {image_info['file_name']} to {mask_path}")


def main(json_file, mask_output_folder, image_output_folder, original_image_dir):
    # Load COCO JSON annotations
    with open(json_file, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    # Ensure the output directories exist
    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    for img in images:
        create_mask(img, annotations, mask_output_folder)
        original_image_path = os.path.join(original_image_dir, img["file_name"])

        new_image_path = os.path.join(
            image_output_folder, os.path.basename(original_image_path)
        )
        print(f"Copied original image to {new_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate masks from COCO annotations and organize images.")
    parser.add_argument("--json_file", type=str, help="Path to the COCO JSON annotations file.")
    parser.add_argument("--mask_output_folder", type=str, help="Folder to save the generated masks.")
    parser.add_argument("--image_output_folder", type=str, help="Folder to save the original images.")
    parser.add_argument("--original_image_dir", type=str, help="Directory containing the original images.")

    args = parser.parse_args()
    main(args.json_file, args.mask_output_folder, args.image_output_folder, args.original_image_dir)
