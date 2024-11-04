import os
import shutil
import json

from cfg import DATASET_DIR, IMAGES_DIR


for json_file in os.listdir(DATASET_DIR):
    if json_file.endswith('.json'):

        output_dir = os.path.basename(json_file).split('_')[2].split('.')[0] # get the name of the split
        output_dir =  os.path.join(DATASET_DIR, output_dir)
        os.makedirs(output_dir, exist_ok=True)


        with open(os.path.join(DATASET_DIR, json_file), 'r') as f:
            coco_data = json.load(f)


        for img_data in coco_data.get('images', []):
            file_name = img_data.get('file_name')
            if file_name:
                src_file_path = os.path.join(IMAGES_DIR, file_name)

                if os.path.exists(src_file_path):
                    shutil.copy(src_file_path, output_dir)
                    print(f"Copied {file_name} to {output_dir}")
                else:
                    print(f"File {file_name} not found in {IMAGES_DIR}")

        print(f"All files successfully copied to {output_dir}")
