
import os


DATASET_DIR = "/path/to/your/dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
WEIGHTS_DIR = os.path.join(os.path.dirname(DATASET_DIR), 'weights')

TEST_DIRECTORY = os.path.join(DATASET_DIR, "test")
GT_TEST_DIRECTORY = os.path.join(DATASET_DIR, "test", "masks")
TRAIN_DIRECTORY = os.path.join(DATASET_DIR, "train")
GT_TRAIN_DIRECTORY = os.path.join(DATASET_DIR, "train", "masks")
TRAIN_DIRECTORY = os.path.join(DATASET_DIR, "val")
GT_VAL_DIRECTORY = os.path.join(DATASET_DIR, "val", "masks")

model_9_epochs_path = os.path.join(WEIGHTS_DIR, "my_model_9.keras")
model_30_epochs_path = os.path.join(WEIGHTS_DIR, "my_model_20.keras")

SIZE = 256 
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1

# Global mean and standard deviation for normalization
GLOBAL_MEAN = 128.02333355732696
GLOBAL_STD = 7.079153483606509