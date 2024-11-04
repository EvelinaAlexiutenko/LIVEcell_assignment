from cfg import IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH
from simple_unet_model import simple_unet_model


def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


def load_model(model_path):
    model = get_model()
    model.load_weights(model_path)
    return model
