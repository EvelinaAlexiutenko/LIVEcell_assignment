import numpy as np


def predict(model, image_input):
    predict = model.predict(np.array([image_input]))
    return predict
