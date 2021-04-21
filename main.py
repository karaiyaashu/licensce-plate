import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
from keras.models import model_from_json
import os
from image_utils import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
plt.style.use('dark_background')


def get_names():
    i =1
    pre_trained_model = load_keras_model('model_LicensePlate')
    model = pre_trained_model
    for root, dirs, files in os.walk(os.path.abspath("images")):
        for file in files:
            path_string = os.path.join(root, file)
            util(path_string, model)
            print(i)
            print(path_string)
            i= i+1


def load_keras_model(model_name):
    # Load json and create model
    json_file = open('./models/{}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # Load weights into new model
    model.load_weights("./models/{}.h5".format(model_name))
    return model


if __name__ == "__main__":
    get_names()
