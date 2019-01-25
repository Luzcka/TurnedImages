# -*- coding: utf-8 -*-

"""
Usage:
    testme.py <images directory> <truth.csv> <outpu directory>
"""


import sys
import os
import pandas as pd
import numpy as np
import csv
import PIL as pil
from keras.models import load_model
from keras.preprocessing import image
from keras import optimizers



LABELS = {"upright":1, "rotated_left":0, "rotated_right":2, "upside_down":3}
MODEL = "./turnedimages.h5"
PREDICTION = "./test.preds.csv"
IMAGES_OUTPUT_DIR = "./output"


def get_file_names(images_source):
    file_names = []
    for file_name in os.listdir(images_source):
        fullname = os.path.join(images_source, file_name)
        file_names.append(fullname)
    return file_names


def load_images(file_names):
    images = []
    for fullname in file_names:
        img = image.load_img(fullname, target_size=(64, 64))
        pix = np.array(img)
        pic = np.expand_dims(pix, axis=0)
        images.append(pic)
    return images


def get_label(value):
    label = list(LABELS.keys())[list(LABELS.values()).index(value)]
    return label


def rotated_left():
    #It needs to rotate right
    return -90


def rotated_right():
    #It needs to rotate left
    return 90


def upside_down():
    #It needs to rotate to standing position
    return 180


def upright():
    #Do nothing
    return 0


def correct_images(data, output_dir):
    for key, value in data.items():
        LABELS = {
            0: rotated_left,
            1: upright,
            2: rotated_right,
            3: upside_down}
        func = LABELS.get(value, lambda: 1)
        angle_to_rotate = func()
        im = pil.Image.open(key)
        rotated = im.rotate(angle_to_rotate)
        rotated.save(os.path.join(output_dir, os.path.basename(key)))



def test_model(prediction_source, prediction_truth, output_dir):
    model = load_model(MODEL)
    model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
    file_names = get_file_names(prediction_source)
    images = load_images(file_names)
    images = np.vstack(images)
    classes = model.predict_classes(images, batch_size=10)
    images_status = dict(zip(file_names, classes))
    with open(PREDICTION, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(["fn", "label"])
        for key, value in images_status.items():
            writer.writerow([os.path.basename(key), get_label(value)])
    correct_images(images_status, output_dir)
            




def main(args):
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    prediction_source = sys.argv[1]
    prediction_truth = sys.argv[2]
    output_dir = sys.argv[3]
    test_model(prediction_source, prediction_truth, output_dir)


if __name__ == '__main__':
    main(sys.argv)