# -*- coding: utf-8 -*-

"""
Usage:
    trainme.py <images directory> <truth.csv>
"""

import sys
import pandas as pd
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import numpy as np


LABELS = ["upright", "rotated_left", "rotated_right", "upside_down"]
MODEL = "./turnedimages.h5"

def train_model(train_source, train_truth):
    df = pd.read_csv(train_truth)
    datagen = ImageDataGenerator(rescale=1./255., validation_split=0.25)
    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=train_source,
        x_col="fn",
        y_col="label",
        subset="training",
        batch_size=5,
        shuffle=True,
        class_mode="categorical",
        target_size=(64,64))
    valid_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=train_source,
        x_col="fn",
        y_col="label",
        subset="validation",
        batch_size=5,
        shuffle=True,
        class_mode="categorical",
        target_size=(64,64))    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64,64,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])  

    STEP_SIZE_TRAIN=train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n // valid_generator.batch_size
    model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=2)
 
    model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID) 
    model.save(MODEL)



def main(args):
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    train_source = sys.argv[1]
    train_truth = sys.argv[2]
    train_model(train_source, train_truth)

if __name__ == '__main__':
    main(sys.argv)