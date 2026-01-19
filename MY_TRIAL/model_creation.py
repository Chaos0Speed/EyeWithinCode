
from tensorflow.keras import layers, Model  #type:ignore
import tensorflow as tf     #type:ignore
import numpy as np
import os
import cv2
import pandas as pd

class L1Dist(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    

def create_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(105, 105, 1)),
        layers.MaxPooling2D((2, 2), padding='same'),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2), padding='same'),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2), padding='same'),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.Flatten(),

        layers.Dense(4096, activation='sigmoid')
    ])
    input_image = layers.Input(shape=(105, 105, 1), name='input_image')
    anchor_image = layers.Input(shape=(105, 105, 1), name='anchor_image')
    l1dist = L1Dist()
    distances = l1dist(model(input_image), model(anchor_image))
    outputs = layers.Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, anchor_image], outputs=outputs)

if __name__ == "__main__":
    siamese_model = create_model()
    siamese_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy','Precision','Recall'])
    siamese_model.summary()
    siamese_model.save('untrained_faceid_model.keras')
    print("Model Created and Saved. \n WARNING: Overwrites existing model with the same name AND THE MODEL IS UNTRAINED!")