import tensorflow as tf   #type:ignore
from tensorflow.keras import Model, layers, optimizers, losses  #type:ignore
import numpy as np
import os
import cv2
from model_creation import L1Dist
#***************************************************************************************************************************************************************

def image_processing(file_path):
    print("Processing image: ", file_path)
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def data_processing():
    n = 300
    anchor = tf.data.Dataset.list_files(os.path.join('processed', 'anchor', '*.png')).take(n)
    positive = tf.data.Dataset.list_files(os.path.join('processed', 'positive', '*.png')).take(n)
    negative = tf.data.Dataset.list_files(os.path.join('processed', 'negative', '*.jpg')).take(n)

    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(n))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(n))))
    data = positives.concatenate(negatives)

    data.cache()
    data = data.shuffle(buffer_size=10*n)
    data = data.map(lambda x, y, z: ((image_processing(x), image_processing(y)), z))

    return data

#***************************************************************************************************************************************************************

if __name__ == "__main__":
    INPUT = input("Train an Untrained Model or Futher Train the Trained Model? (u/t): ").strip().lower()
    if INPUT == 'u':
        model = tf.keras.models.load_model('untrained_faceid_model.keras',
                                        custom_objects={'L1Dist': L1Dist})
        
        print("****************Untrained Model Loaded*****************")

        data = data_processing()

        data = data.batch(32)
        data = data.prefetch(8)

        model.compile(optimizer=optimizers.Adam(1e-4),
                    loss=losses.BinaryCrossentropy(),
                    metrics=['accuracy','Precision','Recall'])
        
        M = int(input("Enter number of epochs to train the model: "))
        history = model.fit(data,epochs = M)

        name = input('Enter Model Name to Save (without .keras extension): ').strip()
        model.save(name+'.keras')
        print("Model Trained and Saved as ", name)

    elif INPUT == 't':
        name = input('Enter Trained Model Name to Load (without .keras extension): ').strip()

        model = tf.keras.models.load_model(name+'.keras',
                                        custom_objects={'L1Dist': L1Dist})
        

        print("****************",name," Model Loaded*****************")
        data = data_processing()

        data = data.batch(32)
        data = data.prefetch(8)

        model.compile(optimizer=optimizers.Adam(1e-4),
                    loss=losses.BinaryCrossentropy(),
                    metrics=['accuracy','Precision','Recall'])
        
        M = int(input("Enter number of epochs to further train the model: "))
        history = model.fit(data,epochs = M)

        model.save(name+'.keras')
        print("Model Further Trained and Saved")
    else:
        print("Invalid Input! Exiting...")