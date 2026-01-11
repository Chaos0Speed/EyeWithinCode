import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import layers, models
import tensorflow as tf
from sklearn.model_selection import train_test_split

def z_score_scaling(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)

training_data = pd.read_excel('Training data.xlsx')
training_data['internet'] = training_data['internet'].map({'no':1,'yes':0})
training_data['sex'] = training_data['sex'].map({'M':1,'F':-1})

x_train = training_data.iloc[:,0:8].copy()
y_train = training_data.iloc[:, 8].copy()

x_train = z_score_scaling(x_train)

if('model.keras') in tf.io.gfile.listdir('.'):
    model = models.load_model('model.keras')
    print('Loading Current Model')
else:
    print("Creating New Model")
    model = models.Sequential()
    model.add(layers.Dense(units=1, activation='linear', input_shape=(8,)))

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.99), loss='mse', metrics=['mae'])
model.fit(x_train, y_train, epochs=10, batch_size=1)
model.save('model.keras')
print(model.weights)

exec(open('2test.py').read())