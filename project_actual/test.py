
from tensorflow.keras.models import Model #type:ignore
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten #type:ignore
import tensorflow as tf
import numpy as np

# 1. Check what device we are running on
print("Visible Devices:", tf.config.get_visible_devices())

# 2. Create a dummy image (Batch Size: 1, Height: 28, Width: 28, Channels: 3)
# This simulates a small color image.
input_image = np.random.random((1, 28, 28, 3)).astype(np.float32)

inp = Input(shape=(105,105,3), name="image_input_layer")
#Block 1
c1 = Conv2D(64, (10,10), activation='relu')(inp)
mp1 = MaxPooling2D(64, (2,2), padding='same')(c1)
#Block 2
c2 = Conv2D(128, (7,7), activation='relu')(mp1)
mp2 = MaxPooling2D(64, (2,2), padding='same')(c2)
#Block 3
c3 = Conv2D(128, (4,4), activation='relu')(mp2)
mp3 = MaxPooling2D(64, (2,2), padding='same')(c3)
#Final Block
c4 = Conv2D(256, (4,4), activation='relu')(mp3)
f1 = Flatten()(c4)
d = Dense(4096, activation='sigmoid')(f1)

mod = Model(input=[inp] ,output=[d], name='embedding')
mod.summary()
