from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import numpy as np
import tensorflow as tf
from layers import L1Dist
import os

class FaceIDApp(App):

    def build(self):

        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist})

        self.layout = BoxLayout(orientation='vertical')

        self.webcam = Image(size_hint=(1, .8))
        self.button = Button(text="Begin Verification", on_press=self.verify, size_hint=(1, .1))
        self.verification_label = Label(text="Verification Not Started", size_hint=(1, .1))

        self.layout.add_widget(self.verification_label)
        self.layout.add_widget(self.webcam)
        self.layout.add_widget(self.button)
        
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return self.layout

    def update(self, dt):
        ret, frame = self.capture.read()
        frame = frame[115:365,235:485, :]

        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.webcam.texture = texture

    def preprocess(self,file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (105,105))
        img = img / 255.0
        return img
    
    def verify(self, *args):   
    # Build results array
        detection_threshold = 0.5
        verification_threshold = 0.5
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[115:365,235:485, :]
        cv2.imwrite(SAVE_PATH, frame)

        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(SAVE_PATH)
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        detection = np.sum(np.array(results) > detection_threshold)

        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold

        self.verification_label.text = "Verified" if verified else "Unverified"

        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
        return results, verified
if __name__ == '__main__':
    FaceIDApp().run()