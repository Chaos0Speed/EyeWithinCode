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
from model_creation import L1Dist
import os
import threading 

class FaceIDApp(App):

    def build(self):
        # 1. Load Model
        self.model = tf.keras.models.load_model('Model_kartik.keras', custom_objects={'L1Dist': L1Dist})
        
        # 2. PRE-LOAD Validation Images (Huge Speedup)
        # We process these once at startup so we don't have to read them during verification
        self.validation_images = self.load_verification_images()

        self.layout = BoxLayout(orientation='vertical')
        self.webcam = Image(size_hint=(1, .8))
        self.button = Button(text="Begin Verification", on_press=self.verify_thread, size_hint=(1, .1))
        self.verification_label = Label(text="Verification Not Started", size_hint=(1, .1))

        self.layout.add_widget(self.verification_label)
        self.layout.add_widget(self.webcam)
        self.layout.add_widget(self.button)
        
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return self.layout

    def load_verification_images(self):
        """Loads and processes all verification images into a single numpy array."""
        images = []
        path = os.path.join('application_data', 'verification_images')
        for image_name in os.listdir(path):
            img_path = os.path.join(path, image_name)
            img = cv2.imread(img_path)
            if img is not None:
                # Reuse the memory-based preprocess logic
                processed = self.preprocess_frame(img)
                images.append(processed)
        
        # Stack them into a single block: (N, 105, 105, 1)
        if len(images) > 0:
            return np.vstack(images)
        return np.array([])

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Crop to focus on face (adjust as needed)
            frame = frame[200:450, 170:420, :]
            frame = cv2.flip(frame, 1)

            # Convert to texture for Kivy
            buf = cv2.flip(frame, 0).tobytes() # .tostring() is deprecated
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.webcam.texture = texture

    def preprocess_frame(self, frame):
        """Processes an image frame directly from memory (no disk I/O)."""
        # Resize
        img = cv2.resize(frame, (105, 105))
        # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Normalize (Crucial for model accuracy, usually /255.0)
        img = img / 255.0
        # Add dimensions to match model input: (1, 105, 105, 1)
        img = np.expand_dims(img, axis=0) # Batch dim
        img = np.expand_dims(img, axis=-1) # Channel dim
        return img
    
    def verify_thread(self, *args):
        # Run verify in a separate thread so the screen doesn't freeze
        t = threading.Thread(target=self.verify)
        t.start()

    def verify(self):   
        detection_threshold = 0.7
        verification_threshold = 0.7
        
        # 1. Capture current frame from webcam
        ret, frame = self.capture.read()
        if not ret:
            return
            
        frame = frame[200:450, 170:420, :]
        frame = cv2.flip(frame, 1)
        
        # 2. Process input directly from memory (Removed cv2.imwrite/imread)
        input_img = self.preprocess_frame(frame) # Shape: (1, 105, 105, 1)
        
        # 3. Batch Prediction (The Optimisation)
        # Instead of a for loop, we prepare one giant batch.
        
        # Repeat the input image N times to match the number of verification images
        num_verification_imgs = len(self.validation_images)
        if num_verification_imgs == 0:
            self.update_label("No Verification Images Found")
            return

        input_batch = np.repeat(input_img, num_verification_imgs, axis=0)
        
        # ONE call to model.predict instead of N calls
        # This sends all pairs to the GPU/CPU at once
        results = self.model.predict([input_batch, self.validation_images], verbose=0)
        
        # 4. Calculate Logic
        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / num_verification_imgs
        verified = verification > verification_threshold

        # Update UI (Must be scheduled on main thread)
        text = "Verified" if verified else "Unverified"
        Clock.schedule_once(lambda dt: self.update_label(text))

        Logger.info(f"Results: {detection}/{num_verification_imgs} passed. Verified: {verified}")
        
        return results, verified

    def update_label(self, text):
        self.verification_label.text = text

if __name__ == '__main__':
    FaceIDApp().run()