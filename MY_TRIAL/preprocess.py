import subprocess
subprocess.run(["rm", "-rf", "./processed/anchor", "./processed/positive", "./processed/negative"], check=True)
subprocess.run(["mkdir", "-p", "./processed/anchor", "./processed/positive", "./processed/negative"], check=True)

import os
import matplotlib.pyplot as plt
import cv2

def preprocess():
    for file in os.listdir('./captured/anchor'):
        img = cv2.imread(os.path.join('./captured/anchor', file))                
        if img is None:
            continue
        img = cv2.resize(img, (105, 105))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        cv2.imwrite(os.path.join('./processed/anchor', file), img)                   
    print("Anchor images processed")

    for file in os.listdir('./captured/positive'):
        img = cv2.imread(os.path.join('./captured/positive', file))                
        if img is None:
            continue
        img = cv2.resize(img, (105, 105))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join('./processed/positive', file), img)                   
    print("Positive images processed")

    for file in os.listdir('./captured/negative'):
        img = cv2.imread(os.path.join('./captured/negative', file))                
        if img is None:
            continue
        img = cv2.resize(img, (105, 105))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join('./processed/negative', file), img)                   
    print("Negative images processed")

    pass

if __name__ == "__main__":
    preprocess()
    print("Preprocessing completed.\n*******************************************************************************************************")