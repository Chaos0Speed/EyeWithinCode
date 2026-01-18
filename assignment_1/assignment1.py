import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

data = {'Name': ['Alice', 'Bob', 'Alice', 'Charlie'],
        'File': ['a1.jpg', 'b1.jpg', 'a2.jpg', 'c1.jpg']}

df = pd.DataFrame(data)

name  = input("Enter the name to search for: ")
name_data = df[df['Name'] == name]
if len(name_data) == 0:
    print("Name not found in the DataFrame.")
    exit()
n = 0
if len(name_data) != 1:
    n = int(input("Multiple entries found for the name. \nEnterwhich entry to use (0 to {}): ".format(len(name_data)-1)))

cpath = 'assignment_1/'+df[df['Name'] == name]['File'].iloc[n]

img_gray = cv2.imread(cpath, cv2.IMREAD_GRAYSCALE)

img_resized = cv2.resize(img_gray, (105, 105))

img_final = img_resized.astype(np.float32) / 255.0

plt.imshow(img_final, cmap='gray')
plt.title("Ready for the Network!")
plt.savefig("assignment_1/final.png")
print("Image processing complete and saved as final.png")

